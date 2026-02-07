import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
from enhance import enhance_d

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


def W(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Attention(nn.Module):
    def __init__(self, in_channels=16, num_head=1, ratio=1):  # 通道数调整为8
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.out_channel = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.W = W(int(in_channels * ratio), in_channels)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 简化融合

    def forward(self, key, query):
        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channel)
        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
        else:
            softmax = F.softmax(att, dim=1).unsqueeze(dim=2)
        weighted_value = v_out * softmax
        weighted_value = weighted_value.sum(dim=1)
        out = self.W(weighted_value)
        return self.fuse(torch.cat([key, out], dim=1))

class encoder_ir(nn.Module):
    def __init__(self):
        super(encoder_ir, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)  # 直接输出8通道
        self.lrelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        return x

class dense_ir(nn.Module):
    def __init__(self):
        super(dense_ir, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(192, 64, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.lrelu(x1)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x2 = self.lrelu(x2)
        x3 = self.conv3(torch.cat([torch.cat([x, x1], dim=1), x2], dim=1))
        x3 = self.lrelu(x3)
        return torch.cat([torch.cat([torch.cat([x, x1], dim=1), x2], dim=1), x3], dim=1)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class Fusion_res(torch.nn.Module):
    def __init__(self, channels):
        super(Fusion_res, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.conv3 = ConvLayer(channels, channels, 3, 1)
        self.conv4 = ConvLayer(channels, channels, 3, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.conv4(x)
        out = x1 + x2
        return out


class shallow_inject(nn.Module):
    def __init__(self):
        super(shallow_inject, self).__init__()
        self.con1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.text_mod = FeatureWiseAffine(
            in_channels=512,
            out_channels=64,
            use_affine_level=True
        )

    def forward(self, x, text_features):
        x1 = self.con1(x)
        x1 = self.text_mod(x1, text_features)
        return x1

class deep_inject(nn.Module):
    def __init__(self):
        super(deep_inject, self).__init__()
        self.con1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.text_mod = FeatureWiseAffine(
            in_channels=512,
            out_channels=64,
            use_affine_level=True
        )

    def forward(self, x, text_features):
        x1 = self.con1(x)
        x1 = self.text_mod(x1, text_features)
        return x1

        
class decoder_fusion(nn.Module):
    def __init__(self):
        super(decoder_fusion, self).__init__()
        self.con2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.con3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x = self.con2(x)
        x = self.lrelu(x)
        x = self.con3(x)
        x = torch.sigmoid(x)
        return x

## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x
        

class fusion(nn.Module):
    def __init__(self, model_clip):
        super(fusion, self).__init__()
        self.model_clip = model_clip
        self.model_clip.eval()

        
        self.ir_encoder = encoder_ir()
        self.shallow = shallow_inject()
        self.deep = deep_inject()
        self.attention = Attention(64)
        self.adjust_vi = nn.Conv2d(512, 128, 1, 1, 0)
        

        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)

        
        self.decoder = decoder_fusion()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0.0)

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


    def forward(self, x_ir, fea_vi, edge_shallow, edge_deep, text):
        b = x_ir.shape[0]
        

        text_features = self.get_text_feature(text.expand(b, -1)).to(x_ir.dtype)
        

        fea_ir = self.ir_encoder(x_ir)  # [B, 8, H, W]
        

        fea_vi = self.adjust_vi(fea_vi)  # 从 [B, 32, H, W] 调整到 [B, 8, H, W]
        

        fea_cat = torch.cat([fea_vi, fea_ir], dim=1)  # [B, 16, H, W]
        fea_conv1 = self.lrelu(self.conv1(fea_cat))  # [B, 16, H, W]
        fea_conv2 = self.lrelu(self.conv2(fea_conv1))  # [B, 16, H, W]
        fea_cat_12 = torch.cat([fea_conv1, fea_conv2], dim=1)  # [B, 32, H, W]
        fea_conv3 = self.lrelu(self.conv3(fea_cat_12))  # [B, 16, H, W]
        fea_cat_123 = torch.cat([fea_conv1, fea_conv2, fea_conv3], dim=1)  # [B, 48, H, W]
        fea_conv4 = self.lrelu(self.conv4(fea_cat_123))  # [B, 16, H, W]
        fea_conv5 = self.lrelu(self.conv5(fea_conv4))  # [B, 8, H, W]
        

        edge_s_1 = self.shallow(edge_shallow, text_features)  # [B, 8, H, W]
        fea_fusion = fea_conv5 * edge_s_1
        

        edge_deep = self.deep(edge_deep, text_features)  # [B, 8, H, W]
        fea_fusion = self.attention(fea_fusion, edge_deep)  # [B, 8, H, W]
        

        I_fusion = self.decoder(fea_fusion)  # [B, 1, H, W]
        
        return I_fusion

