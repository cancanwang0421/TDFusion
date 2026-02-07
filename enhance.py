
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


def lrelu(x, leak=0.2):
    return torch.max(x, leak * x)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1) 
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.trunc_normal_(layer.weight, std=1e-3)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = lrelu(x)
        x = self.bn2(self.conv2(x))
        x = lrelu(x)
        x = self.bn3(self.conv3(x))
        x = lrelu(x)
        x = self.bn4(self.conv4(x))
        x = lrelu(x)
        return x



class CAM_VI_E(nn.Module):
    def __init__(self):
        super(CAM_VI_E, self).__init__()
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        
        nn.init.trunc_normal_(self.conv1.weight, std=1e-3)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.trunc_normal_(self.conv2.weight, std=1e-3)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = lrelu(x)
        x = self.bn2(self.conv2(x))
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = F.softmax(x, dim=1)
        return x


class CAM_L(nn.Module):
    def __init__(self):
        super(CAM_L, self).__init__()
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        
        nn.init.trunc_normal_(self.conv1.weight, std=1e-3)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.trunc_normal_(self.conv2.weight, std=1e-3)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = lrelu(x)
        x = self.bn2(self.conv2(x))
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = F.softmax(x, dim=1)
        return x


class Decoder_VI_E(nn.Module):
    def __init__(self):
        super(Decoder_VI_E, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-5, momentum=0.1)
        self.conv4 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1)
        

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.trunc_normal_(layer.weight, std=1e-3)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = lrelu(x)
        x = self.bn2(self.conv2(x))
        x = lrelu(x)
        x = self.bn3(self.conv3(x))
        x = lrelu(x)
        x = self.bn4(self.conv4(x))
        x = torch.sigmoid(x)
        return x



class Decoder_VI_L(nn.Module):
    def __init__(self):
        super(Decoder_VI_L, self).__init__()

        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1)
        

        self.l_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.l_bn1 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.l_conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.l_bn2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.l_conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.l_bn3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.l_conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.l_bn4 = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1)
        
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4,
                      self.l_conv1, self.l_conv2, self.l_conv3, self.l_conv4]:
            nn.init.trunc_normal_(layer.weight, std=1e-3)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, feature_vi_e, feature_l):
        x = self.bn1(self.conv1(feature_vi_e))
        x = lrelu(x)
        x2 = self.bn2(self.conv2(x))
        x2 = lrelu(x2)
        x3 = self.bn3(self.conv3(x2))
        x3 = lrelu(x3)
        vi_e_r = self.bn4(self.conv4(x3))
        vi_e_r = torch.sigmoid(vi_e_r)
        
        l_x = self.l_bn1(self.l_conv1(feature_l))
        l_x = lrelu(l_x)
        l_x = torch.cat([l_x, x], dim=1)
        l_x = self.l_bn2(self.l_conv2(l_x))
        l_x = lrelu(l_x)
        l_x = self.l_bn3(self.l_conv3(l_x))
        l_x = lrelu(l_x)
        l_x = torch.cat([l_x, x3], dim=1)
        l_r = self.l_bn4(self.l_conv4(l_x))
        l_r = torch.sigmoid(l_r)
        
        return vi_e_r, l_r


class enhance_d(nn.Module):
    def __init__(self):
        super(enhance_d, self).__init__()
        self.encoder = Encoder()
        self.cam_vi_e = CAM_VI_E()
        self.cam_l = CAM_L()
        self.decoder_vi_l = Decoder_VI_L()

    def forward(self, vi):
        feature = self.encoder(vi)
        
        vector_vi_e = self.cam_vi_e(feature)
        feature_vi_e = torch.multiply(vector_vi_e, feature)
        vector_l = self.cam_l(feature)
        feature_l = torch.multiply(vector_l, feature)
        
        vi_e_r, l_r = self.decoder_vi_l(feature_vi_e, feature_l)
        
        return feature_vi_e, vi_e_r, l_r

