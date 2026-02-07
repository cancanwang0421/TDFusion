import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class VGG16Features(nn.Module):
    def __init__(self, local_weights_path=None):
        super(VGG16Features, self).__init__()
        if local_weights_path:
            vgg = models.vgg16(pretrained=False)
            vgg.load_state_dict(torch.load(local_weights_path))
        else:
            vgg = models.vgg16(pretrained=True)
        self.slice1 = nn.Sequential(*list(vgg.features.children())[:4])   # 到conv1_2
        self.slice2 = nn.Sequential(*list(vgg.features.children())[4:9])  # 到conv2_2
        self.slice3 = nn.Sequential(*list(vgg.features.children())[9:16]) # 到conv3_3
        self.slice4 = nn.Sequential(*list(vgg.features.children())[16:23])# 到conv4_3
        self.slice5 = nn.Sequential(*list(vgg.features.children())[23:24])# 到conv5_1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        conv1_2 = x
        x = self.slice2(x)
        conv2_2 = x
        x = self.slice3(x)
        conv3_3 = x
        x = self.slice4(x)
        conv4_3 = x
        x = self.slice5(x)
        conv5_1 = x
        return {
            'conv1_2': conv1_2,
            'conv2_2': conv2_2,
            'conv3_3': conv3_3,
            'conv4_3': conv4_3,
            'conv5_1': conv5_1
        }

def preprocess_image(image):
    if image.dim() == 3:  # [3, H, W] -> [1, 3, H, W]
        image = image.unsqueeze(0)
    return image


def get_feature(image, layer, vgg_model):
    device = next(vgg_model.parameters()).device  # Get the model's device
    image = preprocess_image(image).to(device)    # Move image to the same device
    features = vgg_model(image)                   # Compute features
    return features[layer]                        # Return the specified layer's features

def Perceptual_Loss(original_image, generate_image, vgg_model=None, device='cuda'):
    if vgg_model is None:
        vgg_model = VGG16Features().to(device)
    original_conv5_1 = get_feature(original_image, 'conv5_1', vgg_model)
    generate_conv5_1 = get_feature(generate_image, 'conv5_1', vgg_model)
    conv5_1_loss = F.l1_loss(original_conv5_1, generate_conv5_1)
    return conv5_1_loss

def TV_Loss(batchimg):
    if batchimg.dim() == 3:  # [C, H, W] -> [1, C, H, W]
        batchimg = batchimg.unsqueeze(0)
    tv_norm = torch.sum(torch.abs(batchimg), dim=[1, 2, 3])
    return torch.mean(tv_norm)

