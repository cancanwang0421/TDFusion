import torch.nn
from torch import optim
from Edge_data import edge_data
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torchvision.models as models
from edge import SwinIR
import torch.nn.functional as F
from tqdm import tqdm

window_size = 8
train_dataset = edge_data('./dataset/train/LOL/our485')
train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=0, pin_memory=True)

save_path = './model_edge'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

edge_model = SwinIR(upscale=1, img_size=(64, 64),
                    window_size=window_size, img_range=1., depths=[1],
                    embed_dim=2, num_heads=[1], mlp_ratio=0.5, upsampler='pixelshuffledirect',
                    resi_connection=None)
edge_model = edge_model.cuda()


optimizer = optim.Adam(edge_model.parameters(), lr=0.001)


for epoch in range(0,200):
    edge_model.train()
    train_tqdm = tqdm(train_loader, total=len(train_loader))
    for normal_edge, low_image, _ in train_tqdm:
        normal_edge = normal_edge.cuda()
        low_image = low_image.cuda()
        low_edge = edge_model(low_image)



        optimizer.zero_grad()
        loss1 = torch.nn.BCELoss()
        loss = loss1(low_edge,normal_edge)

        train_tqdm.set_postfix(epoch=epoch,
                               loss_total=loss.item())
        loss.backward()
        optimizer.step()
    torch.save(edge_model.state_dict(), f'{save_path}/edge_epoch_{epoch}.pth')
