
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from LLVIP_data import LLVIP_data
from enhance import enhance_d
from edge import SwinIR
from fusion import fusion
import torch.nn.functional as F
from glob import glob
import numpy as np
import clip 

def rgb_ycbcr_pytorch(img_rgb):
    R = img_rgb[:, 0:1, :, :]  # [batch, 1, H, W]
    G = img_rgb[:, 1:2, :, :]  # [batch, 1, H, W]
    B = img_rgb[:, 2:3, :, :]  # [batch, 1, H, W]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = torch.cat([Y, Cb, Cr], dim=1)  # [batch, 3, H, W]
    return img_ycbcr

def ycbcr_rgb_pytorch(img_ycbcr):
    Y = img_ycbcr[:, 0:1, :, :]  # [batch, 1, H, W]
    Cb = img_ycbcr[:, 1:2, :, :]  # [batch, 1, H, W]
    Cr = img_ycbcr[:, 2:3, :, :]  # [batch, 1, H, W]
    R = Y + 1.402 * (Cr - 128/255)
    G = Y - 0.34414 * (Cb - 128/255) - 0.71414 * (Cr - 128/255)
    B = Y + 1.772 * (Cb - 128/255)
    img_rgb = torch.cat([R, G, B], dim=1)  # [batch, 3, H, W]
    return torch.clamp(img_rgb, 0.0, 1.0) 
    # return img_rgb


def get_if_tensor_pytorch(Yf, vi_3):
    vi_ycbcr = rgb_ycbcr_pytorch(vi_3)  # [batch, 3, H, W]
    cb = vi_ycbcr[:, 1:2, :, :]  # [batch, 1, H, W]
    cr = vi_ycbcr[:, 2:3, :, :]  # [batch, 1, H, W]
    If = torch.cat([Yf, cb, cr], dim=1)  # [batch, 3, H, W]
    If = ycbcr_rgb_pytorch(If)  # [batch, 3, H, W]
    return If


def save_images(filepath, result_1, result_2=None, result_3=None):
    if isinstance(result_1, torch.Tensor):
        result_1 = result_1.cpu().detach().numpy()
    if result_2 is not None and isinstance(result_2, torch.Tensor):
        result_2 = result_2.cpu().detach().numpy()
    if result_3 is not None and isinstance(result_3, torch.Tensor):
        result_3 = result_3.cpu().detach().numpy()
    

    result_1 = np.transpose(result_1.squeeze(axis=0), (1, 2, 0))  # [H, W, 3]
    
    if result_2 is not None:

        result_2 = np.repeat(result_2.squeeze(axis=(0, 1))[:, :, np.newaxis], 3, axis=2)  # [H, W, 1] -> [H, W, 3]
    
    if result_3 is not None:

        result_3 = np.repeat(result_3.squeeze(axis=(0, 1))[:, :, np.newaxis], 3, axis=2)  # [H, W, 1] -> [H, W, 3]
    
    if result_2 is None or not np.any(result_2):
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)
    
    if result_3 is not None and np.any(result_3):
        cat_image = np.concatenate([cat_image, result_3], axis=1)
    
    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'jpg')



def load_images(file, grayscale=False):
    im = Image.open(file)
    if grayscale:
        im = im.convert('L')
    img = np.array(im, dtype="float32") / 255.0
    img_norm = np.float32(img)
    return img_norm


def read_image_text(text_path):
    if not os.path.exists(text_path):
        print(f"警告：图像对应的文本文件不存在：{text_path}，使用默认文本")
        return "Infrared-visible image fusion, retain all details."
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text if text else "Infrared-visible image fusion, retain all details."

def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_clip, _ = clip.load("ViT-B/32", device=device)
model_clip.eval()
for param in model_clip.parameters():
    param.requires_grad = False


save_path = './result10.0-edge-every-maxplus-rain-train-950'
if not os.path.exists(save_path):
    os.makedirs(save_path)


checkpoint_dir = './checkpointtest10.0-edge-every-maxplus-5/end_to_end/'
ckpt_path = os.path.join(checkpoint_dir, 'model_epoch_950.ckpt')


edge_model = SwinIR(
    upscale=1,
    img_size=(64, 64), 
    window_size=8,
    img_range=1.,
    depths=[7, 7],      
    embed_dim=64,       
    num_heads=[8, 8],   
    mlp_ratio=3.8,      
    upsampler='pixelshuffledirect'
).to(device)

enhance_model = enhance_d().to(device)
fusion_model = fusion(model_clip).to(device)

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    enhance_model.load_state_dict(ckpt['enhance_state_dict'])
    edge_model.load_state_dict(ckpt['edge_state_dict'])
    fusion_model.load_state_dict(ckpt['fusion_state_dict'])
    print(f'Loaded checkpoint from {ckpt_path}')
else:
    raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")

edge_model.eval()
enhance_model.eval()
fusion_model.eval()



test_dataset = LLVIP_data('./dataset/test/vis_Rain')
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True
)

eval_vi_data_names = glob('./dataset/test/vis_Rain/visible/*.jpg')

eval_vi_data_names.sort()
print(f"Number of evaluation images: {len(eval_vi_data_names)}")

name_to_path = {os.path.basename(path): path for path in eval_vi_data_names}


test_tqdm = tqdm(test_loader, total=len(test_loader))
with torch.no_grad():
    for idx, (Inf, vi, name) in enumerate(test_tqdm):
        img_name = name[0]
        if img_name not in name_to_path:
            print(f"Warning: Image {img_name} not found in evaluation data")
            continue
            
        img_path = name_to_path[img_name]
        
        im_before_vi = load_images(img_path, grayscale=False)
        vi_y = rgb_ycbcr_np(im_before_vi)[:, :, 0]
        original_h, original_w = vi_y.shape
        
        Inf = Inf.to(device)
        vi = vi.to(device)

        vi_eval = vi_y[:, :, np.newaxis]
        vi_eval_tensor = torch.from_numpy(vi_eval).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        vi_3_eval = im_before_vi
        vi_3_eval_tensor = torch.from_numpy(vi_3_eval).permute(2, 0, 1).float().unsqueeze(0).to(device)


        img_basename = os.path.splitext(img_name)[0]
        text_path = os.path.join('/root/autodl-tmp/SL/dataset/train/EMS_full_dataset1/train/vis_Rain/GT', f"{img_basename}.txt")
        image_text = read_image_text(text_path)
        text_tokens = clip.tokenize([image_text]).to(device)
        test_tqdm.set_postfix(
            img_name=img_name,
            text=image_text[:30] + "..." if len(image_text) > 30 else image_text
        )

        fea_vi, vi_e, _ = enhance_model(vi_eval_tensor)
        edge = edge_model(vi_eval_tensor)
        edge = F.interpolate(edge, size=(original_h, original_w), mode='bilinear', align_corners=False)
        
        I_fusion = fusion_model(Inf, fea_vi, edge, edge, text_tokens)
        I_fusion_rgb = get_if_tensor_pytorch(I_fusion, vi_3_eval_tensor)

        I_fusion_rgb = I_fusion_rgb.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(I_fusion_rgb)
        output_image.save(f'{save_path}/{img_name}')
        
        del vi_eval, vi_eval_tensor, vi_3_eval, vi_3_eval_tensor, I_fusion, I_fusion_rgb, text_tokens
        if device.type == 'cuda':
            torch.cuda.empty_cache()

print("[*] Testing finished.")

