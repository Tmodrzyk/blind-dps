from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from PIL import Image
import seaborn as sns
import pandas as pd
import lpips


device = 'cuda:0'

task = 'gaussian_blur'
factor = 4
sigma = 0.1
scale = 1.0

sns.set_theme()

loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

def stats(label_root, recon_root):

    psnr_normal_list = []
    psnr_img_list = []
    ssim_img_list = []
    lpips_vgg_list = []
    lpips_alex_list = []

    res_dict = {}
    
    for idx in tqdm(range(1000)):
        fname = str(idx).zfill(5)

        label = Image.open(os.path.join(label_root, f'{fname}.png')).convert('RGB')
        label = np.asarray(label) 
        recon = Image.open(os.path.join(recon_root, f'{fname}.png')).convert('RGB')
        recon = np.asarray(recon) 

        label_tensor = torch.tensor(label.swapaxes(1, 2).swapaxes(0, 1)).unsqueeze(0).to(device)
        recon_tensor = torch.tensor(recon.swapaxes(1, 2).swapaxes(0, 1)).unsqueeze(0).to(device)
        
        # PSNR on the whole image
        
        psnr_img = peak_signal_noise_ratio(label, recon)
        
        # SSIM on the whole image
        
        ssim_img = structural_similarity(label, recon, channel_axis=2)

        # LPIPS VGG

        lpips_vgg = loss_fn_vgg(label_tensor, recon_tensor).item()
        
        # LPIPS AlexNet

        lpips_alex = loss_fn_alex(label_tensor, recon_tensor).item()
        
        psnr_img_list.append(psnr_img)
        ssim_img_list.append(ssim_img)
        lpips_vgg_list.append(lpips_vgg)
        lpips_alex_list.append(lpips_alex)
        
    res_dict['psnr_normal'] = psnr_normal_list
    res_dict['psnr_img'] = psnr_img_list
    res_dict['ssim_img'] = ssim_img_list
    res_dict['lpips_vgg'] = lpips_vgg_list
    res_dict['lpips_alex'] = lpips_alex_list

    return res_dict


label_root = Path(f'/home/modrzyk/code/data/EUSIPCO_2024/label/')
recon_root_ours = Path(f'/home/modrzyk/code/data/EUSIPCO_2024/recon-rl-diffusion-dps-refinement/')
recon_root_posterior_sampling = Path(f'/home/modrzyk/code/data/EUSIPCO_2024/recon-dps/')
recon_root_richarson_lucy = Path(f'/home/modrzyk/code/data/EUSIPCO_2024/recon-rl-tv/')
recon_root_pnp_admm = Path(f'/home/modrzyk/code/data/EUSIPCO_2024/recon-pnp-admm/')

res_ours = stats(label_root, recon_root_ours)
res_posterior_sampling = stats(label_root, recon_root_posterior_sampling)
res_richardson_lucy = stats(label_root, recon_root_richarson_lucy)
res_pnp_admm = stats(label_root, recon_root_pnp_admm)

# Calculate average

psnr_img_ours = np.mean(res_ours['psnr_img'])
ssim_img_ours = np.mean(res_ours['ssim_img'])
lpips_vgg_ours = np.mean(res_ours['lpips_vgg'])
lpips_alex_ours = np.mean(res_ours['lpips_alex'])

psnr_img_posterior_sampling = np.mean(res_posterior_sampling['psnr_img'])
ssim_img_posterior_sampling = np.mean(res_posterior_sampling['ssim_img'])
lpips_vgg_posterior_sampling = np.mean(res_posterior_sampling['lpips_vgg'])
lpips_alex_posterior_sampling = np.mean(res_posterior_sampling['lpips_alex'])

psnr_img_richardson_lucy = np.mean(res_richardson_lucy['psnr_img'])
ssim_img_richardson_lucy = np.mean(res_richardson_lucy['ssim_img'])
lpips_vgg_richardson_lucy = np.mean(res_richardson_lucy['lpips_vgg'])
lpips_alex_richardson_lucy = np.mean(res_richardson_lucy['lpips_alex'])

psnr_img_pnp_admm = np.mean(res_pnp_admm['psnr_img'])
ssim_img_pnp_admm = np.mean(res_pnp_admm['ssim_img'])
lpips_vgg_pnp_admm = np.mean(res_pnp_admm['lpips_vgg'])
lpips_alex_pnp_admm = np.mean(res_pnp_admm['lpips_alex'])

# Calculate standard deviation

psnr_img_std_ours = np.std(res_ours['psnr_img'])
ssim_img_std_ours = np.std(res_ours['ssim_img'])
lpips_vgg_std_ours = np.std(res_ours['lpips_vgg'])
lpips_alex_std_ours = np.std(res_ours['lpips_alex'])

psnr_img_std_posterior_sampling = np.std(res_posterior_sampling['psnr_img'])
ssim_img_std_posterior_sampling = np.std(res_posterior_sampling['ssim_img'])
lpips_vgg_std_posterior_sampling = np.std(res_posterior_sampling['lpips_vgg'])
lpips_alex_std_posterior_sampling = np.std(res_posterior_sampling['lpips_alex'])

psnr_img_std_richardson_lucy = np.std(res_richardson_lucy['psnr_img'])
ssim_img_std_richardson_lucy = np.std(res_richardson_lucy['ssim_img'])
lpips_vgg_std_richardson_lucy = np.std(res_richardson_lucy['lpips_vgg'])
lpips_alex_std_richardson_lucy = np.std(res_richardson_lucy['lpips_alex'])

psnr_img_std_pnp_admm = np.std(res_pnp_admm['psnr_img'])
ssim_img_std_pnp_admm = np.std(res_pnp_admm['ssim_img'])
lpips_vgg_std_pnp_admm = np.std(res_pnp_admm['lpips_vgg'])
lpips_alex_std_pnp_admm = np.std(res_pnp_admm['lpips_alex'])

print(f'PSNR Image Ours: {psnr_img_ours:.2f} ± {psnr_img_std_ours:.2f}')
print(f'SSIM Image Ours: {ssim_img_ours:.2f} ± {ssim_img_std_ours:.2f}')
print(f'LPIPS VGG Ours: {lpips_vgg_ours:.2f} ± {lpips_vgg_std_ours:.2f}')
print(f'LPIPS AlexNet Ours: {lpips_alex_ours:.2f} ± {lpips_alex_std_ours:.2f}')
print('---')
print(f'PSNR Image Posterior Sampling: {psnr_img_posterior_sampling:.2f} ± {psnr_img_std_posterior_sampling:.2f}')
print(f'SSIM Image Posterior Sampling: {ssim_img_posterior_sampling:.2f} ± {ssim_img_std_posterior_sampling:.2f}')
print(f'LPIPS VGG Posterior Sampling: {lpips_vgg_posterior_sampling:.2f} ± {lpips_vgg_std_posterior_sampling:.2f}')
print(f'LPIPS AlexNet Posterior Sampling: {lpips_alex_posterior_sampling:.2f} ± {lpips_alex_std_posterior_sampling:.2f}')
print('---')
print(f'PSNR Image Richardson Lucy: {psnr_img_richardson_lucy:.2f} ± {psnr_img_std_richardson_lucy:.2f}')
print(f'SSIM Image Richardson Lucy: {ssim_img_richardson_lucy:.2f} ± {ssim_img_std_richardson_lucy:.2f}')
print(f'LPIPS VGG Richardson Lucy: {lpips_vgg_richardson_lucy:.2f} ± {lpips_vgg_std_richardson_lucy:.2f}')
print(f'LPIPS AlexNet Richardson Lucy: {lpips_alex_richardson_lucy:.2f} ± {lpips_alex_std_richardson_lucy:.2f}')
print('---')
print(f'PSNR Image PNP ADMM: {psnr_img_pnp_admm:.2f} ± {psnr_img_std_pnp_admm:.2f}')
print(f'SSIM Image PNP ADMM: {ssim_img_pnp_admm:.2f} ± {ssim_img_std_pnp_admm:.2f}')
print(f'LPIPS VGG PNP ADMM: {lpips_vgg_pnp_admm:.2f} ± {lpips_vgg_std_pnp_admm:.2f}')
print(f'LPIPS AlexNet PNP ADMM: {lpips_alex_pnp_admm:.2f} ± {lpips_alex_std_pnp_admm:.2f}')