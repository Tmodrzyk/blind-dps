from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from PIL import Image

device = 'cuda:0'

task = 'blind_blur'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'./results/{task}/label/')

normal_recon_root = Path(f'./results/{task}/recon/')

psnr_normal_list = []
crc_list = []

for idx in tqdm(range(1)):
    fname = 'img_' + str(idx).zfill(5)

    label = Image.open(os.path.join(label_root, f'{fname}.png')).convert("L")
    label = np.asarray(label)
    normal_recon = Image.open(os.path.join(normal_recon_root, f'{fname}.png')).convert("L")
    normal_recon = np.asarray(normal_recon)
    
    # Apply a simple threshold to create a binary mask
    _, binary_mask = cv2.threshold(label, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crc_component_list = []
    
    # Iterate over each contour
    for contour in contours:
        # Create a mask for each contour
        mask = np.zeros_like(label)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

        # Extract the sub-region from both label and normal_recon using the mask
        label_component = cv2.bitwise_and(label, label, mask=mask)
        normal_recon_component = cv2.bitwise_and(normal_recon, normal_recon, mask=mask)

        # PSNR
        psnr_component = peak_signal_noise_ratio(label_component, normal_recon_component)
        psnr_normal_list.append(psnr_component)
        
        # Concentration Recovery Coefficient
        crc_component = normal_recon_component.sum() / label_component.sum()
        crc_component_list.append(crc_component)

    crc_image = sum(crc_component_list) / len(crc_component_list)
    crc_list.append(crc_image)
    psnr_normal = peak_signal_noise_ratio(label, normal_recon)
    psnr_normal_list.append(psnr_normal)
    
    print(f'Normal PSNR {fname}: {psnr_normal}')
    print(f'CRC {fname} : {crc_image}')
    
psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
crc_avg = sum(crc_list) / len(crc_list)

print(f'Normal PSNR average: {psnr_normal_avg}')
print(f'CRC average: {crc_avg}')
