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

device = 'cuda:0'

task = 'gaussian_blur'
factor = 4
sigma = 0.1
scale = 1.0

sns.set_theme()

def stats(label_root, recon_root):

    psnr_normal_list = []
    crc_list = []
    psnr_img_list = []
    ssim_img_list = []

    for idx in tqdm(range(10)):
        fname = str(idx).zfill(5)

        label = Image.open(os.path.join(label_root, f'{fname}.png')).convert("L")
        label = np.asarray(label)
        recon = Image.open(os.path.join(recon_root, f'{fname}.png')).convert("L")
        recon = np.asarray(recon)
        
        # PSNR on the whole image
        
        psnr_imr = peak_signal_noise_ratio(label, recon)
        psnr_img_list.append(psnr_imr)
        
        # SSIM on the whole image
        
        ssim_img = structural_similarity(label, recon)
        ssim_img_list.append(ssim_img)
        
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
            normal_recon_component = cv2.bitwise_and(recon, recon, mask=mask)

            # PSNR
            psnr_component = peak_signal_noise_ratio(label_component, normal_recon_component)
            psnr_normal_list.append(psnr_component)
            
            # Concentration Recovery Coefficient
            crc_component = normal_recon_component.sum() / label_component.sum()
            crc_component_list.append(crc_component)

        crc_image = sum(crc_component_list) / len(crc_component_list)
        crc_list.append(crc_image)
        psnr_normal = peak_signal_noise_ratio(label, recon)
        psnr_normal_list.append(psnr_normal)
    
    return psnr_normal_list, crc_list, psnr_img_list, ssim_img_list


label_root = Path(f'/home/modrzyk/code/blind-dps/results/ellipse/hybrid/{task}/label/')
recon_root_ours = Path(f'/home/modrzyk/code/blind-dps/results/ellipse/hybrid/{task}/recon/')
recon_root_posterior_sampling = Path(f'/home/modrzyk/code/diffusion-posterior-sampling/results/{task}/recon/')

psnr_normal_list_ours, crc_list_ours, \
psnr_img_list_ours, ssim_img_list_ours = stats(label_root, recon_root_ours)

psnr_normal_list_ours_posterior_sampling, crc_list_posterior_sampling, \
psnr_img_list_ours_posterior_sampling, ssim_img_list_ours_posterior_sampling \
    = stats(label_root, recon_root_posterior_sampling)

# Calculate average
psnr_normal_avg_ours = sum(psnr_normal_list_ours) / len(psnr_normal_list_ours)
crc_avg_ours = sum(crc_list_posterior_sampling) / len(crc_list_posterior_sampling)
psnr_img_avg_ours = sum(psnr_img_list_ours_posterior_sampling) / len(psnr_img_list_ours_posterior_sampling)
ssim_img_avg_ours = sum(ssim_img_list_ours_posterior_sampling) / len(ssim_img_list_ours_posterior_sampling)

psnr_normal_avg_posterior_sampling = sum(psnr_normal_list_ours_posterior_sampling) / len(psnr_normal_list_ours_posterior_sampling)
crc_avg_posterior_sampling = sum(crc_list_posterior_sampling) / len(crc_list_posterior_sampling)
psnr_img_avg_posterior_sampling = sum(psnr_img_list_ours_posterior_sampling) / len(psnr_img_list_ours_posterior_sampling)
ssim_img_avg_posterior_sampling = sum(ssim_img_list_ours_posterior_sampling) / len(ssim_img_list_ours_posterior_sampling)

psnr_normal_avg_ours_posterior_sampling = sum(psnr_normal_list_ours_posterior_sampling) / len(psnr_normal_list_ours_posterior_sampling)
crc_avg_ours_posterior_sampling = sum(crc_list_posterior_sampling) / len(crc_list_posterior_sampling)
psnr_img_avg_ours_posterior_sampling = sum(psnr_img_list_ours_posterior_sampling) / len(psnr_img_list_ours_posterior_sampling)
ssim_img_avg_ours_posterior_sampling = sum(ssim_img_list_ours_posterior_sampling) / len(ssim_img_list_ours_posterior_sampling)

# Calculate standard deviation
psnr_normal_std_ours = np.std(psnr_normal_list_ours)
crc_std_ours = np.std(crc_list_ours)
psnr_img_std_ours = np.std(psnr_img_list_ours)
ssim_img_std_ours = np.std(ssim_img_list_ours)

psnr_normal_std_posterior_sampling = np.std(psnr_normal_list_ours_posterior_sampling)
crc_std_posterior_sampling = np.std(crc_list_posterior_sampling)
psnr_img_std_posterior_sampling = np.std(psnr_img_list_ours_posterior_sampling)
ssim_img_std_posterior_sampling = np.std(ssim_img_list_ours_posterior_sampling)

# Define the metrics and their averages
metrics = ['PSNR Normal', 'CRC', 'PSNR Image', 'SSIM Image']
averages_ours = [psnr_normal_avg_ours, crc_avg_ours, psnr_img_avg_ours, ssim_img_avg_ours]
averages_posterior_sampling = [psnr_normal_avg_ours_posterior_sampling, crc_avg_ours_posterior_sampling, psnr_img_avg_ours_posterior_sampling, ssim_img_avg_ours_posterior_sampling]
std_devs_ours = [psnr_normal_std_ours, crc_std_ours, psnr_img_std_ours, ssim_img_std_ours]
std_devs_posterior_sampling = [psnr_normal_std_posterior_sampling, crc_std_posterior_sampling, psnr_img_std_posterior_sampling, ssim_img_std_posterior_sampling]

x = np.arange(len(metrics))
bar_width = 0.35

plt.figure(figsize=(10, 8))
# Plot the bar plot
plt.bar(x + bar_width/2, averages_ours, bar_width, yerr=std_devs_ours, capsize=2, label='Ours')
plt.bar(x - bar_width/2, averages_posterior_sampling, bar_width, yerr=std_devs_posterior_sampling, capsize=2, label='Posterior Sampling')

plt.xticks(x, metrics)
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Metrics Comparison')
plt.legend()

plt.show()
