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

    for idx in tqdm(range(200)):
        fname = str(idx).zfill(5)

        label = Image.open(os.path.join(label_root, f'{fname}.png')).convert("L")
        label = np.asarray(label) 
        recon = Image.open(os.path.join(recon_root, f'{fname}.png')).convert("L")
        recon = np.asarray(recon) 
        

        
        # Apply a simple threshold to create a binary mask
        _, binary_mask = cv2.threshold(label, 0, 50, cv2.THRESH_BINARY)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crc_component_list = []
        area_component_list = []
        activity_label_img = 0
        activity_recon_img = 0
        
        # Iterate over each contour
        for contour in contours:
            # Create a mask for each contour
            
            mask = np.zeros_like(label)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            area = np.count_nonzero(mask)

            # Extract the sub-region from both label and normal_recon using the mask
            
            label_component = cv2.bitwise_and(label, label, mask=mask)
            normal_recon_component = cv2.bitwise_and(recon, recon, mask=mask)
            
            # PSNR
            psnr_component = peak_signal_noise_ratio(label_component, normal_recon_component)
            psnr_normal_list.append(psnr_component)
            
            # Activities
            
            activity_label_component = label_component.mean()
            activity_recon_component = normal_recon_component.mean()
            
            activity_label_img += activity_label_component
            activity_recon_img += activity_recon_component

            crc_component = activity_recon_component / activity_label_component
            crc_err_component = np.abs(1 - crc_component)
            
            crc_component_list.append(crc_err_component)
            area_component_list.append(area)
            
        # Concentration Recovery Coefficient 
        
        crc_img = activity_recon_img / activity_label_img
        crc_err = np.abs(1 - crc_img)

        # PSNR on the whole image
        
        psnr_img = peak_signal_noise_ratio(label, recon)
        
        # SSIM on the whole image
        
        ssim_img = structural_similarity(label, recon)

        crc_list.append(crc_err)
        psnr_img_list.append(psnr_img)
        ssim_img_list.append(ssim_img)
    
    plt.scatter(area_component_list, crc_component_list)
    plt.xlabel('Area')
    plt.ylabel('CRC Error')
    plt.title('CRC Error vs. Area')
    plt.grid(True)
    plt.show()

    return psnr_normal_list, crc_list, psnr_img_list, ssim_img_list


label_root = Path(f'/home/modrzyk/code/blind-dps/results/ellipse/hybrid_N30_M20/{task}/label/')
recon_root_ours = Path(f'/home/modrzyk/code/blind-dps/results/ellipse/hybrid_N30_M20/{task}/recon/')
recon_root_posterior_sampling = Path(f'/home/modrzyk/code/diffusion-posterior-sampling/results/{task}/recon/')
recon_root_richarson_lucy = Path(f'/home/modrzyk/code/blind-dps/results/ellipse/richardson-lucy/recon/')

psnr_normal_list_ours, crc_list_ours, \
psnr_img_list_ours, ssim_img_list_ours = stats(label_root, recon_root_ours)

psnr_normal_list_posterior_sampling, crc_list_posterior_sampling, \
psnr_img_list_posterior_sampling, ssim_img_list_posterior_sampling \
    = stats(label_root, recon_root_posterior_sampling)

psnr_normal_list_richardson_lucy, crc_list_richardson_lucy, \
psnr_img_list_richardson_lucy, ssim_img_list_richardson_lucy \
    = stats(label_root, recon_root_richarson_lucy)
    
# Calculate average
psnr_normal_avg_ours = sum(psnr_normal_list_ours) / len(psnr_normal_list_ours)
crc_avg_ours = sum(crc_list_ours) / len(crc_list_ours)
psnr_img_ours = sum(psnr_img_list_ours) / len(psnr_img_list_ours)
ssim_img_ours = sum(ssim_img_list_ours) / len(ssim_img_list_ours)

psnr_normal_avg_ours_posterior_sampling = sum(psnr_normal_list_posterior_sampling) / len(psnr_normal_list_posterior_sampling)
crc_avg_posterior_sampling = sum(crc_list_posterior_sampling) / len(crc_list_posterior_sampling)
psnr_img_posterior_sampling = sum(psnr_img_list_posterior_sampling) / len(psnr_img_list_posterior_sampling)
ssim_img_posterior_sampling = sum(ssim_img_list_posterior_sampling) / len(ssim_img_list_posterior_sampling)

psnr_normal_avg_richardson_lucy = sum(psnr_normal_list_richardson_lucy) / len(psnr_normal_list_richardson_lucy)
crc_avg_richardson_lucy = sum(crc_list_richardson_lucy) / len(crc_list_richardson_lucy)
psnr_img_richardson_lucy = sum(psnr_img_list_richardson_lucy) / len(psnr_img_list_richardson_lucy)
ssim_img_richardson_lucy = sum(ssim_img_list_richardson_lucy) / len(ssim_img_list_richardson_lucy)

# Calculate standard deviation
psnr_normal_std_ours = np.std(psnr_normal_list_ours)
crc_std_ours = np.std(crc_list_ours)
psnr_img_std_ours = np.std(psnr_img_list_ours)
ssim_img_std_ours = np.std(ssim_img_list_ours)

psnr_normal_std_posterior_sampling = np.std(psnr_normal_list_posterior_sampling)
crc_std_posterior_sampling = np.std(crc_list_posterior_sampling)
psnr_img_std_posterior_sampling = np.std(psnr_img_list_posterior_sampling)
ssim_img_std_posterior_sampling = np.std(ssim_img_list_posterior_sampling)

psnr_normal_std_richardson_lucy = np.std(psnr_normal_list_richardson_lucy)
crc_std_richardson_lucy = np.std(crc_list_richardson_lucy)
psnr_img_std_richardson_lucy = np.std(psnr_img_list_richardson_lucy)
ssim_img_std_richardson_lucy = np.std(ssim_img_list_richardson_lucy)

# # Define the metrics and their averages
# metrics = ['PSNR Normal', 'CRC', 'PSNR Image', 'SSIM Image']
# averages_ours = [psnr_normal_avg_ours, crc_avg_ours, psnr_img_avg_ours, ssim_img_avg_ours]
# averages_posterior_sampling = [psnr_normal_avg_ours_posterior_sampling, crc_avg_ours_posterior_sampling, psnr_img_avg_ours_posterior_sampling, ssim_img_avg_ours_posterior_sampling]
# std_devs_ours = [psnr_normal_std_ours, crc_std_ours, psnr_img_std_ours, ssim_img_std_ours]
# std_devs_posterior_sampling = [psnr_normal_std_posterior_sampling, crc_std_posterior_sampling, psnr_img_std_posterior_sampling, ssim_img_std_posterior_sampling]

# x = np.arange(len(metrics))
# bar_width = 0.35

# plt.figure(figsize=(10, 8))
# # Plot the bar plot
# plt.bar(x + bar_width/2, averages_ours, bar_width, yerr=std_devs_ours, capsize=2, label='Ours')
# plt.bar(x - bar_width/2, averages_posterior_sampling, bar_width, yerr=std_devs_posterior_sampling, capsize=2, label='Posterior Sampling')

# plt.xticks(x, metrics)
# plt.xlabel('Metrics')
# plt.ylabel('Scores')
# plt.title('Metrics Comparison')
# plt.legend()

# plt.show()




print(f'PSNR Normal Ours: {psnr_normal_avg_ours:.2f} ± {psnr_normal_std_ours:.2f}')
print(f'CRC Error Ours: {crc_avg_ours:.2f} ± {crc_std_ours:.2f}')
print(f'PSNR Image Ours: {psnr_img_ours:.2f} ± {psnr_img_std_ours:.2f}')
print(f'SSIM Image Ours: {ssim_img_ours:.2f} ± {ssim_img_std_ours:.2f}')
print('---')
print(f'PSNR Normal Posterior Sampling: {psnr_normal_avg_ours_posterior_sampling:.2f} ± {psnr_normal_std_posterior_sampling:.2f}')
print(f'CRC Error Posterior Sampling: {crc_avg_posterior_sampling:.2f} ± {crc_std_posterior_sampling:.2f}')
print(f'PSNR Image Posterior Sampling: {psnr_img_posterior_sampling:.2f} ± {psnr_img_std_posterior_sampling:.2f}')
print(f'SSIM Image Posterior Sampling: {ssim_img_posterior_sampling:.2f} ± {ssim_img_std_posterior_sampling:.2f}')
print('---')
print(f'PSNR Normal Richardson Lucy: {psnr_normal_avg_richardson_lucy:.2f} ± {psnr_normal_std_richardson_lucy:.2f}')
print(f'CRC Error Richardson Lucy: {crc_avg_richardson_lucy:.2f} ± {crc_std_richardson_lucy:.2f}')
print(f'PSNR Image Richardson Lucy: {psnr_img_richardson_lucy:.2f} ± {psnr_img_std_richardson_lucy:.2f}')
print(f'SSIM Image Richardson Lucy: {ssim_img_richardson_lucy:.2f} ± {ssim_img_std_richardson_lucy:.2f}')