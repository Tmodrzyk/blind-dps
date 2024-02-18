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

    crc_component_list = []
    area_component_list = []
    
    res_dict = {}
    
    for idx in tqdm(range(11)):
        fname = str(idx).zfill(5)

        label = Image.open(os.path.join(label_root, f'{fname}.png')).convert("L")
        label = np.asarray(label) 
        recon = Image.open(os.path.join(recon_root, f'{fname}.png')).convert("L")
        recon = np.asarray(recon) 
        

        
        # # Apply a simple threshold to create a binary mask
        # _, binary_mask = cv2.threshold(label, 65, 255, cv2.THRESH_BINARY)
        
        # # Find contours in the binary mask
        # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # activity_label_img = 0
        # activity_recon_img = 0
        
        # skip = False
        
        # # Iterate over each contour
        # for contour in contours:
        #     # Create a mask for each contour
            
        #     mask = np.zeros_like(label)
            
        #     cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

        #     area = np.count_nonzero(mask)
            
        #     if(area > 1000):
        #         skip = True
        #         break
        #     # Extract the sub-region from both label and normal_recon using the mask
            
        #     label_component = cv2.bitwise_and(label, label, mask=mask)
        #     normal_recon_component = cv2.bitwise_and(recon, recon, mask=mask)
            
        #     # PSNR
        #     psnr_component = peak_signal_noise_ratio(label_component, normal_recon_component)
        #     psnr_normal_list.append(psnr_component)
            
        #     # Activities
            
        #     activity_label_component = label_component.mean()
        #     activity_recon_component = normal_recon_component.mean()
            
        #     activity_label_img += activity_label_component
        #     activity_recon_img += activity_recon_component

        #     crc_component = activity_recon_component / activity_label_component
        #     crc_err_component = np.abs(1 - crc_component)
            
        #     crc_component_list.append(crc_err_component)
        #     area_component_list.append(area)
            
        # # Concentration Recovery Coefficient 
        # if skip:
        #     continue
        
        # crc_img = activity_recon_img / activity_label_img
        # crc_err = np.abs(1 - crc_img)

        # PSNR on the whole image
        
        psnr_img = peak_signal_noise_ratio(label, recon)
        
        # SSIM on the whole image
        
        ssim_img = structural_similarity(label, recon)

        # crc_list.append(crc_err)
        psnr_img_list.append(psnr_img)
        ssim_img_list.append(ssim_img)

    res_dict['psnr_normal'] = psnr_normal_list
    res_dict['crc'] = crc_list
    res_dict['psnr_img'] = psnr_img_list
    res_dict['ssim_img'] = ssim_img_list
    res_dict['crc_component'] = crc_component_list
    res_dict['area_component'] = area_component_list

    return res_dict


label_root = Path(f'/home/modrzyk/code/blind-dps/results/ffhq/hybrid4/{task}/label/')
recon_root_ours = Path(f'/home/modrzyk/code/blind-dps/results/ffhq/hybrid4/{task}/recon/')
recon_root_posterior_sampling = Path(f'/home/modrzyk/code/diffusion-posterior-sampling/results/ffhq/{task}/recon')
recon_root_richarson_lucy = Path(f'/home/modrzyk/code/diffusion-posterior-sampling/results/ffhq/{task}/recon')

res_ours = stats(label_root, recon_root_ours)
res_posterior_sampling = stats(label_root, recon_root_posterior_sampling)
res_richardson_lucy = stats(label_root, recon_root_richarson_lucy)

df_ours = pd.DataFrame({'Area': res_ours['area_component'], 'CRC Error': res_ours['crc_component']})
df_posterior_sampling = pd.DataFrame({'Area': res_posterior_sampling['area_component'], 'CRC Error': res_posterior_sampling['crc_component']})
df_richardson_lucy = pd.DataFrame({'Area': res_richardson_lucy['area_component'], 'CRC Error': res_richardson_lucy['crc_component']})

mean_crc_by_area_ours = df_ours.groupby('Area')['CRC Error'].mean()
mean_crc_by_area_posterior_sampling = df_posterior_sampling.groupby('Area')['CRC Error'].mean()
mean_crc_by_area_richardson_lucy = df_richardson_lucy.groupby('Area')['CRC Error'].mean()


# plt.scatter(res_ours['area_component'], res_ours['crc_component'], label='Ours', alpha=0.5)
# plt.scatter(res_posterior_sampling['area_component'], res_posterior_sampling['crc_component'], label='DPS', alpha=0.5)
# plt.scatter(res_richardson_lucy['area_component'], res_richardson_lucy['crc_component'], label='Richardson-Lucy', alpha=0.5)

# plt.xlabel('Area')
# plt.ylabel('CRC Error')
# plt.title('CRC Error vs. Area')
# plt.grid(True)
# plt.legend()
# plt.show()

# Calculate average
psnr_normal_avg_ours = np.mean(res_ours['psnr_normal'])
crc_avg_ours = np.mean(res_ours['crc'])
psnr_img_ours = np.mean(res_ours['psnr_img'])
ssim_img_ours = np.mean(res_ours['ssim_img'])

psnr_normal_avg_ours_posterior_sampling = np.mean(res_posterior_sampling['psnr_normal'])
crc_avg_posterior_sampling = np.mean(res_posterior_sampling['crc'])
psnr_img_posterior_sampling = np.mean(res_posterior_sampling['psnr_img'])
ssim_img_posterior_sampling = np.mean(res_posterior_sampling['ssim_img'])

psnr_normal_avg_richardson_lucy = np.mean(res_richardson_lucy['psnr_normal'])
crc_avg_richardson_lucy = np.mean(res_richardson_lucy['crc'])
psnr_img_richardson_lucy = np.mean(res_richardson_lucy['psnr_img'])
ssim_img_richardson_lucy = np.mean(res_richardson_lucy['ssim_img'])

# Calculate standard deviation
psnr_normal_std_ours = np.std(res_ours['psnr_normal'])
crc_std_ours = np.std(res_ours['crc'])
psnr_img_std_ours = np.std(res_ours['psnr_img'])
ssim_img_std_ours = np.std(res_ours['ssim_img'])

psnr_normal_std_posterior_sampling = np.std(res_posterior_sampling['psnr_normal'])
crc_std_posterior_sampling = np.std(res_posterior_sampling['crc'])
psnr_img_std_posterior_sampling = np.std(res_posterior_sampling['psnr_img'])
ssim_img_std_posterior_sampling = np.std(res_posterior_sampling['ssim_img'])

psnr_normal_std_richardson_lucy = np.std(res_richardson_lucy['psnr_normal'])
crc_std_richardson_lucy = np.std(res_richardson_lucy['crc'])
psnr_img_std_richardson_lucy = np.std(res_richardson_lucy['psnr_img'])
ssim_img_std_richardson_lucy = np.std(res_richardson_lucy['ssim_img'])

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