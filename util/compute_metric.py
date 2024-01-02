from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

device = 'cuda:0'

task = 'blind_blur'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'./results/{task}/label/')

normal_recon_root = Path(f'./results/{task}/recon/')

psnr_normal_list = []

for idx in tqdm(range(1)):
    fname = 'img_' + str(idx).zfill(5)

    label = plt.imread(os.path.join(label_root, f'{fname}.png'))
    normal_recon = plt.imread(os.path.join(label_root, f'{fname}.png'))

    psnr_normal = peak_signal_noise_ratio(label, normal_recon)
    psnr_normal_list.append(psnr_normal)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)

print(f'Normal PSNR: {psnr_normal_avg}')