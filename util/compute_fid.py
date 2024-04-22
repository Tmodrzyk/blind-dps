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
from cleanfid import fid

label_root = f'/home/modrzyk/code/data/EUSIPCO_2024/label/'
recon_root_ours = f'/home/modrzyk/code/data/EUSIPCO_2024/recon-rl-diffusion-dps-refinement/'
recon_root_y = f'/home/modrzyk/code/data/EUSIPCO_2024/ablation/y-dps-refinement/'
recon_root_richarson_lucy = f'/home/modrzyk/code/data/EUSIPCO_2024/ablation/rltv-dps-refinement/'
recon_root_pnp_admm = f'/home/modrzyk/code/data/EUSIPCO_2024/ablation/pnp-admm-dps-refinement/'

fid_ours = fid.compute_fid(label_root, recon_root_ours)
fid_y = fid.compute_fid(label_root, recon_root_y)
fid_richardson_lucy = fid.compute_fid(label_root, recon_root_richarson_lucy)
fid_pnp_admm = fid.compute_fid(label_root, recon_root_pnp_admm)

print(f'FID Ours: {fid_ours}')
print(f'FID Y: {fid_y}')
print(f'FID Richardson Lucy: {fid_richardson_lucy}')
print(f'FID PNP ADMM: {fid_pnp_admm}')