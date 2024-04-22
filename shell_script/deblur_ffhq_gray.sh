#!/bin/bash

python blind_deblur_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config_fast.yaml \
    --task_config=configs/deblur_ffhq_grayscale.yaml \
    --save_dir='./results/ffhq/debug/' \
    --reg_ord=1 \
    --reg_scale=0.0;
