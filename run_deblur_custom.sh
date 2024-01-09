#!/bin/bash

python3 blind_deblur_demo.py \
    --img_model_config=configs/custom_model_config_128.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config_fast.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --reg_ord=2 \
    --reg_scale=0.0;
