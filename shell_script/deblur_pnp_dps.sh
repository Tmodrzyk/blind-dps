#!/bin/bash

/home/modrzyk/miniconda3/envs/scico_test/bin/python pnp_admm_dps_deblur.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config_fast.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --reg_ord=1 \
    --reg_scale=0.0;
