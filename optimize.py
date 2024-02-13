from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise

# Here replaces the regular unet by our trained unet
# from guided_diffusion.unet import create_model
import guided_diffusion.diffusion_model_unet 
import guided_diffusion.unet

from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color
from util.logger import get_logger
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity 
import optuna

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    
    # set seed for reproduce
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True  # if using CUDA
    
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_config.yaml')
    
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/ellipse/hybrid/')
    
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1, 2])
    
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    img_model_config = load_yaml(args.img_model_config)
    kernel_model_config = load_yaml(args.kernel_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Kernel configs to namespace save space
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]
    args.kernel_std = task_config["kernel_std"]
    
    # Load model
    img_model = guided_diffusion.diffusion_model_unet.create_model(**img_model_config)
    # img_model = guided_diffusion.unet.create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()
    
    # kernel_model = guided_diffusion.diffusion_model_unet.create_model(**kernel_model_config)
    kernel_model = guided_diffusion.unet.create_model(**kernel_model_config)
    kernel_model = kernel_model.to(device)
    kernel_model.eval()
    model = {'img': img_model, 'kernel': kernel_model}

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    measurement_cond_fn = cond_method.conditioning

    # Add regularization
    # Not to use regularization, set reg_scale = 0 or remove this part.
    regularization = {'kernel': (args.reg_ord, args.reg_scale)}
    measurement_cond_fn = partial(measurement_cond_fn, regularization=regularization)
    if args.reg_scale == 0.0:
        logger.info(f"Got kernel regularization scale 0.0, skip calculating regularization term.")
    else:
        logger.info(f"Kernel regularization : L{args.reg_ord}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    def objective(trial):
        params = {
            'M': trial.suggest_int('M', 1, 30),
            'N': trial.suggest_int('N', 10, 40),
        }
        
        diffusion_config['steps'] = params['N']
        diffusion_config['timestep_respacing'] = params['N']
        
        sampler = create_sampler(**diffusion_config) 
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
        
        if args.kernel == 'motion':
            kernel = Kernel(size=(args.kernel_size, args.kernel_size), intensity=args.intensity).kernelMatrix
            kernel = torch.from_numpy(kernel).type(torch.float32)
            kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)
        elif args.kernel == 'gaussian':
            conv = Blurkernel('gaussian', kernel_size=args.kernel_size, std=args.kernel_std, device=device)
            kernel = conv.get_kernel().type(torch.float32)
            kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)
            
        psnr_list = []
        
        # Do Inference
        for i, ref_img in enumerate(tqdm(loader)):

            fname = str(i).zfill(5) + '.png'
            ref_img = ref_img.to(device)
            
            y = operator.forward(ref_img)
            y_n = noiser(y)
            y_n = torch.clamp(y_n, 0)
            
            x_start = {'img': y_n,
                    'kernel': kernel}

            sample, norms = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path, gt=ref_img, M=params['M'])
            sample_img = sample['img'].squeeze().detach().cpu().numpy()
            ref_img = ref_img.squeeze().detach().cpu().numpy()
            psnr_list += [peak_signal_noise_ratio(ref_img, sample_img)]
        
        score = np.mean(psnr_list)
        
        return score
    
    study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3", study_name="diffusion")
    study.optimize(objective, n_trials=50)
    
    print("Best hyperparameters : ", study.best_params)
    print("Best score : ", study.best_value)

if __name__ == '__main__':
    main()
