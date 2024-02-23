
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Here replaces the regular unet by our trained unet
# from guided_diffusion.unet import create_model
import guided_diffusion.diffusion_model_unet 
import guided_diffusion.unet

from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise

from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color, normalize_np
from util.logger import get_logger
from functools import partial

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot, random
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

import jax

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_config.yaml')
    
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/ffhq/ablation/pnp-admm-dps-refinement/')
    
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
    args.kernel = task_config["measurement"]["operator"]["name"]
    args.kernel_size = task_config["measurement"]["operator"]["kernel_size"]
    args.intensity = task_config["measurement"]["operator"]["intensity"]
    
    # Load model
    # img_model = guided_diffusion.diffusion_model_unet.create_model(**img_model_config)
    img_model = guided_diffusion.unet.create_model(**img_model_config)
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
    
    ## Prepare refinement conditioning method 
    refinement_cond_method = get_conditioning_method('diffusion-posterior', operator, noiser, scale=0.3)
    measurement_cond_fn_refinement = refinement_cond_method.conditioning
    
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
    sample_fn = partial(sampler.p_sample_loop, model=model, 
                        measurement_cond_fn=measurement_cond_fn, 
                        measurement_cond_fn_refinement=measurement_cond_fn_refinement)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize((256, 256)),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # Careful with the normalization, it caused the reconstruction to fail
                                    # transforms.Normalize(0.5, 0.5)
                                    
                                    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # set seed for reproduce
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True  # if using CUDA
    
    
    if args.kernel == 'motion_blur':
        kernel = Kernel(size=(args.kernel_size, args.kernel_size), intensity=args.intensity).kernelMatrix
        kernel = torch.from_numpy(kernel).type(torch.float32)
        kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)
    elif args.kernel == 'gaussian_blur':
        conv = Blurkernel('gaussian', kernel_size=args.kernel_size, std=args.intensity, device=device)
        kernel = conv.get_kernel().type(torch.float32)
        kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)
                
    psf = snp.array(kernel.squeeze().unsqueeze(2).cpu().numpy())  # convert to jax array
    A = linop.Convolve(h=psf, input_shape=(256, 256, 3))
    
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        
        # Forward measurement model (Ax + n)
        # y = operator.forward(ref_img, kernel)
        y = operator.forward(ref_img)
        y_n = noiser(y)
        
        x_gt = snp.array(ref_img.cpu().numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2))  # convert to jax array

    
        sigma = 0.05  # noise level
        
        Ax = A(x_gt)  # blurred image
        
        noise, key = random.randn(Ax.shape)
        y = Ax + sigma * noise
        
        f = loss.SquaredL2Loss(y=y, A=A)
        g = functional.DnCNN("17M")
        C = linop.Identity(x_gt.shape)

        rho = 0.2  # ADMM penalty parameter
        maxiter = 12  # number of ADMM iterations

        solver = ADMM(
            f=f,
            g_list=[g],
            C_list=[C],
            rho_list=[rho],
            x0=A.T @ y,
            maxiter=maxiter,
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 30}),
            itstat_options={"display": True},
        )

        nc = 61 // 2
        y = snp.clip(y[nc:-nc, nc:-nc], 0, 1)

        x = solver.solve()
        x = snp.clip(x, 0, 1)

        x = np.array(x)
        x = torch.from_numpy(x).to(device)
        x = x.permute(2, 0, 1).unsqueeze(0)
        
        x_start = {'img': x,
                'kernel': kernel}

        # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
        for k in x_start:
            if k in model.keys():
                logger.info(f"{k} will use diffusion prior")
            else:
                logger.info(f"{k} will use uniform prior.")
    
        # sample 
        sample, norms = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path, gt=ref_img)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample['img']))

        # plt.plot(range(sampler.num_timesteps), norms)
        # plt.xlabel('Iteration Index')
        # plt.ylabel('Norm')
        # plt.ylim(bottom=0, top=max(norms))  # Set the Y-axis range
        
        # save_dir = os.path.join(out_path, f'progress_norm/')
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f'norm_{fname}'))
        # plt.close()
        
if __name__ == '__main__':
    main()
