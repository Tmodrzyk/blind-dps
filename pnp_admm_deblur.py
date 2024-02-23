from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.measurements import get_operator, get_noise

from data.dataloader import get_dataset, get_dataloader
from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color, normalize_np
from util.logger import get_logger

import numpy as np
from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot, random
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info
import matplotlib.pyplot as plt


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
    parser.add_argument('--save_dir', type=str, default='./results/ffhq/pnp-admm/')
    
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
    
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
   
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
        
        print(f"Solving on {device_info()}\n")
        x = solver.solve()
        x = snp.clip(x, 0, 1)
        
        x = np.array(normalize_np(np.abs(x)))
        y = np.array(normalize_np(np.abs(y)))
        x_gt = np.array(normalize_np(np.abs(x_gt)))

        plt.imsave(os.path.join(out_path, 'input', fname), y)
        plt.imsave(os.path.join(out_path, 'label', fname), x_gt)
        plt.imsave(os.path.join(out_path, 'recon', fname), x)
        
if __name__ == '__main__':
    main()
