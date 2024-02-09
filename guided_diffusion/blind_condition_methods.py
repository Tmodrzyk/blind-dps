from typing import Dict
import torch

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator, GaussialBlurOperator
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method
import os
import os
import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy, wiener, unsupervised_wiener

from util.img_utils import clear_color
from scipy.signal import convolve
import numpy as np

import torch.nn.functional as F

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class BlindConditioningMethod(ConditioningMethod):
    def __init__(self, operator, noiser=None, **kwargs):
        '''
        Handle multiple score models.
        Yet, support only gaussian noise measurement.
        '''
        assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator) or isinstance(operator, GaussialBlurOperator)
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def grad_and_value(self, 
                       x_0_hat: Dict[str, torch.Tensor], 
                       x_0_hat_prev: Dict[str, torch.Tensor],
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser.__name__ == 'poisson' or self.noiser is None:  # why none?
            
            keys = sorted(x_0_hat_prev.keys())
            idx = kwargs.get('idx', None)
            
            with torch.autograd.set_detect_anomaly(True):
                x_0_hat_prev_values = [x[1] for x in sorted(x_0_hat_prev.items())]
                x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
                difference = measurement - self.operator.forward(*x_0_hat_values)
                norm = torch.linalg.norm(difference)
                norm_squared = norm
                norm_grad = torch.autograd.grad(outputs=norm_squared, inputs=x_0_hat_prev_values)
                
                # if(idx is not None):
                #     save_dir = './results/debug/blind_blur/progress_grad/img/'
                #     os.makedirs(save_dir, exist_ok=True)
                    
                #     image = self.operator.forward(*x_0_hat_values)
                #     plt.imshow(clear_color(image))
                #     plt.colorbar()
                #     plt.savefig(os.path.join(save_dir, f'y_hat_{idx}.png'))
                #     plt.close()
                    
                #     save_dir = './results/debug/blind_blur/progress_grad/kernel/'
                #     os.makedirs(save_dir, exist_ok=True)
                    
                #     plt.imshow(norm_grad[0][0, 0, :, :].detach().cpu().numpy())
                #     plt.colorbar()
                #     plt.savefig(os.path.join(save_dir, f'grad_img_{idx}.png'))
                #     plt.close()
                    
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm

@register_conditioning_method(name='gd')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def gradient_descent(self, x_0_hat, measurement, steps, **kwargs):
        x_0_hat['img'].requires_grad_()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD([x_0_hat['img']], lr=10.0)  # Example with SGD

        for step in range(steps):
            optimizer.zero_grad()
            loss = criterion(self.operator.forward(x_0_hat['img']), measurement)
            loss.backward()
            optimizer.step()
        
        return x_0_hat, loss
    
    def conditioning(self, x_0_hat, measurement, steps, **kwargs):
        x_0_hat['img'].detach_()
        x_0_hat, norm = self.gradient_descent(x_0_hat, measurement, steps, **kwargs)
        
        return x_0_hat, norm

@register_conditioning_method(name='adam')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def gradient_descent(self, x_0_hat, measurement, steps, **kwargs):
        x_0_hat['img'].requires_grad_()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([x_0_hat['img']], lr=0.01)  # Example with Adam

        for step in range(steps):
            optimizer.zero_grad()
            loss = criterion(self.operator.forward(x_0_hat['img']), measurement)
            loss.backward()
            optimizer.step()
        
        return x_0_hat, loss
    
    def conditioning(self, x_0_hat, measurement, steps, **kwargs):
        x_0_hat['img'].detach_()
        x_0_hat, norm = self.gradient_descent(x_0_hat, measurement, steps, **kwargs)
        
        return x_0_hat, norm
    
@register_conditioning_method(name='mlem')
class MLEM(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')
        np.random.seed(123)
    
    def mlem(self, observation, x_0_hat, steps, clip, filter_epsilon, device, **kwargs):
        img = x_0_hat['img']
        # kernel = x_0_hat['kernel'].repeat(1,3,1,1)
        kernel = x_0_hat['kernel']
        
        image = observation.to(torch.float32).clone().to(device)
        psf = kernel.to(torch.float32).clone().to(device)
        im_deconv = img.to(torch.float32).clone().to(device)
        psf_mirror = torch.flip(psf, dims=[0, 1])
        # Small regularization parameter used to avoid 0 divisions
        eps = 1e-6
        with torch.no_grad():
            pad = (psf.size(2) // 2, psf.size(2) // 2, psf.size(3) // 2, psf.size(3) // 2)

            for _ in range(steps):
                conv = F.conv2d(F.pad(im_deconv, pad, mode='replicate'), psf) + eps
                if filter_epsilon:
                    relative_blur = torch.where(conv < filter_epsilon, torch.tensor(0.0, device=device), image / conv)
                else:
                    relative_blur = image / conv
                im_deconv *= F.conv2d(F.pad(relative_blur, pad, mode='replicate'), psf_mirror)

            if clip:
                im_deconv = torch.clamp(im_deconv, 0, 1)

            x_0_hat['img'] = im_deconv.to(device)
        # plt.imshow(x_0_hat['img'].squeeze().detach().cpu().numpy().swapaxes(0, 1).swapaxes(1, 2))
        # plt.show()
        return x_0_hat, 0
    
    def conditioning(self, x_0_hat, measurement, steps, **kwargs):
        
        x_0_hat, norm = self.mlem(observation=measurement, x_0_hat=x_0_hat, steps=steps, clip=True, filter_epsilon=1e-6, device='cuda', **kwargs)

        return x_0_hat, norm
    
@register_conditioning_method(name='wiener')
class Wiener(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    
    def wiener_deconv(self, x_0_hat, **kwargs):
        img = x_0_hat['img']
        kernel = x_0_hat['kernel']
        img = self.operator.forward(img, kernel)

        deconv_img = wiener(image=img.squeeze().cpu().detach().numpy(), balance=1.0, psf=kernel.squeeze().cpu().detach().numpy(), clip=False)
        # deconv_img, _ = unsupervised_wiener(image=img.numpy(), psf=kernel.numpy(), clip=False)
        
        x_0_hat['img'] = torch.from_numpy(deconv_img).to('cuda').unsqueeze(0).unsqueeze(0)
        
        return x_0_hat, 0
    
    def conditioning(self, x_0_hat, x_0_hat_prev, measurement, **kwargs):
        plt.imshow(x_0_hat['img'].squeeze().detach().cpu().numpy())
        plt.colorbar()
        plt.title('Before Wiener')
        plt.show()
        
        x_0_hat, norm = self.wiener_deconv(x_0_hat=x_0_hat, **kwargs)

        plt.imshow(x_0_hat['img'].squeeze().detach().cpu().numpy())
        plt.colorbar()
        plt.title('After Wiener')
        plt.show()
        
        return x_0_hat, norm
