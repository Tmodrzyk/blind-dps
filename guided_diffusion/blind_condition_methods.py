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
        # assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator) or isinstance(operator, GaussialBlurOperator)
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm

@register_conditioning_method(name='diffusion-posterior')
class DiffusionPosterior(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name='gd')
class GradientDescent(BlindConditioningMethod):
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
class Adam(BlindConditioningMethod):
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
    
@register_conditioning_method(name='richardson-lucy')
class MLEM(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')
        np.random.seed(123)
    
    def mlem(self, observation, x_0_hat, steps, clip, filter_epsilon, device, **kwargs):
        img = x_0_hat['img']
        kernel = x_0_hat['kernel'].repeat(1,3,1,1)
        # kernel = x_0_hat['kernel']
        
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
