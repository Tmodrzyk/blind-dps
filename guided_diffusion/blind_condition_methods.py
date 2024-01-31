from typing import Dict
import torch

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method
import os
import os
import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy, wiener, unsupervised_wiener

from util.img_utils import clear_color
from scipy.signal import convolve
import numpy as np

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
        assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator)
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

@register_conditioning_method(name='ps')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def conditioning(self, x_0_hat, x_0_hat_prev, measurement, **kwargs):
        # norm_grad, norm = self.grad_and_value(x_0_hat, x_0_hat_prev, measurement, **kwargs)

        g_scale = kwargs.get('scale')
        if g_scale is None:
            g_scale = self.scale
        
        steps = 30
        
        for step in range(steps):
            # scale = g_scale['img'] *(steps - step) / steps
            scale = g_scale['img']
            norm_grad, norm = self.grad_and_value(x_0_hat, x_0_hat_prev, measurement, **kwargs)
            x_0_hat_prev = x_0_hat
            x_0_hat['img'] = x_0_hat['img'] - scale*norm_grad['img']
        
        return x_0_hat, norm
    
@register_conditioning_method(name='mlem')
class MLEM(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')
        np.random.seed(123)

    def mlem(self, observation, x_0_hat, steps, clip, filter_epsilon, **kwargs):
        img = x_0_hat['img']
        kernel = x_0_hat['kernel']
        
        image = observation.detach().cpu().numpy().astype(np.float32, copy=True)
        psf = kernel.detach().cpu().numpy().astype(np.float32, copy=False)
        # im_deconv = np.full(image.shape, 0.5, dtype=np.float32)
        im_deconv = img.detach().cpu().numpy().astype(np.float32, copy=True)
        psf_mirror = np.flip(psf)

        # Small regularization parameter used to avoid 0 divisions
        eps = 1e-12

        for _ in range(steps):
            conv = convolve(im_deconv, psf, mode='same') + eps
            if filter_epsilon:
                relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
            else:
                relative_blur = image / conv
            im_deconv *= convolve(relative_blur, psf_mirror, mode='same')

        if clip:
            im_deconv[im_deconv > 1] = 1
            im_deconv[im_deconv < -1] = -1

        x_0_hat['img'] = torch.from_numpy(im_deconv).to('cuda')
    
        return x_0_hat, 0

    
    def conditioning(self, x_0_hat, x_0_hat_prev, measurement, **kwargs):
        steps = 20
        # plt.imshow(x_0_hat['img'].squeeze().detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Before MLEM')
        # plt.show()
        
        x_0_hat, norm = self.mlem(observation=measurement, x_0_hat=x_0_hat, steps=steps, clip=False, filter_epsilon=1e-3, **kwargs)
        # plt.imshow(x_0_hat['img'].squeeze().detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('After MLEM')
        # plt.show()
        
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
