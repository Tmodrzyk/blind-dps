from typing import Dict
import torch

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method
import os
import os
import matplotlib.pyplot as plt

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
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       x_0_hat_prev: Dict[str, torch.Tensor],
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser.__name__ == 'poisson' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            
            with torch.autograd.set_detect_anomaly(True):
                x_prev_values = [x[1] for x in sorted(x_prev.items())] 

                x_0_hat_prev_values = [x[1] for x in sorted(x_0_hat_prev.items())]
                x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
                # difference = measurement - self.operator.forward(*x_0_hat_prev_values)
                difference = measurement - self.operator.forward(*x_0_hat_prev_values)
                
                save_dir = './results/debug/blind_blur/progress_y/'
                import matplotlib.pyplot as plt
                os.makedirs(save_dir, exist_ok=True)

                image = self.operator.forward(*x_0_hat_prev_values).detach().cpu().numpy()[0, :, :]
                plt.imshow(image)
                plt.colorbar()
                plt.savefig(os.path.join(save_dir, 'y.png'))
                plt.close()
                
                norm = torch.linalg.norm(difference)

                ## Begin lines 12 of Algorithm 1

                reg_info = kwargs.get('regularization', None)
                if reg_info is not None:
                    for reg_target in reg_info:
                        assert reg_target in keys, \
                            f"Regularization target {reg_target} does not exist in x_0_hat."

                        reg_ord, reg_scale = reg_info[reg_target]
                        if reg_scale != 0.0:  # if got scale 0, skip calculating.
                            norm = norm + reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)                        

                ## End lines 12 of Algorithm 1
                
                # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_0_hat_prev_values)
                
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm

@register_conditioning_method(name='ps')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def conditioning(self, x_prev, x_0_hat, x_0_hat_prev, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev, x_0_hat, x_0_hat_prev, measurement, **kwargs)

        scale = kwargs.get('scale')
        if scale is None:
            scale = self.scale
        
        ## Begin lines 11-13 of Algorithm 1 
        keys = sorted(x_prev.keys())
        for k in keys:
            x_0_hat.update({k: x_0_hat[k] - scale[k]*norm_grad[k]})   
                     
        ## End lines 11-13 of Algorithm 1
        
        return x_0_hat, norm