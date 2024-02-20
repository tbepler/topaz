from __future__ import absolute_import, print_function, division

import warnings
import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

def _format(message, category, filename, lineno, line=None):
    w = '{}: {}\n'.format(category.__name__, message)
    return w
warnings.formatwarning = _format


class GpuWarning(UserWarning):
    pass


def set_device(device, error=False, warn=True):
    use_device = 'cpu'
    if device >= 0: # try to set GPU when device >= 0
        if torch.cuda.is_available():
            import torch.cuda as acc
            use_device = 'cuda'
        elif hasattr(torch,'xpu'):
            if torch.xpu.is_available():
                import torch.xpu as acc
                use_device = 'xpu'
            else:
                import torch.cpu as acc
        else:
            import torch.cpu as acc
        try:
            acc.set_device(device)
        except Exception as e:
            ## setting the device failed
            if error:
                raise e
            if warn:
                # warn the user
                message = str(e) + '\nFalling back to CPU.'
                warnings.warn(message, GpuWarning)
            # fallback to CPU
            use_device = 'cpu'
    return use_device

