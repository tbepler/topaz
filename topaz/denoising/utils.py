import sys
import torch
from torch import nn

# this seems unnecessary
def set_device(model, device, log=sys.stderr):
    # set the device or devices
    d = device
    use_cuda = (d != -1) and torch.cuda.is_available()
    num_devices = 1
    if use_cuda:
        device_count = torch.cuda.device_count()
        try:
            if d >= 0:
                assert d < device_count
                torch.cuda.set_device(d)
                print('# using CUDA device:', d, file=log)
            elif d == -2:
                print('# using all available CUDA devices:', device_count, file=log)
                num_devices = device_count
                model = nn.DataParallel(model)
            else:
                raise ValueError
        except (AssertionError, ValueError):
            print('ERROR: Invalid device id or format', file=log)
            sys.exit(1)
        except Exception:
            print('ERROR: Something went wrong with setting the compute device', file=log)
            sys.exit(2)

    if use_cuda:
        model.cuda()

    return model, use_cuda, num_devices
