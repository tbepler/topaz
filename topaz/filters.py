import numpy as np
import torch
from torch import nn


def gaussian_filter(sigma, s=11, dims=2):
    dim = s//2
    ranges = np.arange(-dim, dim+1)

    if dims == 2:
        xx,yy = np.meshgrid(ranges, ranges)
    elif dims == 3:
        xx,yy,zz = np.meshgrid(ranges, ranges, ranges)

    d = xx**2 + yy**2
    d = d + zz**2 if dims == 3 else d

    f = np.exp(-0.5*d/sigma**2)
    return f


def inverse_filter(w):
    F = np.fft.rfft2(np.fft.ifftshift(w))
    F = np.fft.fftshift(np.fft.irfft2(1/F, s=w.shape))
    return F


class AffineFilter(nn.Module):
    def __init__(self, weights):
        super(AffineFilter, self).__init__()
        n = weights.shape[0]
        self.filter = nn.Conv2d(1, 1, n, padding=n//2)
        self.filter.weight.data[:] = torch.from_numpy(weights).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


class AffineDenoise(nn.Module):
    def __init__(self, max_size=31):
        super(AffineDenoise, self).__init__()
        self.filter = nn.Conv2d(1, 1, max_size, padding=max_size//2)
        self.filter.weight.data.zero_()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


class GaussianDenoise(nn.Module):
    '''
     Apply Gaussian filter with sigma to image. Truncates the kernel at scale times sigma pixels.
    '''    
    def __init__(self, sigma, scale=5, dims=2, use_cuda=False):
        super(GaussianDenoise, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter(sigma, s=width, dims=dims)
        f /= f.sum()

        if dims == 2:
            self.filter = nn.Conv2d(1, 1, width, padding=width//2)
        elif dims ==3 :
            self.filter = nn.Conv3d(1, 1, width, padding=width//2)

        self.filter.weight.data[:] = torch.from_numpy(f).float()
        self.filter.bias.data.zero_()
        self.use_cuda = use_cuda

    def forward(self, x):
        return self.filter(x)

    @torch.no_grad()
    def apply(self, x):
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        if self.use_cuda:
            self.filter.cuda()
            x = x.cuda()
        y = self.forward(x).squeeze().cpu().numpy()
        return y
    

class InvGaussianFilter(GaussianDenoise, nn.Module):
    def __init__(self, sigma, scale=5, use_cuda=False):
        super(InvGaussianFilter, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter(sigma, s=width)
        f /= f.sum()

        # now, invert the filter
        F = inverse_filter(f)

        self.filter = nn.Conv2d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(F).float()
        self.filter.bias.data.zero_()
        self.use_cuda = use_cuda