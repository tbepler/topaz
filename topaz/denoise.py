from __future__ import absolute_import, print_function, division

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


from topaz.utils.data.loader import load_image


def load_model(name):

    if name == 'none':
        return Identity()

    # set the name aliases
    if name == 'unet':
        name = 'unet_L2_v0.2.2.sav'
    elif name == 'unet-small':
        name = 'unet_small_L1_v0.2.2.sav'
    elif name == 'fcnn':
        name = 'fcnn_L1_v0.2.2.sav'
    elif name == 'affine':
        name = 'affine_L1_v0.2.2.sav'
    elif name == 'unet-v0.2.1':
        name = 'unet_L2_v0.2.1.sav'

    # construct model and load the state
    if name == 'unet_L2_v0.2.1.sav':
        model = UDenoiseNet(base_width=7, top_width=3)
    elif name == 'unet_L2_v0.2.2.sav':
        model = UDenoiseNet(base_width=11, top_width=5)
    elif name == 'unet_small_L1_v0.2.2.sav':
        model = UDenoiseNetSmall(width=11, top_width=5)
    elif name == 'fcnn_L1_v0.2.2.sav':
        model = DenoiseNet2(64, width=11)
    elif name == 'affine_L1_v0.2.2.sav':
        model = AffineDenoise(max_size=31)
    else:
        # if not set to a pretrained model, try loading path directly
        return torch.load(name)

    # load the pretrained model parameters
    import pkg_resources
    pkg = __name__
    path = 'pretrained/denoise/' + name
    f = pkg_resources.resource_stream(pkg, path)
    state_dict = torch.load(f) # load the parameters

    model.load_state_dict(state_dict)

    return model


def denoise(model, x, patch_size=-1, padding=128):

    # check the patch plus padding size
    use_patch = False
    if patch_size > 0:
        s = patch_size + padding
        use_patch = (s < x.size(0)) or (s < x.size(1))

    if use_patch:
        return denoise_patches(model, x, patch_size, padding=padding)

    with torch.no_grad():
        x = x.unsqueeze(0).unsqueeze(0)
        y = model(x).squeeze()

    return y


def denoise_patches(model, x, patch_size, padding=128):
    y = torch.zeros_like(x)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                # include padding extra pixels on either side
                si = max(0, i - padding)
                ei = min(x.size(2), i + patch_size + padding)

                sj = max(0, j - padding)
                ej = min(x.size(3), j + patch_size + padding)

                xij = x[:,:,si:ei,sj:ej]
                yij = model(xij).squeeze() # denoise the patch

                # match back without the padding
                si = i - si
                sj = j - sj

                y[i:i+patch_size,j:j+patch_size] = yij[si:si+patch_size,sj:sj+patch_size]

    return y

def denoise_stack(model, stack, batch_size=20, use_cuda=False):
    denoised = np.zeros_like(stack)
    with torch.no_grad():
        stack = torch.from_numpy(stack).float()

        for i in range(0, len(stack), batch_size):
            x = stack[i:i+batch_size]
            if use_cuda:
                x = x.cuda()
            mu = x.view(x.size(0), -1).mean(1)
            std = x.view(x.size(0), -1).std(1)
            x = (x - mu.unsqueeze(1).unsqueeze(2))/std.unsqueeze(1).unsqueeze(2)

            y = model(x.unsqueeze(1)).squeeze(1)
            y = std.unsqueeze(1).unsqueeze(2)*y + mu.unsqueeze(1).unsqueeze(2)

            y = y.cpu().numpy()
            denoised[i:i+batch_size] = y

    return denoised

def spatial_covariance(x, n=11, s=11):

    # first, calculate the mean of each region
    #f = torch.ones(n, n)/(n*n)
    #f = f.to(x.device)
    #mu = F.conv2d(x.unsqueeze(0).unsqueeze(1), f.unsqueeze(0).unsqueeze(1)).squeeze()
    
    # now, to calculate the spatial covariance, we calculate
    # the expectation of (x_ii-mu)*(x_jk-mu)

    p = n//2

    """
    x_c = x[p:-p,p:-p] - mu
    cov = torch.zeros(n, n).to(x.device)
    N = x.size(0) - n + 1
    M = x.size(1) - n + 1
    for i in range(n):
        for j in range(n):
            x_ij = x[i:i+N,j:j+M] - mu[i,j]
            cov[i,j] = torch.mean(x_c*x_ij)
    """

    x_c = x[p:-p,p:-p]
    cov = F.conv2d(x.unsqueeze(0).unsqueeze(1), x_c.unsqueeze(0).unsqueeze(1)).squeeze()
    cov /= x_c.size(0)*x_c.size(1)

    return cov

def spatial_covariance_old(x, n=11, s=11):
    tiles = []
    for i in range(0, x.shape[0], s):
        for j in range(0, x.shape[1], s):
            if i+n <= x.shape[0] and j+n <= x.shape[1]:
                t = x[i:i+n,j:j+n]
                tiles.append(t)
    tiles = torch.stack(tiles, 0)
    tiles = tiles.view(len(tiles), -1)
    m = tiles.mean(1, keepdim=True)
    tiles = tiles - m

    m = tiles.mean(0)
    x = (tiles - m)
    cov = torch.mm(x.t(), x)/x.size(0)
    cov = cov.view(n, n, n, n)

    # we are only interested in the central pixel
    i = n//2
    return cov[i,i]


def gaussian_filter(sigma, s=11):
    dim = s//2
    xx,yy = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1))
    d = xx**2 + yy**2
    f = np.exp(-0.5*d/sigma**2)
    return f


def gaussian_filter_3d(sigma, s=11):
    dim = s//2
    xx,yy,zz = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1), np.arange(-dim,dim+1))
    d = xx**2 + yy**2 + zz**2
    f = np.exp(-0.5*d/sigma**2)
    return f


def inverse_filter(w):
    F = np.fft.rfft2(np.fft.ifftshift(w))
    F = np.fft.fftshift(np.fft.irfft2(1/F, s=w.shape))
    return F


def estimate_unblur_filter(x, width=11, s=11):
    """
    Estimate parameters of the affine filter that would give
    zero autocovariance to the image.
    """

    cov = spatial_covariance(x, n=width, s=s)

    # calculate the power spectrum, this is the fft of the autocovariance
    cov = cov.cpu().numpy()
    ps = np.fft.ifftshift(cov)
    ps = np.fft.fft2(ps)
    # clip ps <= 0 to 1...
    ps.real[ps.real <= 0] = 1
    # also set the zero frequency term to 1
    ps[0,0] = 1

    # calculate the filter that flattens the power spectrum
    # this corrects the autocorrelation

    F = 1/np.sqrt(ps.real)
    w_inv = np.fft.fftshift(np.fft.ifft2(F)).real

    return AffineFilter(w_inv), cov


def estimate_unblur_filter_gaussian(x, width=11, s=11):
    """
    Estimate parameters of the Gaussian filter that would give
    the closest spatial covariance structure to that observed
    in image x if the unfiltered image had zero spatial covariance.
    Then, return inverse of that filter.
    """

    from scipy.signal import convolve2d
    from scipy.optimize import minimize

    cov = spatial_covariance(x, n=width, s=s)

    dim = s//2
    xx,yy = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1))
    d = xx**2 + yy**2
    d = torch.from_numpy(d).float().to(cov.device)

    # solve for Gaussian kernel parameters that give closest covariance structure
    def loss(params):
        params = torch.from_numpy(params).float().to(cov.device)
        params.requires_grad = True
        params_exp = torch.exp(params)
        sigma = params_exp[0]
        alpha = params_exp[1]

        w = alpha*torch.exp(-0.5*d/sigma**2)
        w = w.unsqueeze(0).unsqueeze(1)

        c_w = F.conv2d(w, w, padding=width//2).squeeze() # covaraince structure given by application of w

        err = torch.sum((c_w - cov)**2)

        err.backward()
        grad = params.grad.data.cpu().numpy()
        err = err.item()

        return err, grad

    w0 = np.array([0, 0])
    result = minimize(loss, w0, jac=True)

    #return result, cov

    sigma = np.exp(result.x[0])
    alpha = np.exp(result.x[1])

    w = gaussian_filter(sigma)*alpha
    w_inv = inverse_filter(w)

    return AffineFilter(w_inv), sigma, alpha, cov


def correct_spatial_covariance(x, width=11, s=11, patch=1):
    """
    Estimates the spatial covariance in the micrograph, finds closest Guassian kernel
    parameters that would give this covariance structure, and applies inverse of that
    filter to correct the spatial covariance.
    """
    if patch > 1:

        N = [x.size(0)//patch]*patch
        for i in range(x.size(0)%patch):
            N[i] += 1
        M = [x.size(1)//patch]*patch
        for i in range(x.size(1)%patch):
            M[i] += 1

        y = torch.zeros_like(x)
        
        i = 0
        for n in N:
            j = 0
            for m in M:
                pad_ii = max(0, i-width//2)
                pad_ij = min(x.size(0), i+n+width//2)

                pad_ji = max(0, j-width//2)
                pad_jj = min(x.size(1), j+m+width//2)

                x_ij = x[pad_ii:pad_ij,pad_ji:pad_jj]
                y_ij = correct_spatial_covariance(x_ij, width=width, s=s)
                y[i:i+n,j:j+m] = y_ij[i-pad_ii:i-pad_ii+n,j-pad_ji:j-pad_ji+m]

                j += m
            i += n

    else:
        # estimate the parameters
        #f,sigma,alpha,cov = estimate_unblur_filter(x, width=width, s=s)
        f,cov = estimate_unblur_filter(x, width=width, s=s)
        f = f.to(x.device)

        x = x.unsqueeze(0).unsqueeze(1)
        y = f(x).squeeze()

    return y


class GaussianDenoise(nn.Module):
    def __init__(self, sigma, scale=5):
        super(GaussianDenoise, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter(sigma, s=width)
        f /= f.sum()

        self.filter = nn.Conv2d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(f).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


class GaussianDenoise3d(nn.Module):
    def __init__(self, sigma, scale=5):
        super(GaussianDenoise3d, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter_3d(sigma, s=width)
        f /= f.sum()

        self.filter = nn.Conv3d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(f).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


class InvGaussianFilter(nn.Module):
    def __init__(self, sigma, scale=5):
        super(InvGaussianFilter, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter(sigma, s=width)
        f /= f.sum()

        # now, invert the filter
        F = inverse_filter(f)

        self.filter = nn.Conv2d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(F).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


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


class DenoiseNet(nn.Module):
    def __init__(self, base_filters):
        super(DenoiseNet, self).__init__()

        self.base_filters = base_filters
        nf = base_filters
        self.net = nn.Sequential( nn.Conv2d(1, nf, 11, padding=5)
                                , nn.LeakyReLU(0.1)
                                , nn.MaxPool2d(3, stride=1, padding=1)
                                , nn.Conv2d(nf, 2*nf, 3, padding=2, dilation=2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(2*nf, 2*nf, 3, padding=4, dilation=4)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(2*nf, 3*nf, 3, padding=1)
                                , nn.LeakyReLU(0.1)
                                , nn.MaxPool2d(3, stride=1, padding=1)
                                , nn.Conv2d(nf, 2*nf, 3, padding=2, dilation=2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(3*nf, 3*nf, 3, padding=4, dilation=4)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(3*nf, 1, 7, padding=3)
                                )

    def forward(self, x):
        return self.net(x)


class DenoiseNet2(nn.Module):
    def __init__(self, base_filters, width=11):
        super(DenoiseNet2, self).__init__()

        self.base_filters = base_filters
        nf = base_filters
        self.net = nn.Sequential( nn.Conv2d(1, nf, width, padding=width//2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(nf, nf, width, padding=width//2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(nf, 1, width, padding=width//2)
                                )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class UDenoiseNet(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class UDenoiseNetSmall(nn.Module):
    def __init__(self, nf=48, width=11, top_width=3):
        super(UDenoiseNetSmall, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, width, padding=width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec3 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        h = self.enc4(p3)

        # upsampling with skip connections
        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class UDenoiseNet2(nn.Module):
    # modified U-net from noise2noise paper
    def __init__(self, nf=48):
        super(UDenoiseNet2, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf, 64, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, 3, padding=1)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')

        y = self.dec1(h)

        return y


class UDenoiseNet3(nn.Module):
    def __init__(self):
        super(UDenoiseNet3, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, 48, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(97, 64, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, 3, padding=1)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = x - self.dec1(h) # learn only noise component

        return y

class UDenoiseNet3D(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet3D, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv3d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv3d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        o = p4.size(4)
        #h = F.upsample(h, size=(n,m))
        #h = F.upsample(h, size=(n,m), mode='bilinear', align_corners=False)
        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        o = p3.size(4)
        
        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        o = p2.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        o = p1.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        o = x.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y



class PairedImages:
    def __init__(self, x, y, crop=800, xform=True, preload=False, cutoff=0):
        self.x = x
        self.y = y
        self.crop = crop
        self.xform = xform
        self.cutoff = cutoff

        self.preload = preload
        if preload:
            self.x = [self.load_image(p) for p in x]
            self.y = [self.load_image(p) for p in y]

    def load_image(self, path):
        x = np.array(load_image(path), copy=False)
        x = x.astype(np.float32) # make sure dtype is single precision
        mu = x.mean()
        std = x.std()
        x = (x - mu)/std
        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.preload:
            x = self.x[i]
            y = self.y[i]
        else:
            x = self.load_image(self.x[i])
            y = self.load_image(self.y[i])

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]
            y = y[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
                y = np.flip(y, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)
                y = np.flip(y, 1)


            k = np.random.randint(4)
            x = np.rot90(x, k=k)
            y = np.rot90(y, k=k)

            # swap x and y
            if np.random.rand() > 0.5:
                t = x
                x = y
                y = t

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        return x, y


class NoiseImages:
    def __init__(self, x, crop=800, xform=True, preload=False, cutoff=0):
        self.x = x
        self.crop = crop
        self.xform = xform
        self.cutoff = cutoff

        self.preload = preload
        if preload:
            x = [self.load_image(p) for p in x]

    def load_image(self, path):
        x = np.array(load_image(path), copy=False)
        mu = x.mean()
        std = x.std()
        x = (x - mu)/std
        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.preload:
            x = self.x[i]
        else:
            x = self.load_image(self.x[i])

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)

            k = np.random.randint(4)
            x = np.rot90(x, k=k)

        x = np.ascontiguousarray(x)

        return x


class GaussianNoise:
    def __init__(self, x, sigma=1.0, crop=500, xform=True):
        self.x = x
        self.sigma = sigma
        self.crop = crop
        self.xform = xform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)

            k = np.random.randint(4)
            x = np.rot90(x, k=k)

        # generate random noise
        std = x.std()*self.sigma
        n,m = x.shape
        r1 = np.random.randn(n, m).astype(np.float32)*std
        r2 = np.random.randn(n, m).astype(np.float32)*std

        return x+r1, x+r2
    

class L0Loss:
    def __init__(self, eps=1e-8, gamma=2):
        self.eps = eps
        self.gamma = gamma

    def __call__(self, x, y):
        return torch.mean((torch.abs(x - y) + self.eps)**self.gamma)


def eval_noise2noise(model, dataset, criteria, batch_size=10
                    , use_cuda=False, num_workers=0):
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size
                                               , num_workers=num_workers)

    n = 0
    loss = 0

    model.eval()
        
    with torch.no_grad():
        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss_ = criteria(y, x2).item()

            b = x1.size(0)
            n += b
            delta = b*(loss_ - loss)
            loss += delta/n

    return loss


def train_noise2noise(model, dataset, lr=0.001, optim='adagrad', batch_size=10, num_epochs=100
                     , criteria=nn.MSELoss(), dataset_val=None
                     , use_cuda=False, num_workers=0, shuffle=True):

    gamma = None
    if criteria == 'L0':
        gamma = 2
        eps = 1e-8
        criteria = L0Loss(eps=eps, gamma=gamma)
    elif criteria == 'L1':
        criteria = nn.L1Loss()
    elif criteria == 'L2':
        criteria = nn.MSELoss()
    
    if optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                               , num_workers=num_workers)

    total = len(dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        
        n = 0
        loss_accum = 0

        if gamma is not None:
            # anneal gamma to 0
            criteria.gamma = 2 - (epoch-1)*2/num_epochs

        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss = criteria(y, x2)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()
            b = x1.size(0)

            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            print('# [{}/{}] {:.2%} loss={:.5f}'.format(epoch, num_epochs, n/total, loss_accum)
                 , file=sys.stderr, end='\r')
        print(' '*80, file=sys.stderr, end='\r')

        if dataset_val is not None:
            loss_val = eval_noise2noise(model, dataset_val, criteria
                                       , batch_size=batch_size
                                       , num_workers=num_workers
                                       , use_cuda=use_cuda
                                       )
            yield epoch, loss_accum, loss_val
        else:
            yield epoch, loss_accum


def eval_mask_denoise(model, dataset, criteria, p=0.01 # masking rate
                     , batch_size=10, use_cuda=False, num_workers=0):
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size
                                               , num_workers=num_workers)

    n = 0
    loss = 0

    model.eval()
        
    with torch.no_grad():
        for x in data_iterator:
            # sample the mask
            mask = (torch.rand(x.size()) < p)
            r = torch.randn(x.size())

            if use_cuda:
                x = x.cuda()
                mask = mask.cuda()
                r = r.cuda()

            # mask out x by replacing from N(0,1)
            x_ = mask.float()*r + (1-mask.float())*x

            # denoise the image
            y = model(x_.unsqueeze(1)).squeeze(1)

            # calculate the loss for the masked entries
            x = x[mask]
            y = y[mask]

            loss_ = criteria(y, x).item()

            b = x.size(0)
            n += b
            delta = b*(loss_ - loss)
            loss += delta/n

    return loss


def train_mask_denoise(model, dataset, p=0.01, lr=0.001, optim='adagrad', batch_size=10, num_epochs=100
                      , criteria=nn.MSELoss(), dataset_val=None
                      , use_cuda=False, num_workers=0, shuffle=True):

    gamma = None
    if criteria == 'L0':
        gamma = 2
        eps = 1e-8
        criteria = L0Loss(eps=eps, gamma=gamma)
    elif criteria == 'L1':
        criteria = nn.L1Loss()
    elif criteria == 'L2':
        criteria = nn.MSELoss()
    
    if optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                               , num_workers=num_workers)

    total = len(dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        
        n = 0
        loss_accum = 0

        if gamma is not None:
            # anneal gamma to 0
            criteria.gamma = 2 - (epoch-1)*2/num_epochs

        for x in data_iterator:
            b = x.size(0)

            # sample the mask
            mask = (torch.rand(x.size()) < p)
            r = torch.randn(x.size())

            if use_cuda:
                x = x.cuda()
                mask = mask.cuda()
                r = r.cuda()

            # mask out x by replacing from N(0,1)
            x_ = mask.float()*r + (1-mask.float())*x

            # denoise the image
            y = model(x_.unsqueeze(1)).squeeze(1)

            # calculate the loss for the masked entries
            x = x[mask]
            y = y[mask]

            loss = criteria(y, x)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()
            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            print('# [{}/{}] {:.2%} loss={:.5f}'.format(epoch, num_epochs, n/total, loss_accum)
                 , file=sys.stderr, end='\r')
        print(' '*80, file=sys.stderr, end='\r')

        if dataset_val is not None:
            loss_val = eval_mask_denoise(model, dataset_val, criteria, p=p
                                        , batch_size=batch_size
                                        , num_workers=num_workers
                                        , use_cuda=use_cuda
                                        )
            yield epoch, loss_accum, loss_val
        else:
            yield epoch, loss_accum


def lowpass(x, factor=1):
    """ low pass filter with FFT """

    freq0 = np.fft.fftfreq(x.shape[-2])
    freq1 = np.fft.rfftfreq(x.shape[-1])
    freq = np.meshgrid(freq0, freq1, indexing='ij')
    freq = np.stack(freq, 2)

    r = np.abs(freq)
    mask = np.any((r > 0.5/factor), 2) 

    F = np.fft.rfft2(x)
    F[...,mask] = 0

    f = np.fft.irfft2(F, s=x.shape)
    f = f.astype(x.dtype)

    return f


def lowpass3d(x, factor=1):
    """ low pass filter with FFT """

    freq0 = np.fft.fftfreq(x.shape[-3])
    freq1 = np.fft.fftfreq(x.shape[-2])
    freq2 = np.fft.rfftfreq(x.shape[-1])
    freq = np.meshgrid(freq0, freq1, freq2, indexing='ij')
    freq = np.stack(freq, 3)

    r = np.abs(freq)
    mask = np.any((r > 0.5/factor), 3) 

    F = np.fft.rfftn(x)
    F[...,mask] = 0

    f = np.fft.irfftn(F, s=x.shape)
    f = f.astype(x.dtype)

    return f


def gaussian(x, sigma=1, scale=5, use_cuda=False):
    """
    Apply Gaussian filter with sigma to image. Truncates the kernel at scale times sigma pixels
    """

    f = GaussianDenoise(sigma, scale=scale)
    if use_cuda:
        f.cuda()

    with torch.no_grad():
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        y = f(x).squeeze().cpu().numpy()
    return y


def gaussian3d(x, sigma=1, scale=5, use_cuda=False):
    """
    Apply Gaussian filter with sigma to volume. Truncates the kernel at scale times sigma pixels
    """

    f = GaussianDenoise3d(sigma, scale=scale)
    if use_cuda:
        f.cuda()

    with torch.no_grad():
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        y = f(x).squeeze().cpu().numpy()
    return y

    


