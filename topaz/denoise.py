from __future__ import absolute_import, division, print_function

import os
import sys
from tabnanny import verbose
from tkinter import N
from turtle import st
from typing import Any, List, Union

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
from topaz import mrc
from topaz.denoising.datasets import PatchDataset
from topaz.denoising.models import load_model, train_model
from topaz.filters import (AffineFilter, GaussianDenoise, gaussian_filter,
                           inverse_filter)
from topaz.utils.data.loader import load_image
from topaz.utils.image import save_image
from torch.utils.data import DataLoader


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

    from scipy.optimize import minimize
    from scipy.signal import convolve2d

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

def lowpass(x, factor=1, dims=2):
    """ low pass filter with FFT """

    if dims == 2:
        freq0 = np.fft.fftfreq(x.shape[-2])
        freq1 = np.fft.rfftfreq(x.shape[-1])
    elif dims == 3:
        freq0 = np.fft.fftfreq(x.shape[-3])
        freq1 = np.fft.fftfreq(x.shape[-2])
        freq2 = np.fft.rfftfreq(x.shape[-1])

    freq = np.meshgrid(freq0, freq1, indexing='ij') if dims ==2 else np.meshgrid(freq0, freq1, freq2, indexing='ij')
    freq = np.stack(freq, dims)

    r = np.abs(freq)
    mask = np.any((r > 0.5/factor), dims) 

    F = np.fft.rfftn(x)
    F[...,mask] = 0

    f = np.fft.irfftn(F, s=x.shape)
    f = f.astype(x.dtype)

    return f


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


def denoise_image(mic, models, lowpass=1, cutoff=0, gaus=None, inv_gaus=None, deconvolve=False
                 , deconv_patch=1, patch_size=-1, padding=0, normalize=False
                 , use_cuda=False):
    if lowpass > 1:
        mic = lowpass(mic, lowpass)

    mic = torch.from_numpy(mic)
    if use_cuda:
        mic = mic.cuda()

    # normalize and remove outliers
    mu = mic.mean()
    std = mic.std()
    x = (mic - mu)/std
    if cutoff > 0:
        x[(x < -cutoff) | (x > cutoff)] = 0

    # apply guassian/inverse gaussian filter
    if gaus is not None:
        x = denoise(gaus, x)
    elif inv_gaus is not None:
        x = denoise(inv_gaus, x)
    elif deconvolve:
        # estimate optimal filter and correct spatial correlation
        x = correct_spatial_covariance(x, patch=deconv_patch)

    # denoise
    mic = 0
    for model in models:
        mic += denoise(model, x, patch_size=patch_size, padding=padding)
    mic /= len(models)

    # restore pixel scaling
    if normalize:
        mic = (mic - mic.mean())/mic.std()
    else:
        # add back std. dev. and mean
        mic = std*mic + mu

    # back to numpy/cpu
    mic = mic.cpu().numpy()

    return mic





###########################################################
# new stuff below
###########################################################

class Denoise():
    ''' Object for micrograph denoising utilities.
    '''
    def __init__(self, model:Union[torch.nn.Module, str], use_cuda=False):
        if type(model) == torch.nn.Module or type(model) == torch.nn.Sequential:
            self.model = model
        elif type(model) == str:
            try:
                self.model = load_model(model)
            except:
                raise ValueError('Unable to load model: ' + model)
        else:
            raise TypeError('Unrecognized model:' + model)
        if use_cuda:
            self.model = self.model.cuda()
        self.device = next(iter(self.model.parameters())).device
    
    def __call__(self, input):
        self.denoise(input)
 
    
    def train(self, train_dataset, val_dataset, loss_fn:str='L2', optim:str='adam', lr:float=0.001, weight_decay:float=0, batch_size:int=10, num_epochs:int=500, 
                        shuffle:bool=True, num_workers:int=1, verbose:bool=True, save_best:bool=False, save_interval:int=None, save_prefix:str=None) -> None:
        train_model(self.model, train_dataset, val_dataset, loss_fn, optim, lr, weight_decay, batch_size, num_epochs, shuffle, self.use_cuda, num_workers, verbose, save_best, save_interval, save_prefix)

    
    @torch.no_grad()        
    def denoise(self, input:np.ndarray):
        self.model.eval()
        mu, std =  input.mean(), input.std()
        # normalize, add singleton batch and input channel dims 
        input = torch.from_numpy( (input-mu)/std ).to(self.device).unsqueeze(0).unsqueeze(0)
        pred = self.model(input)
        # remove singleton dims, unnormalize
        return pred.squeeze().cpu().numpy() * std + mu
 
    
    @torch.no_grad()
    def denoise_patches(self, model, x, patch_size, padding=128):
        ''' Denoise micrograph patches.
        '''
        y = torch.zeros_like(x)

        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                # include padding extra pixels on either side
                si = max(0, i - padding)
                ei = min(x.size(2), i + patch_size + padding)
                sj = max(0, j - padding)
                ej = min(x.size(3), j + patch_size + padding)

                xij = x[:,:,si:ei,sj:ej]
                yij = self.denoise(xij) # denoise the patch

                # match back without the padding
                si = i - si
                sj = j - sj

                y[i:i+patch_size,j:j+patch_size] = yij[si:si+patch_size,sj:sj+patch_size]
        return y
  


class Denoise3D(Denoise):
    ''' Object for denoising tomograms.
    '''
    @torch.no_grad()
    def denoise(self, tomo:np.ndarray, patch_size:int=128, padding:int=128, batch_size:int=1, volume_num:int=1, total_volumes:int=1, verbose:bool=True) -> np.ndarray:
        denoised = np.zeros_like(tomo)
        mu, std =  tomo.mean(), tomo.std()
        
        if patch_size < 1:
            # no patches, simple denoise
            denoised[:] = super().denoise(tomo)
        else:
            # denoise volume in patches
            patch_data = PatchDataset(tomo, patch_size, padding)
            batch_iterator = DataLoader(patch_data, batch_size=batch_size)
            count, total = 0, len(patch_data)
            
            for index,x in batch_iterator:
                x = super().denoise( (x - mu)/std ) * std + mu

                # stitch into denoised volume
                for b in range(len(x)):
                    i,j,k = index[b]
                    xb = x[b]

                    patch = denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size]
                    pz,py,px = patch.shape

                    xb = xb[padding:padding+pz,padding:padding+py,padding:padding+px]
                    denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size] = xb

                    count += 1
                    if verbose:
                        print(f'# [{volume_num}/{total_volumes}] {round(count*100/total)}', file=sys.stderr, end='\r')

            print(' '*100, file=sys.stderr, end='\r')

        return denoised





def denoise_stack(path:str, output_path:str, models:List[Denoise], lowpass:float=1, pixel_cutoff:float=0, 
                  gaus=None, inv_gaus=None, deconvolve:bool=True, deconv_patch:int=1, patch_size:int=1024, 
                  padding:int=500, normalize:bool=True, use_cuda:bool=False):
    # TODO: REMOVE USE OF DENOISE_IMAGE
    with open(path, 'rb') as f:
        content = f.read()
    stack,_,_ = mrc.parse(content)
    print('# denoising stack with shape:', stack.shape, file=sys.stderr)
    total = len(stack)
    count = 0

    denoised = np.zeros_like(stack)
    for i in range(len(stack)):
        mic = stack[i]
        # process and denoise the micrograph
        mic_denoised = 0
        for model in models:
            mic_denoised += model.denoise()
            denoise(model, x, patch_size=patch_size, padding=padding)
            mic_denoised /= len(models)
    
    
        mic = denoise_image(mic, models, lowpass=lowpass, cutoff=pixel_cutoff, gaus=gaus, 
                            inv_gaus=inv_gaus, deconvolve=deconvolve, deconv_patch=deconv_patch,
                            patch_size=patch_size, padding=padding, normalize=normalize, use_cuda=use_cuda)
        denoised[i] = mic

        count += 1
        print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')

    print('', file=sys.stderr)
    # write the denoised stack
    print('# writing to', output_path, file=sys.stderr)
    with open(output_path, 'wb') as f:
        mrc.write(f, denoised)
    
    return denoised


def denoise_stream(micrographs:List[str], output_path:str, format:str='mrc', suffix:str='', models:List[Denoise]=None, lowpass:float=1, 
                   pixel_cutoff:float=0, gaus=None, inv_gaus=None, deconvolve:bool=True, deconv_patch:int=1, patch_size:int=1024, 
                   padding:int=500, normalize:bool=True, use_cuda:bool=False):
    # stream the micrographs and denoise them
    total = len(micrographs)
    count = 0
    denoised = [] 

    # make the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path in micrographs:
        name,_ = os.path.splitext(os.path.basename(path))
        mic = np.array(load_image(path), copy=False).astype(np.float32)

        # process and denoise the micrograph
        # TODO: REMOVE USE OF DENOISE_IMAGE
        mic = denoise_image(mic, models, lowpass=lowpass, cutoff=pixel_cutoff, gaus=gaus, 
                            inv_gaus=inv_gaus, deconvolve=deconvolve, deconv_patch=deconv_patch, 
                            patch_size=patch_size, padding=padding, normalize=normalize, use_cuda=use_cuda)
        denoised.append(mic)

        # write the micrograph
        if not output_path:
            if suffix == '' or suffix is None:
                suffix = '.denoised'
            # write the file to the same location as input
            no_ext,ext = os.path.splitext(path)
            outpath = no_ext + suffix + '.' + format
        else:
            outpath = output_path + os.sep + name + suffix + '.' + format
        save_image(mic, outpath) #, mi=None, ma=None)

        count += 1
        print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')
    print('', file=sys.stderr)
    
    return denoised


def denoise_tomogram(path:str, model:Denoise3D, outdir:str=None, suffix:str='', patch_size:int=96, 
                     padding:str=48, batch_size:int=10, volume_num=1, total_volumes=1,
                     gaus=None, verbose:bool=True):
    name = os.path.basename(path)
    
    with open(path, 'rb') as f:
        content = f.read()
    tomo,header,extended_header = mrc.parse(content)
    tomo = tomo.astype(np.float32)

    denoised = model.denoise(tomo, patch_size=patch_size, padding=padding, batch_size=batch_size, 
                             volume_num=volume_num, total_volumes=total_volumes, verbose=verbose)

    # Gaussian filter output
    if gaus is not None:
        tomo = gaus.apply(tomo)

    ## save the denoised tomogram
    if outdir is None:
        # write denoised tomogram to same location as input, but add the suffix
        if suffix is '': # use default
            suffix = '.denoised'
        no_ext,ext = os.path.splitext(path)
        outpath = no_ext + suffix + ext
    else:
        no_ext,ext = os.path.splitext(name)
        outpath = outdir + os.sep + no_ext + suffix + ext

    # use the read header except for a few fields
    header = header._replace(mode=2) # 32-bit real
    header = header._replace(amin=denoised.min())
    header = header._replace(amax=denoised.max())
    header = header._replace(amean=denoised.mean())

    with open(outpath, 'wb') as f:
        mrc.write(f, denoised, header=header, extended_header=extended_header)
        
        
def denoise_tomogram_stream(volumes:List[str], output_path:str, format:str='mrc', suffix:str='', models:List[Any]=None, lowpass:float=1, 
                   pixel_cutoff:float=0, gaus=None, inv_gaus=None, deconvolve:bool=True, deconv_patch:int=1, patch_size:int=1024, 
                   padding:int=500, normalize:bool=True, use_cuda:bool=False):
        # stream the micrographs and denoise them
    total = len(volumes)
    count = 0
    denoised = [] 

    # make the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path in volumes:
        name,_ = os.path.splitext(os.path.basename(path))
        volume = np.array(load_image(path), copy=False).astype(np.float32)

        # process and denoise the micrograph
        #TODO: make sure volumes are being counted
        #TODO: double check method signatures
        volume = denoise_tomogram(volume, models, lowpass=lowpass, cutoff=pixel_cutoff, gaus=gaus, 
                                  inv_gaus=inv_gaus, deconvolve=deconvolve, deconv_patch=deconv_patch, 
                                  patch_size=patch_size, padding=padding, normalize=normalize, use_cuda=use_cuda)
        denoised.append(volume)
        
        count += 1
        print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')
    print('', file=sys.stderr)
    
    return denoised