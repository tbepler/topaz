from __future__ import absolute_import, division, print_function

import os
import sys
from typing import List, Union

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
from topaz import mrc
from topaz.denoising.datasets import DenoiseDataset, PatchDataset
from topaz.denoising.models import load_model, train_model
from topaz.filters import (AffineFilter, GaussianDenoise, InvGaussianFilter,
                           gaussian_filter, inverse_filter)
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

    
    def __call__(self, input:Union[np.ndarray, torch.Tensor]):
        self._denoise(input)
 
    
    def train(self, train_dataset:DenoiseDataset, val_dataset:DenoiseDataset, loss_fn:str='L2', optim:str='adam', lr:float=0.001, weight_decay:float=0, batch_size:int=10, num_epochs:int=500, 
                        shuffle:bool=True, num_workers:int=1, verbose:bool=True, save_best:bool=False, save_interval:int=None, save_prefix:str=None) -> None:
        train_model(self.model, train_dataset, val_dataset, loss_fn, optim, lr, weight_decay, batch_size, num_epochs, shuffle, self.use_cuda, num_workers, verbose, save_best, save_interval, save_prefix)

    
    @torch.no_grad()        
    def _denoise(self, input:Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        '''Call stored denoising model.
        '''
        self.model.eval()
        # convert to tensor if necessary, move to device
        input = torch.from_numpy(input) if type(input) == np.ndarray else input
        input = input.to(self.device)
        # normalize, add singleton batch and input channel dims 
        mu, std = input.mean(), input.std()
        input = (input - mu) / std 
        input = input.unsqueeze(0).unsqueeze(0)
        # predict, remove extra dims
        pred = self.model(input).squeeze()
        # unnormalize
        pred = pred * std + mu
        return pred.cpu().numpy()
 

    @torch.no_grad()
    def denoise_patches(self, x:Union[np.ndarray, torch.Tensor], patch_size:int, padding:int=128) -> np.ndarray:
        ''' Denoise micrograph patches.
        '''
        x = torch.from_numpy(x) if type(x) == np.ndarray else x
        y = torch.zeros_like(x)

        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                # include padding extra pixels on either side
                si = max(0, i - padding)
                ei = min(x.size(2), i + patch_size + padding)
                sj = max(0, j - padding)
                ej = min(x.size(3), j + patch_size + padding)

                xij = x[:,:,si:ei,sj:ej]
                yij = self._denoise(xij) # denoise the patch

                # match back without the padding
                si = i - si
                sj = j - sj

                y[i:i+patch_size,j:j+patch_size] = yij[si:si+patch_size,sj:sj+patch_size]
        y = y.squeeze().cpu().numpy()
        return y


    @torch.no_grad()
    def denoise(self, x:Union[np.ndarray, torch.Tensor], patch_size=-1, padding=128):
        s = patch_size + padding  # check the patch plus padding size
        use_patch = (patch_size > 0) and (s < x.shape[0] or s < x.shape[1]) # must denoise in patches
        result = self.denoise_patches(x, patch_size, padding=padding) if use_patch else self._denoise(x)
        return result
  


class Denoise3D(Denoise):
    ''' Object for denoising tomograms.
    '''
    @torch.no_grad()
    def denoise(self, tomo:np.ndarray, patch_size:int=96, padding:int=48, batch_size:int=1, 
                volume_num:int=1, total_volumes:int=1, verbose:bool=True) -> np.ndarray:
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



#2D Denoising Functions
def denoise_image(mic:np.ndarray, models:List[Denoise], lowpass=1, cutoff=0, gaus:GaussianDenoise=None, inv_gaus:InvGaussianFilter=None, deconvolve=False, 
                    deconv_patch=1, patch_size=-1, padding=0, normalize=False, use_cuda=False) -> np.ndarray:
    ''' Denoise micrograph using (pre-)trained neural networks and various filters.
    '''
    mic = lowpass(mic, lowpass) if lowpass > 1 else mic
    # normalize and remove outliers
    mu, std = mic.mean(), mic.std()
    x = (mic - mu)/std
    if cutoff > 0:
        x[(x < -cutoff) | (x > cutoff)] = 0

    # convert to tensor and move to device
    mic = torch.from_numpy(mic.copy())
    mic = mic.cuda() if use_cuda else mic

    # apply guassian/inverse gaussian filter
    if gaus is not None:
        x = gaus.apply(x)
    elif inv_gaus is not None:
        x = inv_gaus.apply(x)
    elif deconvolve:
        # estimate optimal filter and correct spatial correlation
        x = correct_spatial_covariance(x, patch=deconv_patch)

    # denoise
    mic = sum( [model.denoise(x, patch_size=patch_size, padding=padding) for model in models] ) / len(models)

    if normalize:
        # restore pixel scaling
        mic = (mic - mic.mean())/mic.std()
    else:
        # add back std. dev. and mean
        mic = std*mic + mu

    return mic


def denoise_stack(path:str, output_path:str, models:List[Denoise], lowpass:float=1, pixel_cutoff:float=0, 
                  gaus=None, inv_gaus=None, deconvolve:bool=True, deconv_patch:int=1, patch_size:int=1024, 
                  padding:int=500, normalize:bool=True, use_cuda:bool=False):
    with open(path, 'rb') as f:
        content = f.read()
    stack,header,extended_header = mrc.parse(content)
    print('# denoising stack with shape:', stack.shape, file=sys.stderr)
    total = len(stack)
    count = 0

    denoised = np.zeros_like(stack)
    for i in range(len(stack)):
        mic = stack[i]
        # process and denoise the micrograph
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
        mrc.write(f, denoised, header=header, extender_header=extended_header)
    
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
        image = load_image(path, make_image=False)
        # check if MRC with header and extender header 
        (image, header, extended_header) = image if type(image) is tuple else (image, None, None)

        # process and denoise the micrograph
        mic = denoise_image(image, models, lowpass=lowpass, cutoff=pixel_cutoff, gaus=gaus, 
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
        save_image(mic, outpath, header=header, extended_header=extended_header) #, mi=None, ma=None)

        count += 1
        print(f'# {count} of {total} completed.', file=sys.stderr, end='\r')
    print('', file=sys.stderr)
    
    return denoised



#3D Denoising Functions
def denoise_tomogram(path:str, model:Denoise3D, outdir:str=None, suffix:str='', patch_size:int=96, padding:int=48, 
                     volume_num:int=1, total_volumes:int=1, gaus:GaussianDenoise=None, verbose:bool=True):
    name = os.path.basename(path)
    
    with open(path, 'rb') as f:
        content = f.read()
    tomo,header,extended_header = mrc.parse(content)
    tomo = tomo.astype(np.float32)

    # Use train or pre-trained model to denoise
    denoised = model.denoise(tomo, patch_size=patch_size, padding=padding, batch_size=1, 
                             volume_num=volume_num, total_volumes=total_volumes, verbose=verbose)

    # Gaussian filter output
    tomo = gaus.apply(tomo) if gaus is not None else tomo

    ## save the denoised tomogram
    if outdir is None:
        # write denoised tomogram to same location as input, but add the suffix
        if suffix == '': # use default
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
    return tomo
        
        
def denoise_tomogram_stream(volumes:List[str], model:Denoise3D, output_path:str, suffix:str='', gaus:float=None, 
                            patch_size:int=96, padding:int=48, verbose:bool=True, use_cuda:bool=False):
        # stream the micrographs and denoise them
    total = len(volumes)
    count = 0
    denoised = [] 

    # make the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create Gaussian filter for post-processing
    gaus = GaussianDenoise(gaus, use_cuda=use_cuda) if gaus > 0 else None

    for idx, path in enumerate(volumes):
        volume = denoise_tomogram(path, model, outdir=output_path, suffix=suffix, patch_size=patch_size, padding=padding, 
                                  volume_num=idx, total_volumes=total, gaus=gaus, verbose=verbose)
        denoised.append(volume)
        
        count += 1
        print(f'# {count} of {total} completed.', file=sys.stderr, end='\r')
    print('', file=sys.stderr)
    
    return denoised