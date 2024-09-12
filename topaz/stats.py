from __future__ import absolute_import, division, print_function

import json
import multiprocessing as mp
import os
import sys
from typing import List

import numpy as np
import scipy.stats

import torch
from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample, save_image


def pixels_given_radius(radius, dims=2):
    '''Given a radius, calculate the number of pixels in a particle.'''
    grid = np.linspace(-radius, radius, 2*radius+1)
    xx,yy,zz = np.meshgrid(grid, grid, grid)
    d2 = xx**2 + yy**2
    if dims == 3:
        d2 += zz**2
    mask = (d2 <= radius**2).astype(int)
    return mask.sum()


def calculate_pi(expected_num_particles, radius, total_pixels, dims=2):
    '''Given the expected number of particles in dataset and radius, calculate what pi should be.'''
    pixels_per_particle = pixels_given_radius(radius, dims=dims)
    # total_regions is number of pixels in the data
    pi = pixels_per_particle*expected_num_particles / total_pixels
    return pi


def normalize(x, alpha=900, beta=1, num_iters=100, sample=1
             , method='gmm', use_cuda=False, verbose=False):
    if method == 'affine':
        mu = x.mean()
        std = x.std()
        mu = float(mu)
        std = float(std)
        metadata = {'mu': mu, 'std': std, 'pi': 1}
        x = (x - mu)/std
        x = x.astype(np.float32)
        return x, metadata


    # normalizes x using GMM

    # fit the parameters of the model
    x_sample = x
    scale = 1
    if sample > 1:
        # estimate parameters using sample from x
        n = int(np.round(x.size/sample))
        scale = x.size/n
        x_sample = np.random.choice(x.ravel(), size=n, replace=False)

    mu, std, pi, logp, mus, stds, pis, logps = norm_fit(x_sample, alpha=alpha, beta=beta
                                                       , scale=scale
                                                       , num_iters=num_iters, use_cuda=use_cuda
                                                       , verbose=verbose)

    # normalize the data
    x = (x - mu)/std
    x = x.astype(np.float32)

    # also record the metadata
    metadata = {'mu': mu
               ,'std': std
               ,'pi': pi
               ,'logp': logp
               ,'mus': mus
               ,'stds': stds
               ,'pis': pis
               ,'logps': logps
               ,'alpha': alpha
               ,'beta': beta
               ,'sample': sample
               }

    return x, metadata


def norm_fit(x, alpha=900, beta=1, scale=1
            , num_iters=100, use_cuda=False, verbose=False):
    
    # try multiple initializations of pi
    pis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1])
    splits = np.quantile(x, 1-pis)

    logps = np.zeros(len(pis))
    mus = np.zeros(len(pis))
    stds = np.zeros(len(pis))

    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()
    
    for i in range(len(pis)):
        pi = pis[i]
        split = splits[i]
        if pi == 1: # single component model
            mu = x.mean()
            var = x.var()
            logp = scale*torch.sum(-(x - mu)**2/2/var - 0.5*torch.log(2*np.pi*var)) + scipy.stats.beta.pdf(1, alpha, beta)
        else:
            logp, mu0, var0, mu, var, pi = gmm_fit(x, pi=pi, split=split, alpha=alpha, beta=beta
                                                  , scale=scale, num_iters=num_iters)
        pis[i] = pi.item()
        logps[i] = logp.item()
        mus[i] = mu.item()
        stds[i] = np.sqrt(var.item())
        
    # select normalization parameters with maximum logp
    i = np.argmax(logps)
    
    return mus[i], stds[i], pis[i], logps[i], mus, stds, pis, logps


def gmm_fit(x, pi=0.5, split=None, alpha=0.5, beta=0.5, scale=1
           , tol=1e-3, num_iters=100, share_var=True, verbose=False): 
    # fit 2-component GMM
    # put a beta distribution prior on pi

    mu = torch.mean(x)
    pi = torch.as_tensor(pi)
    
    # split into everything > and everything <= pi pixel value
    # for assigning initial parameters
    if split is None:
        split = np.quantile(x.cpu().numpy(), 1-pi)
    mask = x <= split

    p0 = mask.float()
    p1 = 1 - p0

    mu0 = mu
    s = torch.sum(p0)
    if s > 0:
        mu0 = torch.sum(x*p0)/s

    mu1 = mu
    s = torch.sum(p1)
    if s > 0:
        mu1 = torch.sum(x*p1)/s

    if share_var:
        var = torch.mean(p0*(x - mu0)**2 + p1*(x - mu1)**2)
        var0 = var
        var1 = var
    else:
        var0 = torch.sum(p0*(x - mu0)**2)/torch.sum(p0)
        var1 = torch.sum(p1*(x - mu1)**2)/torch.sum(p1)

    # first, calculate p(k | x, mu, var, pi)
    log_p0 = -(x - mu0)**2/2/var0 - 0.5*torch.log(2*np.pi*var0) + torch.log1p(-pi)
    log_p1 = -(x - mu1)**2/2/var1 - 0.5*torch.log(2*np.pi*var1) + torch.log(pi)
    
    ma = torch.max(log_p0, log_p1)
    Z = ma + torch.log(torch.exp(log_p0 - ma) + torch.exp(log_p1 - ma))
    
    # the probability of the data is
    logp = scale*torch.sum(Z) + scipy.stats.beta.logpdf(pi.cpu().numpy(), alpha, beta)
    logp_cur = logp
    
    for it in range(1, num_iters+1):
        # calculate the assignments
        p0 = torch.exp(log_p0 - Z)
        p1 = torch.exp(log_p1 - Z)
        
        # now, update distribution parameters
        s = torch.sum(p1)
        a = alpha + s
        b = beta + p1.numel() - s
        pi = (a-1)/(a + b - 2) # MAP estimate of pi

        mu0 = mu
        s = torch.sum(p0)
        if s > 0:
            mu0 = torch.sum(x*p0)/s

        mu1 = mu
        s = torch.sum(p1)
        if s > 0:
            mu1 = torch.sum(x*p1)/s

        if share_var:
            var = torch.mean(p0*(x - mu0)**2 + p1*(x - mu1)**2)
            var0 = var
            var1 = var
        else:
            var0 = torch.sum(p0*(x - mu0)**2)/torch.sum(p0)
            var1 = torch.sum(p1*(x - mu1)**2)/torch.sum(p1)

        # recalculate likelihood p(k | x, mu, var, pi)
        log_p0 = -(x - mu0)**2/2/var0 - 0.5*torch.log(2*np.pi*var0) + torch.log1p(-pi)
        log_p1 = -(x - mu1)**2/2/var1 - 0.5*torch.log(2*np.pi*var1) + torch.log(pi)
        
        ma = torch.max(log_p0, log_p1)
        Z = ma + torch.log(torch.exp(log_p0 - ma) + torch.exp(log_p1 - ma))
        
        logp = scale*torch.sum(Z) + scipy.stats.beta.logpdf(pi.cpu().numpy(), alpha, beta)

        if verbose:
            print(it, logp)

        # check for termination
        if logp - logp_cur <= tol:
            break # done
        logp_cur = logp
        
    return logp, mu0, var0, mu1, var1, pi


def gmm_fit_numpy(x, pi=0.5, alpha=0.5, beta=0.5, tol=1e-3, num_iters=50, verbose=False): 
    # fit 2-component GMM
    # put a beta distribution prior on pi
    
    # split into everything > and everything <= pi pixel value
    # for assigning initial parameters
    split = np.quantile(x, 1-pi)
    mask = x <= split

    p0 = np.zeros(x.shape)
    p0[mask] = 1
    p1 = np.zeros(x.shape)
    p1[~mask] = 1
    
    mu0 = np.average(x, weights=p0)
    mu1 = np.average(x, weights=p1)
    var = np.mean(p0*(x - mu0)**2 + p1*(x - mu1)**2)

    # first, calculate p(k | x, mu, var, pi)
    log_p0 = -(x - mu0)**2/2/var - 0.5*np.log(2*np.pi*var) + np.log1p(-pi)
    log_p1 = -(x - mu1)**2/2/var - 0.5*np.log(2*np.pi*var) + np.log(pi)
    
    ma = np.maximum(log_p0, log_p1)
    Z = ma + np.log(np.exp(log_p0 - ma) + np.exp(log_p1 - ma))
    
    # the probability of the data is
    logp = np.sum(Z) + scipy.stats.beta.logpdf(pi, alpha, beta)
    logp_cur = logp
    
    for it in range(1, num_iters+1):
        # calculate the assignments
        p0 = np.exp(log_p0 - Z)
        p1 = np.exp(log_p1 - Z)
        
        # now, update distribution parameters
        s = np.sum(p1)
        a = alpha + s
        b = beta + p1.size - s
        pi = (a-1)/(a + b - 2) # MAP estimate of pi
        
        mu0 = np.average(x, weights=p0)
        mu1 = np.average(x, weights=p1)
        
        var = np.mean(p0*(x - mu0)**2 + p1*(x - mu1)**2)

        # recalculate likelihood p(k | x, mu, var, pi)
        log_p0 = -(x - mu0)**2/2/var - 0.5*np.log(2*np.pi*var) + np.log1p(-pi)
        log_p1 = -(x - mu1)**2/2/var - 0.5*np.log(2*np.pi*var) + np.log(pi)
        
        ma = np.maximum(log_p0, log_p1)
        Z = ma + np.log(np.exp(log_p0 - ma) + np.exp(log_p1 - ma))
        
        logp = np.sum(Z) + scipy.stats.beta.logpdf(pi, alpha, beta)

        if verbose:
            print(it, logp)

        # check for termination
        if logp - logp_cur <= tol:
            break # done
        logp_cur = logp
        
    return logp, mu0, var, mu1, var, pi


class Normalize:
    def __init__(self, dest, scale, affine, num_iters, alpha, beta
                , sample, metadata, formats, use_cuda):
        self.dest = dest
        self.scale = scale
        self.affine = affine
        self.num_iters = num_iters
        self.alpha = alpha
        self.beta = beta
        self.sample = sample
        self.metadata = metadata
        self.formats = formats
        self.use_cuda = use_cuda

    def __call__(self, path):
        # load the image
        image = load_image(path, make_image=False)
        # check if MRC with header and extender header 
        (image, header, extended_header) = image if type(image) is tuple else (image, None, None)
        x = image.astype(np.float32)

        if self.scale > 1:
            x = downsample(x, self.scale)
            if header:
                # update image size (pixels) in header if present
                new_height, new_width = x.shape
                header = header._replace(ny=new_height)
                header = header._replace(nx=new_width)

        # normalize it
        method = 'gmm'
        if self.affine:
            method = 'affine'
        x,metadata = normalize(x, alpha=self.alpha, beta=self.beta, num_iters=self.num_iters, 
                               method=method, sample=self.sample, use_cuda=self.use_cuda)

        # save the image and the metadata
        name,_ = os.path.splitext(os.path.basename(path))
        base = os.path.join(self.dest, name)
        for f in self.formats:
            save_image(x, base, f=f, header=header, extended_header=extended_header)

        if self.metadata:
            # save the metadata in json format
            mdpath = base + '.metadata.json'
            if not self.affine:
                metadata['mus'] = metadata['mus'].tolist()
                metadata['stds'] = metadata['stds'].tolist()
                metadata['pis'] = metadata['pis'].tolist()
                metadata['logps'] = metadata['logps'].tolist()
            with open(mdpath, 'w') as f:
                json.dump(metadata, f, indent=4)

        return name


def normalize_images(paths:List[str], dest:str, num_workers:int, scale:int, affine:bool, niters:int, alpha:float, 
                     beta:float, sample:int, metadata:bool, formats:List[str], use_cuda:bool, verbose:bool):
    if not os.path.exists(dest):
        os.makedirs(dest)

    process = Normalize(dest, scale, affine, niters, alpha, beta,
                        sample, metadata, formats, use_cuda)

    if num_workers > 1:
        pool = mp.Pool(num_workers)
        for name in pool.imap_unordered(process, paths):
            if verbose:
                print('# processed:', name, file=sys.stderr)
    else:
        for path in paths:
            name = process(path)
            if verbose:
                print('# processed:', name, file=sys.stderr)