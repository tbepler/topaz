from __future__ import division, print_function

import numpy as np
from PIL import Image
import os

import topaz.mrc as mrc

def downsample(x, factor):
    """ Downsample 2d array using fourier transform """

    m,n = x.shape[-2:]

    F = np.fft.rfft2(x)
    #F = np.fft.fftshift(F)

    S = 2*factor
    A = F[...,0:m//S+1,0:n//S+1]
    B = F[...,-m//S:,0:n//S+1]

    F = np.concatenate([A,B], axis=-2)
    
    f = np.fft.irfft2(F)

    return f

def quantize(x, mi=-3, ma=3, dtype=np.uint8):
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x
    #buckets = np.linspace(mi, ma, 255)
    #return np.digitize(x, buckets).astype(dtype)

def unquantize(x, mi=-3, ma=3, dtype=np.float32):
    """ convert quantized image array back to approximate unquantized values """
    x = x.astype(dtype)
    y = x*(ma-mi)/255 + mi
    return y

def save_image(x, path, f=None, verbose=False):
    if f is None:
        f = os.path.splitext(path)[1]
        f = f[1:] # remove the period
    else:
        path = path + '.' + f

    if verbose:
        print('# saving:', path)

    if f == 'mrc':
        save_mrc(x, path)
    elif f == 'tiff':
        save_tiff(x, path)
    elif f == 'png':
        save_png(x, path)

def save_mrc(x, path):
    with open(path, 'wb') as f:
        x = x[np.newaxis] # need to add z-axis for mrc write
        mrc.write(f, x)

def save_tiff(x, path):
    im = Image.fromarray(x) 
    im.save(path, 'tiff')

def save_png(x, path):
    # byte encode the image
    im = Image.fromarray(quantize(x))
    im.save(path, 'png')



