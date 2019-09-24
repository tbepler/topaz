from __future__ import division, print_function

import numpy as np
from PIL import Image
import os

import topaz.mrc as mrc

def downsample(x, factor=1, shape=None):
    """ Downsample 2d array using fourier transform """

    if shape is None:
        m,n = x.shape[-2:]
        m = int(m/factor)
        n = int(n/factor)
        shape = (m,n)

    F = np.fft.rfft2(x)

    m,n = shape
    A = F[...,0:m//2,0:n//2+1]
    B = F[...,-m//2:,0:n//2+1]
    F = np.concatenate([A,B], axis=0)

    ## scale the signal from downsampling
    a = n*m
    b = x.shape[-2]*x.shape[-1]
    F *= (a/b)

    f = np.fft.irfft2(F, s=shape)

    return f.astype(x.dtype)

def quantize(x, mi=-3, ma=3, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
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

def save_image(x, path, mi=-3, ma=3, f=None, verbose=False):
    if f is None:
        f = os.path.splitext(path)[1]
        f = f[1:] # remove the period
    else:
        path = path + '.' + f

    if verbose:
        print('# saving:', path)

    if f == 'mrc':
        save_mrc(x, path)
    elif f == 'tiff' or f == 'tif':
        save_tiff(x, path)
    elif f == 'png':
        save_png(x, path, mi=mi, ma=ma)
    elif f == 'jpg' or f == 'jpeg':
        save_jpeg(x, path, mi=mi, ma=ma)

def save_mrc(x, path):
    with open(path, 'wb') as f:
        x = x[np.newaxis] # need to add z-axis for mrc write
        mrc.write(f, x)

def save_tiff(x, path):
    im = Image.fromarray(x) 
    im.save(path, 'tiff')

def save_png(x, path, mi=-3, ma=3):
    # byte encode the image
    im = Image.fromarray(quantize(x, mi=mi, ma=ma))
    im.save(path, 'png')

def save_jpeg(x, path, mi=-3, ma=3):
    # byte encode the image
    im = Image.fromarray(quantize(x, mi=mi, ma=ma))
    im.save(path, 'jpeg')




