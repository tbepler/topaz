from __future__ import division, print_function

import numpy as np
from PIL import Image
import os
import sys

import topaz.mrc as mrc
from topaz.utils.data.loader import load_image


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


def downsample_file(path:str, scale:int, output:str, verbose:bool):
    ## load image
    image = load_image(path)
    # check if MRC with header and extender header 
    image, header, extended_header = image if type(image) is tuple else image, None, None
    # convert PIL image to array
    image = np.array(image, copy=False).astype(np.float32)

    small = downsample(image, scale)
    if header:
        # update image size (pixels) in header if present
        new_height, new_width = small.shape
        header.ny, header.nx = new_height, new_width

    if verbose:
        print('Downsample image:', path, file=sys.stderr)
        print('From', image.shape, 'to', small.shape, file=sys.stderr)

    # write the downsampled image
    save_image(small, output, header=header, extended_header=extended_header)
    
    return small


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


def unquantize(x, mi=-3, ma=3, dtype=np.float32):
    """ convert quantized image array back to approximate unquantized values """
    x = x.astype(dtype)
    y = x*(ma-mi)/255 + mi
    return y


def save_image(x, path, mi=-3, ma=3, f=None, verbose=False, header=None, extended_header=None):
    if f is None:
        f = os.path.splitext(path)[1]
        f = f[1:] # remove the period
    else:
        path = path + '.' + f

    if verbose:
        print('# saving:', path)

    if f == 'mrc':
        save_mrc(x, path, header=header, extended_header=extended_header)
    elif f == 'tiff' or f == 'tif':
        save_tiff(x, path)
    elif f == 'png':
        save_png(x, path, mi=mi, ma=ma)
    elif f == 'jpg' or f == 'jpeg':
        save_jpeg(x, path, mi=mi, ma=ma)


def save_mrc(x, path, header=None, extended_header=None):
    with open(path, 'wb') as f:
        x = x[np.newaxis] # need to add z-axis for mrc write
        mrc.write(f, x, header=header, extended_header=extended_header)


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