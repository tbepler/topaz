from __future__ import division, print_function

import numpy as np

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

def quantize(x, mi=-4, ma=4, dtype=np.uint8):
    buckets = np.linspace(mi, ma, 255)
    return np.digitize(x, buckets).astype(dtype)

def unquantize(x, mi=-4, ma=4, dtype=np.float32):
    """ convert quantized image array back to approximate unquantized values """
    x = x.astype(dtype)
    y = x*(ma-mi)/255 + mi
    return y

    

