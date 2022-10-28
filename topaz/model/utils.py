from __future__ import division, print_function

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from topaz.utils.data.loader import load_image


def insize_from_outsize(layers, outsize):
    """ calculates in input size of a convolution stack given the layers and output size """
    for layer in layers[::-1]:
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size # assume square
            if type(kernel_size) is tuple:
                kernel_size = kernel_size[0]
        else:
            kernel_size = 1
        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) is tuple:
                stride = stride[0]
        else:
            stride = 1
        if hasattr(layer, 'padding'):
            pad = layer.padding
            if type(pad) is tuple:
                pad = pad[0]
        else:
            pad = 0
        if hasattr(layer, 'dilation'):
            dilation = layer.dilation
            if type(dilation) is tuple:
                dilation = dilation[0]
        else:
            dilation = 1

        outsize = (outsize-1)*stride + 1 + (kernel_size-1)*dilation - 2*pad 
    return outsize


def segment_images(model, paths:List[str], output_dir:str, use_cuda:bool, verbose:bool):
    ## make output directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## load the images and process with the model
    for path in paths:
        basename = os.path.basename(path)
        image_name = os.path.splitext(basename)[0]
        image = load_image(path, make_image=False, return_header=False)

        ## process image with the model
        with torch.no_grad():
            X = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            if use_cuda:
                X = X.cuda()
            score = model(X).data[0,0].cpu().numpy()
        
        im = Image.fromarray(score) 
        path = os.path.join(output_dir, image_name) + '.tiff'
        if verbose:
            print('# saving:', path)
        im.save(path, 'tiff')
