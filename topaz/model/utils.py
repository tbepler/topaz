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


def segment_images(model, paths:List[str], output_dir:str, use_cuda:bool, verbose:bool, patch_size:int=None):
    ## make output directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## load the images and process with the model
    for path in paths:
        basename = os.path.basename(path)
        image_name = os.path.splitext(basename)[0]
        image = load_image(path, make_image=False, return_header=False)
        is_3d = len(image.shape) == 3

        ## process image with the model
        with torch.no_grad():
            # add batch and channel dimensions
            X = torch.from_numpy(image.copy()).unsqueeze(0).unsqueeze(0)
            if patch_size is not None:
                # patches move on and off GPU as processed, returns numpy array
                # use half of receptive field as padding
                score = predict_in_patches(model, X, patch_size=patch_size*2, patch_padding=model.width//2, is_3d=is_3d, use_cuda= use_cuda)
            else:
                if use_cuda:
                    X = X.cuda()
                score = model(X) # torch Tensor
                score = score.cpu().numpy()
            score = score[0,0] # remove added dimensions
        
        path = os.path.join(output_dir, image_name)
        if verbose:
            print('# saving:', path)
        if is_3d:
            np.save(path+'.npy', score)
        else:         
            im = Image.fromarray(score) 
            im.save(path+'.tiff', 'tiff')




def predict_in_patches(model, X, patch_size, is_3d=False, use_cuda=False):
    ''' Predict on an image in patches and reassemble the image. Assumes batch and channel dimensions are present. '''
    # use half of receptive field as padding
    patch_padding = model.width//2
    
    # Split image into smaller patches
    patches = get_patches(X, patch_size, patch_padding=patch_padding, is_3d=is_3d)
    # Predict on the patches
    scores = []
    for patch in patches:
        with torch.no_grad():
            patch = patch.cuda() if use_cuda else patch # send only patch to GPU
            score = model(patch).data[0,0].cpu().numpy()
            score = score[..., patch_padding:-patch_padding, patch_padding:-patch_padding]
            if is_3d:
                score = score[..., patch_padding:-patch_padding, :, :]
        scores.append(score)

    # Reassemble the image
    score = reconstruct_from_patches(scores, X.shape, patch_size, patch_padding=patch_padding, is_3d=is_3d)
    return score


def get_patches(X, patch_size, patch_padding=0, is_3d=False):
    y, x = X.shape[-2:]
    z = X.shape[-3] if is_3d else None
    pad = (patch_padding, patch_padding) * (3 if is_3d else 2)
    X = torch.nn.functional.pad(X, pad)
    # get padded sizes
    y_pad, x_pad = X.shape[-2:]
    z_pad = X.shape[-3] if is_3d else None
    # take steps the size of the actual crop/patch, not including padding
    step_size = patch_size - 2*patch_padding
    patches = []
    # for i in range(0, y_pad, step_size):
    for i in range(0, y, step_size):
    # for i in range(0, y_pad, step_size):
        for j in range(0, x, step_size):
            # Ensure the patch is within the image boundaries (including padding)
            i_end = min(i + patch_size, y_pad)
            j_end = min(j + patch_size, x_pad)
            # i_end = min(i + patch_size, y)
            # j_end = min(j + patch_size, x)           
            if is_3d:
                # for k in range(0, z_pad, step_size):
                for k in range(0, z, step_size):
                    k_end = min(k + patch_size, z_pad)
                    # k_end = min(k + patch_size, z)
                    patch = X[..., k:k_end, i:i_end, j:j_end]
                    if patch.abs().sum() == 0:  # ignore patches that are all zero padding
                        continue
                    patches.append(patch)
            else:
                patch = X[..., i:i_end, j:j_end]
                # ignore patches that are all zero padding
                if patch.abs().sum() == 0:
                    continue
                patches.append(patch)
    return patches



def reconstruct_from_patches(patches, original_shape, patch_size, patch_padding=0, is_3d=False):
    ''' Reassemble patches into an image. Assumes batch and channel dimensions removed. '''
    y, x = original_shape[-2:]
    z = original_shape[-3] if is_3d else None

    step_size = patch_size - patch_padding * 2 # good crop size
    reassembled = np.zeros(original_shape)
    # Reassemble the image
    patch_idx = 0
    for i in range(0, y, step_size):
        for j in range(0, x, step_size):
            if is_3d:
                for k in range(0, z, step_size):
                    patch = patches[patch_idx]
                    reassembled[..., k:k+patch.shape[-3], i:i+patch.shape[-2], j:j+patch.shape[-1]] = patch
                    patch_idx += 1
            else:
                patch = patches[patch_idx]
                reassembled[..., i:i+patch.shape[-2], j:j+patch.shape[-1]] = patch
                patch_idx += 1

    return reassembled