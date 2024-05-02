#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import argparse

import torch

from topaz.utils.data.loader import load_image
import topaz.gpu
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

name = 'segment'
help = 'segment images using a trained region classifier'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for segmenting images using a trained model.')

    parser.add_argument('paths', nargs='+', help='paths to image files for processing')

    parser.add_argument('-m', '--model', default='resnet16', help='path to trained classifier. uses the pretrained resnet16 model by default.')
    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU (default: GPU if available)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    return parser


def main(args):
    verbose = args.verbose

    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    ## set the device
    device = topaz.gpu.set_device(args.device)

    ## load the model
    from topaz.model.factory import load_model
    model = load_model(args.model)
    model.eval()
    model.fill()

    model.to(device)
#    if 'ipex' in dir():
#        model = ipex.optimize(model)

    ## make output directory if doesn't exist
    destdir = args.destdir 
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    ## load the images and process with the model
    for path in args.paths:
        basename = os.path.basename(path)
        image_name = os.path.splitext(basename)[0]
        image = load_image(path)

        ## process image with the model
        with torch.no_grad():
            X = torch.from_numpy(np.array(image, copy=False)).unsqueeze(0).unsqueeze(0)
            X = X.to(device)
            score = model(X).data[0,0].cpu().numpy()
        
        im = Image.fromarray(score) 
        path = os.path.join(destdir, image_name) + '.tiff'
        if verbose:
            print('# saving:', path)
        im.save(path, 'tiff')



if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
