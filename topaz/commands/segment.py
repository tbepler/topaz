#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

import torch

from topaz.utils.data.loader import load_image

name = 'segment'
help = 'segment images using a trained region classifier'

def add_arguments(parser):

    parser.add_argument('paths', nargs='+', help='paths to image files for processing')

    parser.add_argument('-m', '--model', help='path to trained classifier')
    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU (default: GPU if available)')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    return parser


def main(args):
    verbose = args.verbose

    ## set the device
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        torch.cuda.set_device(args.device)

    ## load the model
    model = torch.load(args.model)
    model.eval()
    model.fill()

    if use_cuda:
        model.cuda()

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
            if use_cuda:
                X = X.cuda()
            score = model(X).data[0,0].cpu().numpy()
        
        im = Image.fromarray(score) 
        path = os.path.join(destdir, image_name) + '.tiff'
        if verbose:
            print('# saving:', path)
        im.save(path, 'tiff')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for segmenting images using a trained model.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)




















