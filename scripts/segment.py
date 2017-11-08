#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.autograd import Variable

from topaz.utils.data.loader import load_mrc, load_pil

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for segmenting images using a trained model.')

    parser.add_argument('paths', nargs='+', help='paths to image files for processing')

    parser.add_argument('-m', '--model', help='path to trained classifier')
    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU (default: GPU if available)')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    return parser.parse_args()


def load_image(path):
    ext = os.path.splitext(path)[1]
    if ext == 'mrc':
        image = load_mrc(path)
    else:
        image = load_pil(path)
    return image 

if __name__ == '__main__':
    args = parse_args()

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
        X = torch.from_numpy(np.array(image, copy=False)).unsqueeze(0).unsqueeze(0)
        if use_cuda:
            X = X.cuda()
        X = Variable(X, volatile=True)
        score = model(X).data[0,0].cpu().numpy()
        
        im = Image.fromarray(score) 
        path = os.path.join(destdir, image_name) + '.tiff'
        if verbose:
            print('# saving:', path)
        im.save(path, 'tiff')
























