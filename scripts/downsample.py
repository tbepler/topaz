from __future__ import print_function

import os
import sys
here = os.path.abspath(__file__)
impath = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, impath)

import numpy as np
from PIL import Image # for saving images

from topaz.utils.image import downsample
import topaz.mrc as mrc


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=4, type=int, help='downsampling factor')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print info')
    return parser.parse_args()

def main(args):
    ## load image
    path = args.file
    with open(path, 'rb') as f:
        content = f.read()
    im, header, extended_header = mrc.parse(content)
    #im = (im - header.amean)/header.rms

    scale = args.scale # how much to downscale by
    small = downsample(im, scale)

    if args.verbose:
        print('Downsample image:', path, file=sys.stderr)
        print('From', im.shape, 'to', small.shape, file=sys.stderr)

    # write the downsampled image
    with open(args.output, 'wb') as f:
        im = Image.fromarray(small)
        if small.dtype == np.uint8:
            im.save(f, 'png')
        else:
            im.save(f, 'tiff')


if __name__ == '__main__':
    args = parse_args()
    main(args)


