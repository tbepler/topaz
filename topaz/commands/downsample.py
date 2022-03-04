from __future__ import print_function

import sys
import numpy as np
from PIL import Image # for saving images

from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample

name = 'downsample'
help = 'downsample micrographs with truncated DFT'

def add_arguments(parser):
    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=4, type=int, help='downsampling factor (default: 4)')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print info')
    return parser

def main(args):
    ## load image
    path = args.file
    im = load_image(path)
    # convert PIL image to array
    im = np.array(im, copy=False).astype(np.float32)

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
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


