from __future__ import division, print_function

import argparse

import numpy as np
from topaz.utils.picks import create_particle_stack

name = 'particle_stack'
help = 'extract mrc particle stack given coordinates table'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for extracting mrc stack from particle coordinates')

    parser.add_argument('file', help='path to input coordinates file')
    parser.add_argument('--image-root', help='root directory of the micrograph files')
    parser.add_argument('-o', '--output', help='path to write particle stack file')

    parser.add_argument('--size', type=int, help='size of particle stack images')
    parser.add_argument('--threshold', type=float, default=-np.inf, help='only take particles with scores >= this value (default: -inf)')
    parser.add_argument('--resize', default=-1, type=int, help='rescaled particle stack size. downsamples particle images from size to resize pixels. (default: off)')

    parser.add_argument('--image-ext', default='.mrc', help='image file extension (default=.mrc)')

    parser.add_argument('--metadata', help='path to .star file containing per-micrograph metadata, e.g. CTF parameters (optional)')

    return parser


def main(args):
    create_particle_stack(args.file, args.output, args.threshold, args.size, 
                          args.resize, args.image_root, args.image_ext, args.metadata)
    

if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
