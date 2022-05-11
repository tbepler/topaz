from __future__ import division, print_function
from ast import arg

import numpy as np
from topaz.utils.star import threshold_star_particles

name = 'star_particles_threshold'
help = 'filter the particles in a .star file by score threshold'


def add_arguments(parser):
    parser.add_argument('file', help='path to input star file')
    parser.add_argument('-o', '--output', help='path to write particle stack file')
    parser.add_argument('-t', '--threshold', type=float, default=-np.inf, help='only take particles with scores >= this value (default: -inf)')
    return parser


def main(args):
    threshold_star_particles(args.file, args.threshold, args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for thresholding particles in a star file')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


