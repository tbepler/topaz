from __future__ import print_function,division
from posixpath import split

import sys
import os
import pandas as pd
import numpy as np
import argparse

import topaz.utils.star as star
from topaz.utils.files import split_particle_file

name = 'split'
help = 'split particle file containing coordinates for multiple micrographs into one file per micrograph'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script to ' + help)

    parser.add_argument('file', help='path to input particle file')
    parser.add_argument('-o', '--output', help='path to output directory')

    parser.add_argument('--format', dest='_from', choices=['auto', 'coord', 'star'], default='auto'
                       , help='file format of the INPUT file. outputs will be written in the same format. (default: detect format automatically based on file extension)')

    parser.add_argument('--suffix', default='', help='suffix to append to file names (default: none)')

    # arguments for file format specific parameters
    parser.add_argument('-t', '--threshold', type=float, default=-np.inf, help='threshold the particles by score (optional)')

    return parser


def main(args):
    split_particle_file(args.file, args._from, args.suffix, args.threshold, args.output)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)

