from __future__ import print_function,division

import sys
import os
import pandas as pd

import topaz.utils.star as star
from topaz.utils.conversions import star_to_coordinates

name = 'star_to_coordinates'
help = 'convert .star file coordinates to tab delimited coordinates table'


def add_arguments(parser):
    parser.add_argument('file', help='path to input star file')
    parser.add_argument('-o', '--output', help='output file (default: stdout)')
    return parser


def main(args):
    star_to_coordinates(args.file, args.output)    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for converting star file coordinates to tab delimited coordinates table')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)