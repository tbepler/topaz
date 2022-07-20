from __future__ import print_function

import argparse

from topaz.utils.image import downsample_file

name = 'downsample'
help = 'downsample micrographs with truncated DFT'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=4, type=int, help='downsampling factor (default: 4)')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='print info')
    return parser


def main(args):
    small = downsample_file(args.file, args.scale, args.output, args.verbose)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)