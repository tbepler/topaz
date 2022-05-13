from __future__ import print_function, division

from topaz.utils.picks import scale_coordinates

name = 'scale_coordinates'
help = 'scale particle coordinates for resized images'

def add_arguments(parser):
    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=0.25, type=float, help='scaling factor, coordinates become roughly <scale*x,scale*y> (default: 0.25)')
    parser.add_argument('-o', '--output', help='output file')
    return parser

def main(args):
    scale_coordinates(args.file, args.scale, args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for rescaling particle coordinates')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


