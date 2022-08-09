from __future__ import division, print_function

import argparse

from topaz.utils.conversions import file_coordinates_to_eman2_json

name = 'coordinates_to_eman2_json'
help = 'convert coordinates table to EMAN2 json format files per image'


def add_arguments(parser):
    parser.add_argument('paths', nargs='+',  help='path to input coordinates file')
    parser.add_argument('--destdir', required=True, help='directory to write per image files')

    parser.add_argument('--invert-y', action='store_true', help='invert (mirror) the y-axis particle coordinates. appears to be necessary for .tiff compatibility with EMAN2')
    parser.add_argument('--imagedir', help='directory of images. only required to invert the y-axis - necessary for particles picked on .tiff images in EMAN2')
    parser.add_argument('--image-ext', default='tiff', help='image format extension, * corresponds to matching the first image file with the same name as the box file (default: tiff)')

    return parser


def main(args):
    file_coordinates_to_eman2_json(args.paths, args.destdir, args.invert_y, args.imagedir, args.image_ext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(help)
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)