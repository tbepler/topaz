from __future__ import division, print_function

import argparse

from topaz.utils.conversions import file_boxes_to_coordinates

name = 'boxes_to_coordinates'
help = 'convert .box format coordinates to tab delimited coordinates table'

def add_arguments(parser):
    parser.add_argument('files', nargs='+', help='path to input box files')
    parser.add_argument('--invert-y', action='store_true', help='invert (mirror) the y-axis particle coordinates. appears to be necessary for .tiff compatibility with EMAN2')
    parser.add_argument('--imagedir', help='directory of images. only required to invert the y-axis - necessary for particles picked on .tiff images in EMAN2')
    parser.add_argument('--image-ext', default='tiff', help='image format extension, * corresponds to matching the first image file with the same name as the box file (default: tiff)')
    parser.add_argument('-o', '--output', help='destination file (default: stdout)')
    return parser


def main(args):
    file_boxes_to_coordinates(args.files, args.imagedir, args.image_ext, args.invert_y, args.output)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(help)
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)