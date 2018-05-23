from __future__ import print_function,division

import sys
import os
import pandas as pd
from PIL import Image
import glob

name = 'boxes_to_coordinates'
help = 'convert .box format coordinates to tab delimited coordinates table'

def add_arguments(parser):
    parser.add_argument('files', nargs='+', help='path to input box files')
    parser.add_argument('--imagedir', help='directory of images, eman2 inverts the y-axis')
    parser.add_argument('--image-ext', default='*', help='image format extension, * corresponds to matching the first image file with the same name as the box file (default: *)')
    parser.add_argument('-o', '--output', help='destination file (default: stdout)')

def main(args):
    from topaz.utils.conversions import boxes_to_coordinates
    from topaz.utils.data.loader import load_image

    tables = []

    for path in args.files:
        if os.path.getsize(path) == 0:
            continue

        image_name = os.path.splitext(os.path.basename(path))[0]
        impath = os.path.join(args.imagedir, image_name) + '.' + args.image_ext
        print(impath)
        # use glob incase image_ext is '*'
        impath = glob.glob(impath)[0]

        im = load_image(impath)
        shape = (im.height,im.width)
        box = pd.read_csv(path, sep='\t', header=None).values

        coords = boxes_to_coordinates(box, shape, image_name=image_name)

        tables.append(coords)

    table = pd.concat(tables, axis=0)

    output = sys.stdout
    if args.output is not None:
        output = args.output
    table.to_csv(output, sep='\t', index=False)




