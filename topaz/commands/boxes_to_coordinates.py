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
    parser.add_argument('--invert-y', action='store_true', help='invert (mirror) the y-axis particle coordinates. appears to be necessary for .tiff compatibility with EMAN2')
    parser.add_argument('--imagedir', help='directory of images. only required to invert the y-axis - necessary for particles picked on .tiff images in EMAN2')
    parser.add_argument('--image-ext', default='tiff', help='image format extension, * corresponds to matching the first image file with the same name as the box file (default: tiff)')
    parser.add_argument('-o', '--output', help='destination file (default: stdout)')

def main(args):
    from topaz.utils.conversions import boxes_to_coordinates
    from topaz.utils.data.loader import load_image

    tables = []

    invert_y = args.invert_y

    for path in args.files:
        if os.path.getsize(path) == 0:
            continue

        shape = None
        image_name = os.path.splitext(os.path.basename(path))[0]
        if invert_y:
            impath = os.path.join(args.imagedir, image_name) + '.' + args.image_ext
            # use glob incase image_ext is '*'
            impath = glob.glob(impath)[0]
            im = load_image(impath)
            shape = (im.height,im.width)

        box = pd.read_csv(path, sep='\t', header=None).values

        coords = boxes_to_coordinates(box, shape=shape, invert_y=invert_y, image_name=image_name)

        tables.append(coords)

    table = pd.concat(tables, axis=0)

    output = sys.stdout
    if args.output is not None:
        output = args.output
    table.to_csv(output, sep='\t', index=False)




