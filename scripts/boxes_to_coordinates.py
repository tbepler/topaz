from __future__ import print_function,division

import sys
import os
import pandas as pd
from PIL import Image


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for converting box file coordinates to tab delimited table')
    parser.add_argument('files', nargs='+', help='path to input box files')
    parser.add_argument('--imagedir', help='directory of images, eman2 inverts the y-axis')
    parser.add_argument('--image-ext', default='tiff', help='image format extension (default: tiff)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    tables = []

    for path in args.files:
        if os.path.getsize(path) == 0:
            continue

        image_name = os.path.splitext(os.path.basename(path))[0]

        im = Image.open(args.imagedir + '/' + image_name + '.' + args.image_ext)
        im_height = im.height

        box = pd.read_csv(path, sep='\t', header=None) 
        ## first 2 columns are x and y coordinates of lower left box corners
        ## requires knowing image size to inver y-axis
        ## to conform with origin in upper-left rather than lower-left
        ## next 2 columns are width and height
        x_lo = box[0]
        y_lo = box[1]
        width = box[2]
        height = box[3]
        x_coord = x_lo + width//2
        y_coord = (im_height-1-y_lo) - height//2

        table = pd.DataFrame({'image_name': [image_name]*len(box)})
        table['x_coord'] = x_coord
        table['y_coord'] = y_coord

        tables.append(table)

    table = pd.concat(tables, axis=0)

    table.to_csv(sys.stdout, sep='\t', index=False)




