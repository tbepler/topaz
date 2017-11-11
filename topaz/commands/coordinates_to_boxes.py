from __future__ import print_function,division

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image

from topaz.utils.conversions import coordinates_to_boxes


name = 'coordinates_to_boxes'
help = 'convert coordinates table to .box format files per image'


def add_arguments(parser):
    parser.add_argument('paths', nargs='+',  help='path to input coordinates file')
    parser.add_argument('--boxsize', required=True, type=int, help='size of particle boxes')
    parser.add_argument('--destdir', required=True, help='directory to write per image files')
    parser.add_argument('--imagedir', required=True, help='directory of images, boxfiles index from lower-left so y-axis needs to be inverted')
    return parser


def main(args):
    dfs = []
    for path in args.paths:
        coords = pd.read_csv(path, sep='\t')
        dfs.append(coords)
    coords = pd.concat(dfs, axis=0)

    coords = coords.drop_duplicates()
    print(len(coords))

    if not os.path.exists(args.destdir):
        os.makedirs(args.destdir)

    for image_name,group in coords.groupby('image_name'):
        path = args.destdir + '/' + image_name + '.box'

        im = Image.open(args.imagedir + '/' + image_name + '.tiff')
        shape = (im.height, im.width)

        xy = group[['x_coord', 'y_coord']].values.astype(np.int32)
        boxes = coordinates_to_boxes(xy, shape, args.boxsize, args.boxsize)
        boxes = pd.DataFrame(boxes)

        boxes.to_csv(path, sep='\t', header=False, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for converting coordinates for images in one file to multiple files')
    add_argument(parser)
    args = parser.parse_args()
    main(args)


