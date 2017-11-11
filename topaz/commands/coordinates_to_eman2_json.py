from __future__ import print_function,division

import sys
import os
import numpy as np
import pandas as pd
import json
from PIL import Image

from topaz.utils.conversions import coordinates_to_eman2_json


name = 'coordinates_to_eman2_json'
help = 'convert coordinates table to EMAN2 json format files per image'


def add_arguments(parser):
    parser.add_argument('paths', nargs='+',  help='path to input coordinates file')
    parser.add_argument('--destdir', required=True, help='directory to write per image files')
    parser.add_argument('--imagedir', required=True, help='directory of images, EMAN2 indexes from lower-left, so y-axis must be inverted')
    return parser


if __name__ == '__main__':
    args = parse_args()

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
        path = args.destdir + '/' + image_name + '_info.json'

        im = Image.open(args.imagedir + '/' + image_name + '.tiff')
        shape = (im.height, im.width)
        xy = group[['x_coord','y_coord']].values.astype(int)

        boxes = coordinates_to_eman2_json(xy, shape)

        with open(path, 'w') as f:
            json.dump({'boxes': boxes}, f, indent=0)




