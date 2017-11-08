from __future__ import print_function,division

import sys
import os
import pandas as pd
import json
from PIL import Image


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for converting coordinates for images in one file to multiple files')
    parser.add_argument('paths', nargs='+',  help='path to input coordinates file')
    parser.add_argument('--destdir', help='directory to write per image files')
    parser.add_argument('--imagedir', help='directory of images, EMAN2 indexes from lower-left, so y-axis must be inverted')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dfs = []
    for path in args.paths:
        coords = pd.read_csv(path, sep='\t')
        dfs.append(coords)
    coords = pd.concat(dfs, axis=0)

    coords = coords.drop_duplicates()
    print(len(coords))

    for image_name,group in coords.groupby('image_name'):
        path = args.destdir + '/' + image_name + '_info.json'

        im = Image.open(args.imagedir + '/' + image_name + '.tiff')
        height = im.height

        boxes = []
        for _,row in group.iterrows():
            boxes.append([row.x_coord, height-1-row.y_coord, "manual"])

        with open(path, 'w') as f:
            json.dump({'boxes': boxes}, f, indent=0)




