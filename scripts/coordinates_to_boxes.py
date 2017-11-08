from __future__ import print_function,division

import sys
import os
import pandas as pd
from PIL import Image


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for converting coordinates for images in one file to multiple files')
    parser.add_argument('paths', nargs='+',  help='path to input coordinates file')
    parser.add_argument('--width', type=int, help='size of particle boxes')
    parser.add_argument('--destdir', help='directory to write per image files')
    parser.add_argument('--imagedir', help='directory of images, boxfiles index from lower-left so y-axis needs to be inverted')

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

    if not os.path.exists(args.destdir):
        os.makedirs(args.destdir)

    for image_name,group in coords.groupby('image_name'):
        path = args.destdir + '/' + image_name + '.box'

        im = Image.open(args.imagedir + '/' + image_name + '.tiff')
        height = im.height

        boxes = []
        for _,row in group.iterrows():
            boxes.append([row.x_coord, height-1-row.y_coord, args.width, args.width])
        boxes = pd.DataFrame(boxes)

        boxes.to_csv(path, sep='\t', header=False, index=False)




