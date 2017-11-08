from __future__ import print_function, division

import sys
import pandas as pd
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for filtering an image list to only contain images with particles listed in a targets file.')
    parser.add_argument('file')
    parser.add_argument('--targets', help='picks file with images to filter for')
    parser.add_argument('-o', '--output', help='output file')
    return parser.parse_args()

def main(args):

    images = pd.read_csv(args.file, sep='\t')
    targets = pd.read_csv(args.targets, sep='\t')

    images = images.loc[images.image_name.isin(targets.image_name)]

    ## write the filtered images
    out = sys.stdout if args.output is None else open(args.output, 'w')
    images.to_csv(out, sep='\t', header=True, index=False)
    if args.output is not None:
        out.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)


