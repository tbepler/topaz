from __future__ import print_function,division

import sys
import os
import pandas as pd

import topaz.utils.star as star


name = 'star_to_coordinates'
help = 'convert .star file coordinates to tab delimited coordinates table'


def add_arguments(parser):
    parser.add_argument('file', help='path to input star file')
    parser.add_argument('-o', '--output', help='output file (default: stdout)')
    return parser


def strip_ext(name):
    clean_name,ext = os.path.splitext(name)
    return clean_name


def main(args):
    with open(args.file, 'r') as f:
        table = star.parse(f)

    if 'ParticleScore' in table.columns:
        ## columns of interest are 'MicrographName', 'CoordinateX', 'CoordinateY', and 'ParticleScore'
        table = table[['MicrographName', 'CoordinateX', 'CoordinateY', 'ParticleScore']]
        table.columns = ['image_name', 'x_coord', 'y_coord', 'score']
    else:
        ## columns of interest are 'MicrographName', 'CoordinateX', and 'CoordinateY'
        table = table[['MicrographName', 'CoordinateX', 'CoordinateY']]
        table.columns = ['image_name', 'x_coord', 'y_coord']
    ## convert the coordinates to integers
    table['x_coord'] = table['x_coord'].astype(float).astype(int)
    table['y_coord'] = table['y_coord'].astype(float).astype(int)
    ## strip file extensions off the image names if present
    table['image_name'] = table['image_name'].apply(strip_ext) 

    out = sys.stdout
    if args.output is not None:
        out = args.output
    table.to_csv(out, sep='\t', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for converting star file coordinates to tab delimited coordinates table')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


