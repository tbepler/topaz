from __future__ import print_function, division

import sys
import pandas as pd
import numpy as np

name = 'scale_coordinates'
help = 'scale particle coordinates for resized images'

def add_arguments(parser):
    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=0.25, type=float, help='scaling factor, coordinates become roughly <scale*x,scale*y> (default: 0.25)')
    parser.add_argument('-o', '--output', help='output file')
    return parser

def main(args):
    ## load picks
    df = pd.read_csv(args.file, sep='\t')

    scale = args.scale

    #scaled_x = df.x_coord/scale
    #scaled_y = df.y_coord/scale

    if 'diameter' in df:
        df['diameter'] = np.ceil(df.diameter*scale).astype(np.int32)
    df['x_coord'] = np.round(df.x_coord*scale).astype(np.int32)
    df['y_coord'] = np.round(df.y_coord*scale).astype(np.int32)
    ## write the scaled df
    out = sys.stdout if args.output is None else open(args.output, 'w')
    df.to_csv(out, sep='\t', header=True, index=False)
    if args.output is not None:
        out.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for rescaling particle coordinates')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


