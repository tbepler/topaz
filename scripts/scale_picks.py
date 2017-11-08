from __future__ import print_function, division

import sys
import pandas as pd
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for rescaling particle coordinates')
    parser.add_argument('file')
    parser.add_argument('-s', '--scale', default=4, type=float, help='downsampling factor')
    parser.add_argument('-o', '--output', help='output file')
    return parser.parse_args()

def main(args):
    ## load picks
    df = pd.read_csv(args.file, sep='\t')
    if 'diameter' in df:
        df['diameter'] = np.ceil(df.diameter/args.scale).astype(np.int32)
    df['x_coord'] = np.round(df.x_coord/args.scale).astype(np.int32)
    df['y_coord'] = np.round(df.y_coord/args.scale).astype(np.int32)
    ## write the scaled df
    out = sys.stdout if args.output is None else open(args.output, 'w')
    df.to_csv(out, sep='\t', header=True, index=False)
    if args.output is not None:
        out.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)


