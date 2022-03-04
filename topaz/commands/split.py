from __future__ import print_function,division

import sys
import os
import pandas as pd
import numpy as np

import topaz.utils.star as star
import topaz.utils.files as file_utils

name = 'split'
help = 'split particle file containing coordinates for multiple micrographs into one file per micrograph'


def add_arguments(parser):
    parser.add_argument('file', help='path to input particle file')
    parser.add_argument('-o', '--output', help='path to output directory')

    parser.add_argument('--format', dest='_from', choices=['auto', 'coord', 'star'], default='auto'
                       , help='file format of the INPUT file. outputs will be written in the same format. (default: detect format automatically based on file extension)')

    parser.add_argument('--suffix', default='', help='suffix to append to file names (default: none)')

    # arguments for file format specific parameters
    parser.add_argument('-t', '--threshold', type=float, default=-np.inf, help='threshold the particles by score (optional)')

    return parser

def main(args):

    fmt = args._from

    # detect the input file formats
    path = args.file
    if fmt == 'auto':
        try:
            fmt = file_utils.detect_format(path)
        except file_utils.UnknownFormatError as e:
            print('Error: unrecognized input coordinates file extension ('+e.ext+')', file=sys.stderr)
            sys.exit(1)
    _,ext = os.path.splitext(path)
    
    suffix = args.suffix

    t = args.threshold
    base = args.output
    if fmt == 'star':
        with open(path, 'r') as f:
            table = star.parse(f)
        # apply score threshold
        if star.SCORE_COLUMN_NAME in table.columns:
            table = table.loc[table[star.SCORE_COLUMN_NAME] >= t]

        # write per micrograph files
        for image_name,group in table.groupby('MicrographName'):
            image_name,_ = os.path.splitext(image_name)
            path = base + '/' + image_name + suffix + ext
            with open(path, 'w') as f:
                star.write(group, f)
    else: # format is coordinate table
        table = pd.read_csv(path, sep='\t')
        if 'score' in table.columns:
            table = table.loc[table['score'] >= t]
        # write per micrograph files
        for image_name,group in table.groupby('image_name'):
            path = base + '/' + image_name + suffix + ext
            group.to_csv(path, sep='\t', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script to ' + help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

