from __future__ import print_function,division

import sys
import os
import pandas as pd


name = 'star_to_coordinates'
help = 'convert star file coordinates to tab delimited coordinates table'


def add_arguments(parser):
    parser.add_argument('file', help='path to input star file')
    parser.add_argument('-o', '--output', help='output file (default: stdout)')
    return parser


def parse_star(f):
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('data_images'): 
            return parse_star_body(lines[i+1:])


def parse_star_body(lines):
    ## data_images line has been read, next is loop
    for i in range(len(lines)):
        if lines[i].startswith('loop_'):
            lines = lines[i+1:]
            break
    header,lines = parse_star_loop(lines)
    ## parse the body
    content = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('data'): # done with image block
            break
        if line.startswith('#') or line.startswith(';'): # comment lines
            continue
        if line != '':
            tokens = line.split()
            content.append(tokens)

    return pd.DataFrame(content, columns=header)


def parse_star_loop(lines):
    columns = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line.startswith('_'):
            break
        name = line[1:]
        # strip trailing comments from name
        loc = name.find('#')
        if loc >= 0:
            name = name[:loc]
        name = name.strip()
        columns.append(name)
    return columns, lines[i:]


def strip_ext(name):
    clean_name,ext = os.path.splitext(name)
    return clean_name


def main(args):
    with open(args.file, 'r') as f:
        table = parse_star(f)

    ## columns of interest are 'rlnMicrographName', 'rlnCoordinateX', and 'rlnCoordinateY'
    table = table[['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY']]
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


