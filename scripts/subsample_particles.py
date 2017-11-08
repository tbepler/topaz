from __future__ import print_function,division

import os
import sys

import numpy as np
import pandas as pd

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for subsampling particles from a coordinates table')
    parser.add_argument('file', help='path to particle coordinates file')
    parser.add_argument('-n', '--number', type=int, help='number of particles to sample')
    parser.add_argument('--seed', default=0, type=int, help='random seed for sampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random = np.random.RandomState(seed)

    n = args.number

    ## load the data
    targets = pd.read_csv(args.file, sep='\t') # particle coordinates file

    order = random.permutation(len(targets))
    sampled_targets = targets.iloc[order[:n]].copy()
    ## resort by image name
    sampled_targets.sort_values('image_name', inplace=True)

    sampled_targets.to_csv(sys.stdout, sep='\t', index=False)



