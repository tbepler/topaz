from __future__ import print_function,division

import os


import numpy as np
import pandas as pd

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for splitting coordinates table and image list into train and test sets')

    parser.add_argument('--images')
    parser.add_argument('--targets')
    parser.add_argument('-n', '--number', type=int, help='number of images to put into test set')
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random = np.random.RandomState(seed)

    n = args.number

    ## load the data
    images = pd.read_csv(args.images, sep='\t') # image file list
    targets = pd.read_csv(args.targets, sep='\t') # particle coordinates file


    order = random.permutation(len(images))
    images = images.iloc[order]
    test_images = images.iloc[:n]
    train_images = images.iloc[n:]

    test_images = test_images.sort_values('image_name')
    train_images = train_images.sort_values('image_name')

    test_targets = targets.loc[targets.image_name.isin(test_images.image_name)]
    train_targets = targets.loc[targets.image_name.isin(train_images.image_name)]

    image_prefix,image_ext = os.path.splitext(args.images)
    target_prefix,target_ext = os.path.splitext(args.targets)

    test_images.to_csv(image_prefix + '_test' + image_ext, sep='\t', index=False)
    test_targets.to_csv(target_prefix + '_test' + target_ext, sep='\t', index=False)

    train_images.to_csv(image_prefix + '_train' + image_ext, sep='\t', index=False)
    train_targets.to_csv(target_prefix + '_train' + target_ext, sep='\t', index=False)




