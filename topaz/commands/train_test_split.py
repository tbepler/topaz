from __future__ import print_function,division

import sys
import os
import glob
import pandas as pd
import numpy as np

import topaz.utils.files as file_utils

name = 'train_test_split'
help = 'split micrographs with labeled particles into train/test sets'


def add_arguments(parser):
    parser.add_argument('file', help='path to particle file')
    parser.add_argument('--image-dir', help='path to images directory')
    parser.add_argument('--image-ext', default='*', help='extension of images (default: auto detect)')

    parser.add_argument('--format', dest='format_', choices=['auto', 'coord', 'csv', 'star', 'box'], default='auto'
                       , help='file format of the particle file (default: detect format automatically based on file extension)')

    parser.add_argument('-n', '--number', type=int, help='number of images to put into test set')
    parser.add_argument('--seed', default=0, type=int, help='random seed used to generate the random split (default: 0)')

    return parser


def get_image_path(image_name, root, ext):
    tmp = root + os.sep + image_name + '.' + ext
    paths = glob.glob(tmp) # candidates...
    if len(paths) > 1:
        print('WARNING: multiple images detected matching to image_name='+image_name, file=sys.stderr)
        # resolve this by taking #1 .tiff, #2 .mrc, #3 .png
        tiff = None
        mrc = None
        png = None
        for path in paths:
            if path.endswith('.tiff'):
                tiff = path
            elif path.endswith('.mrc'):
                mrc = path
            elif path.endswith('.png'):
                png = path
        path = None
        if tiff is not None:
            path = tiff
        elif mrc is not None:
            path = mrc
        elif png is not None:
            path = png
        if path is None:
            print('ERROR: unable to find .tiff, .mrc, or .png image matching to image_name='+image_name, file=sys.stderr)
            sys.exit(1)
    elif len(paths) == 1:
        path = paths[0]
    else:
        # no matches for thie image name
        print('WARNING: no micrograph found matching image name "' + image_name + '". Skipping it.', file=sys.stderr)
        return None


    ## make absolute path
    path = os.path.abspath(path)

    return path


def main(args):

    seed = args.seed
    random = np.random.RandomState(seed)

    n = args.number

    ## load the labels

    path = args.file
    format_ = args.format_
    coords = file_utils.read_coordinates(path, format=format_)

    ## split to coordinates up by image name
    image_names = []
    groups = []
    for name,group in coords.groupby('image_name'):
        image_names.append(name)
        groups.append(group)

    print('# splitting {} micrographs with {} labeled particles into {} train and {} test micrographs'.format(len(image_names), len(coords), len(image_names) - n, n), file=sys.stderr)

    ## randomly split the labels by micrograph
    order = random.permutation(len(image_names))

    image_names_test = []
    groups_test = []
    for i in range(n):
        j = order[i]
        image_names_test.append(image_names[j])
        groups_test.append(groups[j])

    image_names_train = []
    groups_train = []
    for i in range(n, len(image_names)):
        j = order[i]
        image_names_train.append(image_names[j])
        groups_train.append(groups[j])
    
    targets_train = pd.concat(groups_train, 0)
    targets_test = pd.concat(groups_test, 0)


    ## if the image-dir is specified, make the image list files
    root = args.image_dir
    ext = args.image_ext

    paths_train = []
    for image_name in image_names_train:
        path = get_image_path(image_name, root, ext)
        if path is not None:
            paths_train.append(path)

    paths_test = []
    for image_name in image_names_test:
        path = get_image_path(image_name, root, ext)
        if path is not None:
            paths_test.append(path)

    image_list_train = pd.DataFrame({'image_name': image_names_train, 'path': paths_train})
    image_list_test = pd.DataFrame({'image_name': image_names_test, 'path': paths_test})


    ## write the files to the same location as the original labels
    root = os.path.dirname(args.file)
    basename = os.path.splitext(args.file)[0]

    ## write the split targets table
    path = basename + '_train.txt'
    print('# writing:', path, file=sys.stderr)
    targets_train.to_csv(path, sep='\t', index=False)

    path = basename + '_test.txt'
    print('# writing:', path, file=sys.stderr)
    targets_test.to_csv(path, sep='\t', index=False)

    ## write the image list tables
    path = root + os.sep + 'image_list_train.txt'
    print('# writing:', path, file=sys.stderr)
    image_list_train.to_csv(path, sep='\t', index=False)

    path = root + os.sep + 'image_list_test.txt'
    print('# writing:', path, file=sys.stderr)
    image_list_test.to_csv(path, sep='\t', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script to ' + help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

