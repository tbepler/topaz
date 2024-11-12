import sys
import os
import glob
import pandas as pd
import numpy as np
import argparse

import topaz.utils.files as file_utils
from topaz.utils.files import get_image_path

def train_test_split_micrographs(seed, n, path, format, image_dir, file_ext):
    random = np.random.RandomState(seed)

    ## load the labels
    coords = file_utils.read_coordinates(path, format=format)

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
    root = image_dir
    ext = file_ext

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
    root = os.path.dirname(path)
    basename = os.path.splitext(path)[0]

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
        
    return image_list_train, image_list_test, targets_train, targets_test