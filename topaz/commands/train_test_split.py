from __future__ import print_function,division

import argparse
from topaz.utils.data.train_test_split_micrographs import train_test_split_micrographs


name = 'train_test_split'
help = 'split micrographs with labeled particles into train/test sets'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script to ' + help)
    
    parser.add_argument('file', help='path to particle file')
    parser.add_argument('--image-dir', help='path to images directory')
    parser.add_argument('--image-ext', default='*', help='extension of images (default: auto detect)')

    parser.add_argument('--format', dest='format_', choices=['auto', 'coord', 'csv', 'star', 'box'], default='auto'
                       , help='file format of the particle file (default: detect format automatically based on file extension)')

    parser.add_argument('-n', '--number', type=int, help='number of images to put into test set')
    parser.add_argument('--seed', default=0, type=int, help='random seed used to generate the random split (default: 0)')

    return parser


def main(args):
    image_list_train, image_list_test, targets_train, targets_test =\
        train_test_split_micrographs(args.seed, args.number, args.file, args.format_,
                                     args.image_dir, args.image_ext)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)