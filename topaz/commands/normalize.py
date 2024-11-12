from __future__ import print_function

import argparse
import imp
import os
import sys

import numpy as np
from topaz.cuda import set_device
from topaz.stats import normalize_images
from topaz.torch import set_num_threads

name = 'normalize'
help = 'normalize a set of images using the 2-component Gaussian mixture model'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for normalizing a list of images using 2-component Gaussian mixture model')

    parser.add_argument('files', nargs='+')

    parser.add_argument('-s', '--scale', default=1, type=int, help='downsample images by this factor (default: 1)')
    parser.add_argument('--affine', action='store_true', help='use standard normalization (x-mu)/std of whole image rather than GMM normalization')

    parser.add_argument('--sample', default=10, type=int, help='pixel sampling factor for model fit. speeds up estimation of parameters but introduces sample error if set >1. (default: 10)')
    parser.add_argument('--niters', default=100, type=int, help='maximum number of EM iterations to run for model fit (default: 100)')

    parser.add_argument('-a', '--alpha', default=900, type=float, help='alpha parameter of the beta distribution prior on the mixing proportion (default: 900)')
    parser.add_argument('-b', '--beta', default=1, type=float, help='beta parameter of the beta distribution prior on the mixing proportion (default: 1)')

    parser.add_argument('--metadata', action='store_true', help='if set, save parameter metadata for each micrograph')

    parser.add_argument('-d', '--device', default=-1, type=int, help='which device to use, set to -1 to force CPU. >=0 specifies GPU number (default: -1)')
    parser.add_argument('-t', '--num-workers', type=int, default=0, help='number of parallel processes to use, 0 specifies main process only (default: 0)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')

    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('--format', dest='format_', default='mrc', help='image format(s) to write. choices are mrc, tiff, and png. images can be written in multiple formats by specifying each in a comma separated list, e.g. mrc,png would write mrc and png format images (default: mrc)')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    return parser


def main(args):
    formats = args.format_.split(',')

    # set the number of threads
    set_num_threads(args.num_threads)

    # set CUDA device
    use_cuda = set_device(args.device)
    # when using GPU, turn off multiple processes
    num_workers = 0 if use_cuda else args.num_workers

    normalize_images(args.files, args.destdir, num_workers, args.scale, args.affine, args.niters, args.alpha, args.beta,
                        args.sample, args.metadata, formats, use_cuda, args.verbose)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
