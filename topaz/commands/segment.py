#!/usr/bin/env python
from __future__ import division, print_function

import argparse

import topaz.cuda
from topaz.model.factory import load_model
from topaz.model.utils import segment_images
from topaz.torch import set_num_threads


name = 'segment'
help = 'segment images using a trained region classifier'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for segmenting images using a trained model.')

    parser.add_argument('paths', nargs='+', help='paths to image files for processing')

    parser.add_argument('-m', '--model', default='resnet16', help='path to trained classifier. uses the pretrained resnet16 (2D) model by default.')
    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU (default: GPU if available)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')
    parser.add_argument('-p', '--patch-size', type=int, default=None, help='size of patches to predict on, None will predict on the whole image (default: None)')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    return parser


def main(args):
    verbose = args.verbose

    # set the number of threads
    num_threads = args.num_threads
    set_num_threads(num_threads)

    ## set the device
    use_cuda = topaz.cuda.set_device(args.device)

    ## load the model
    model = load_model(args.model)
    model.eval()
    model.fill()

    if use_cuda:
        model.cuda()

    patch_size = args.patch_size
    if (patch_size is not None) and (patch_size <= 0):
        raise ValueError('patch size must be positive')

    segment_images(model, args.paths, args.destdir, use_cuda, verbose, args.patch_size)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)