from __future__ import print_function

import os
import sys
import json
import numpy as np
import multiprocessing as mp

import torch

from topaz.stats import normalize
from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample, save_image
import topaz.cuda

name = 'normalize'
help = 'normalize a set of images using the 2-component Gaussian mixture model'

def add_arguments(parser):
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

class Normalize:
    def __init__(self, dest, scale, affine, num_iters, alpha, beta
                , sample, metadata, formats, use_cuda):
        self.dest = dest
        self.scale = scale
        self.affine = affine
        self.num_iters = num_iters
        self.alpha = alpha
        self.beta = beta
        self.sample = sample
        self.metadata = metadata
        self.formats = formats
        self.use_cuda = use_cuda

    def __call__(self, path):
        # load the image
        x = np.array(load_image(path), copy=False).astype(np.float32)

        if self.scale > 1:
            x = downsample(x, self.scale)

        # normalize it
        method = 'gmm'
        if self.affine:
            method = 'affine'
        x,metadata = normalize(x, alpha=self.alpha, beta=self.beta, num_iters=self.num_iters
                              , method=method, sample=self.sample, use_cuda=self.use_cuda)

        # save the image and the metadata
        name,_ = os.path.splitext(os.path.basename(path))
        base = os.path.join(self.dest, name)
        for f in self.formats:
            save_image(x, base, f=f)

        if self.metadata:
            # save the metadata in json format
            mdpath = base + '.metadata.json'
            if not self.affine:
                metadata['mus'] = metadata['mus'].tolist()
                metadata['stds'] = metadata['stds'].tolist()
                metadata['pis'] = metadata['pis'].tolist()
                metadata['logps'] = metadata['logps'].tolist()
            with open(mdpath, 'w') as f:
                json.dump(metadata, f, indent=4)

        return name


def main(args):
    paths = args.files
    dest = args.destdir
    verbose = args.verbose

    scale = args.scale
    affine = args.affine

    num_iters = args.niters
    alpha = args.alpha
    beta = args.beta
    sample = args.sample

    num_workers = args.num_workers
    metadata = args.metadata
    formats = args.format_.split(',')

    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    # set CUDA device
    use_cuda = topaz.cuda.set_device(args.device)
    if use_cuda:
        # when using GPU, turn off multiple processes
        num_workers = 0

    if not os.path.exists(dest):
        os.makedirs(dest)

    process = Normalize(dest, scale, affine, num_iters, alpha, beta
                       , sample, metadata, formats, use_cuda)

    if num_workers > 1:
        pool = mp.Pool(num_workers)
        for name in pool.imap_unordered(process, paths):
            if verbose:
                print('# processed:', name, file=sys.stderr)
    else:
        for path in paths:
            name = process(path)
            if verbose:
                print('# processed:', name, file=sys.stderr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for normalizing a list of images using 2-component Gaussian mixture model')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


