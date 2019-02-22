from __future__ import print_function

import os
import sys
import json
import numpy as np
from PIL import Image # for saving images

from topaz.transform import ScaledGaussianMixture
from topaz.utils.data.loader import load_image
from topaz.utils.image import save_image

name = 'normalize'
help = 'normalize a set of images using a per image scaled 2-component Gaussian mixture model'

def add_arguments(parser):
    parser.add_argument('files', nargs='+')
    parser.add_argument('-s', '--sample', default=25, type=int, help='pixel sampling factor for model fit (default: 25)')
    parser.add_argument('--niters', default=200, type=int, help='number of iterations to run for model fit (default: 200)')
    parser.add_argument('--seed', default=1, type=int, help='random seed for model initialization (default: 1)')

    parser.add_argument('-o', '--destdir', help='output directory')

    parser.add_argument('--format', dest='format_', default='mrc', help='image format(s) to write. choices are mrc, tiff, and png. images can be written in multiple formats by specifying each in a comma separated list, e.g. mrc,png would write mrc and png format images (default: mrc)')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    return parser


def sgmm_scaling(X, niters, sample, seed, verbose):
    Xsample = [x.ravel()[::sample] for x in X]

    sgmm = ScaledGaussianMixture(2)
    if seed is None:
        random = np.random
    else:
        random = np.random.RandomState(seed)
    scale, probas = sgmm.fit(Xsample, niters=niters, random=random, verbose=verbose)
    weights = sgmm.weights
    means = sgmm.means
    variances = sgmm.variances

    i = np.argmax(weights)
    mu = means[i]
    std = np.sqrt(variances[i])

    scaled = [(X[i]/scale[i] - mu)/std for i in range(len(X))]
    metadata = {'weights': weights.tolist(),
                'means': means.tolist(),
                'variances': variances.tolist(),
                'scales': scale.tolist(),
                'niters': niters,
                'sample': sample,
                'seed': seed,
               }
    return scaled, metadata

def load_images(paths):
    images = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        im = load_image(path)
        images[name] = np.array(im, copy=False)
    return images

def main(args):
    images = load_images(args.files)
    names = list(images.keys())
    images = list(images.values())

    if args.verbose:
        print('# fit scaled GMM, niters={}, sample={}, seed={}'.format(args.niters, args.sample, args.seed))
    images, metadata = sgmm_scaling(images, args.niters, args.sample, args.seed, args.verbose) 
    if args.verbose:
        print('# weights:', metadata['weights'])
        print('# means:', metadata['means'])
        print('# variances:', metadata['variances'])

    scales = {}
    for name,scale in zip(names, metadata['scales']):
        scales[name] = scale
    metadata['scales'] = scales

    destdir = args.destdir 
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    ## what image formats are we writing
    verbose = args.verbose
    formats = args.format_.split(',')
    for name,x in zip(names, images):
        base = os.path.join(destdir, name)
        for f in formats:
            save_image(x, base, f=f, verbose=verbose)

    ## save the metadata in json format
    path = os.path.join(destdir, 'metadata.json')
    with open(path, 'w') as f:
        if args.verbose:
            print('# saving metadata:', path)
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for normalizing a list of images using a per image scaled 2-component Gaussian mixture model')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


