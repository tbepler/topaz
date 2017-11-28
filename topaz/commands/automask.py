from __future__ import print_function

import os
import sys
import json
import numpy as np
from PIL import Image # for saving images

from topaz.transform import ScaledGaussianMixture
from topaz.utils.data.loader import load_image

name = 'automask'
help = 'automask images using a per image scaled 2-component Gaussian mixture model'

def add_arguments(parser):
    parser.add_argument('files', nargs='+')
    parser.add_argument('-s', '--sample', default=100, type=int, help='pixel sampling factor for model fit (default: 100)')
    parser.add_argument('--niters', default=200, type=int, help='number of iterations to run for model fit (default: 200)')
    parser.add_argument('--seed', default=1, type=int, help='random seed for model initialization (default: 1)')
    parser.add_argument('-o', '--destdir', help='output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    return parser

def fit_model(X, niters, sample, seed, verbose):
    Xsample = [x.ravel()[::sample] for x in X]

    sgmm = ScaledGaussianMixture(2)
    if seed is None:
        random = np.random
    else:
        random = np.random.RandomState(seed)
    scale, probas = sgmm.fit(Xsample, niters=niters, random=random, verbose=verbose)

    return sgmm

def automask(model, image_iterator, niters=100, verbose=False):
    mask_component = np.argmin(model.means)
    for name,x in image_iterator:
        _,probas = model.transform([x], niters=niters, verbose=verbose)
        yield name, probas[0][:,:,mask_component]

def make_metadata(sgmm, niters, sample, seed):
    weights = sgmm.weights
    means = sgmm.means
    variances = sgmm.variances

    metadata = {'weights': weights.tolist(),
                'means': means.tolist(),
                'variances': variances.tolist(),
                'niters': niters,
                'sample': sample,
                'seed': seed,
               }

    return metadata

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

    niters = args.niters
    sample = args.sample
    seed = args.seed
    verbose = args.verbose

    ## fit the model
    if verbose:
        print('# fit scaled GMM, niters={}, sample={}, seed={}'.format(niters, sample, seed))
    model = fit_model(images, niters, sample, seed, verbose)

    if verbose:
        print('# weights:', model.weights)
        print('# means:', model.means)
        print('# variances:', model.variances)

    ## calculate the mask probabilities and write to disk
    destdir = args.destdir 
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    if verbose:
        print('# calculate mask probabilities')

    for name,probas in automask(model, zip(names,images), niters=niters, verbose=verbose):
        im = Image.fromarray(probas) 
        path = os.path.join(destdir, name) + '.tiff'
        if verbose:
            print('# saving:', path)
        im.save(path, 'tiff')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for automasking a list of images using a per image scaled 2-component Gaussian mixture model')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)




