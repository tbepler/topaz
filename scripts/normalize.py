from __future__ import print_function

import os
import sys
here = os.path.abspath(__file__)
impath = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, impath)

import json
import numpy as np
from PIL import Image # for saving images

from topaz.transform import ScaledGaussianMixture

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for normalizing a list of images using a per image scaled 2-component Gaussian mixture model')
    parser.add_argument('files', nargs='+')
    parser.add_argument('-s', '--sample', default=100, type=int, help='image sampling factor for model fit (default: 100)')
    parser.add_argument('--niters', default=200, type=int, help='number of iterations to run for model fit (default: 200)')
    parser.add_argument('--seed', default=1, type=int, help='random seed for model initialization (default: 1)')
    parser.add_argument('-o', '--destdir', help='output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    return parser.parse_args()


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
        im = Image.open(path)
        fp = im.fp
        im.load()
        fp.close()
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

    for name,im in zip(names, images):
        im = Image.fromarray(im) 
        path = os.path.join(destdir, name) + '.tiff'
        if args.verbose:
            print('# saving:', path)
        im.save(path, 'tiff')

    ## save the metadata in json format
    path = os.path.join(destdir, 'metadata.json')
    with open(path, 'w') as f:
        if args.verbose:
            print('# saving metadata:', path)
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)


