from __future__ import print_function

import os
import sys
import json
import numpy as np
from PIL import Image # for saving images

from topaz.transform import ScaledGaussianMixture
from topaz.utils.image import downsample, save_image
from topaz.utils.data.loader import load_image


name = 'preprocess'
help = 'downsample and normalize images in one step'


def add_arguments(parser):
    parser.add_argument('files', nargs='+')

    parser.add_argument('-s', '--scale', default=4, type=int, help='rescaling factor for image downsampling (default: 4)')
    parser.add_argument('-t', '--num-workers', default=0, type=int, help='number of processes to use for parallel image downsampling (default: 0)')

    parser.add_argument('--format', dest='format_', default='mrc', help='image format(s) to write. choices are mrc, tiff, and png. images can be written in multiple formats by specifying each in a comma separated list, e.g. mrc,png would write mrc and png format images (default: mrc)')

    parser.add_argument('--pixel-sampling', default=25, type=int, help='pixel sampling factor for model fit (default: 25)')
    parser.add_argument('--niters', default=200, type=int, help='number of iterations to run for model fit (default: 200)')
    parser.add_argument('--seed', default=1, type=int, help='random seed for model initialization (default: 1)')

    parser.add_argument('-o', '--destdir', required=True, help='output directory')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    return parser


def load_images(paths):
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        yield name, load_image(path)

class Process:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, args):
        name,image = args
        # convert PIL image to array
        image = np.array(image, copy=False)
        image = downsample(image, self.scale)
        return name, image

def downsample_images(image_iterator, scale, num_workers=0):
    if scale > 1:
        process = Process(scale)

        if num_workers > 0:
            from multiprocessing import Pool
            pool = Pool(num_workers)
            for x in pool.imap_unordered(process, image_iterator):
                yield x
        else:
            for x in image_iterator:
                yield process(x)

    else:

        for x in image_iterator:
            # need to convert PIL to numpy array
            x = np.array(x, copy=False)
            yield x

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

def main(args):
    verbose = args.verbose
    
    N = len(args.files)
    image_iterator = load_images(args.files)

    ## downsample the images
    names = []
    images = []
    for name,image in downsample_images(image_iterator, args.scale, num_workers=args.num_workers):
        names.append(name)
        images.append(image)
        if verbose:
            print('# [{} of {}] downsampled: {}'.format(len(images), N, name))


    niters = args.niters
    samples = args.pixel_sampling
    seed = args.seed
    if verbose:
        print('# fit scaled GMM, niters={}, sample={}, seed={}'.format(niters, samples, seed))
    images, metadata = sgmm_scaling(images, niters, samples, seed, verbose) 
    if verbose:
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
    formats = args.format_.split(',')
    for name,x in zip(names, images):
        base = os.path.join(destdir, name)
        for f in formats:
            save_image(x, base, f=f, verbose=verbose)

    ## save the metadata in json format
    path = os.path.join(destdir, 'metadata.json')
    with open(path, 'w') as f:
        if verbose:
            print('# saving metadata:', path)
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for performing image downsampling and normalization in one step')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


