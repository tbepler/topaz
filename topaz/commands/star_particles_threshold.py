from __future__ import print_function,division

import sys
import os
import numpy as np
import pandas as pd

import topaz.utils.star as star

name = 'star_particles_threshold'
help = 'filter the particles in a .star file by score threshold'


def add_arguments(parser):
    parser.add_argument('file', help='path to input star file')
    parser.add_argument('-o', '--output', help='path to write particle stack file')
    parser.add_argument('-t', '--threshold', type=float, default=-np.inf, help='only take particles with scores >= this value (default: -inf)')

    return parser


def main(args):
    with open(args.file, 'r') as f:
        particles = star.parse_star(f)
    n = len(particles)
    t = args.threshold
    particles['ParticleScore'] = [float(s) for s in particles['ParticleScore']]
    particles = particles.loc[particles['ParticleScore'] >= t]
    print('# filtered', n, 'particles to', len(particles), 'with treshold of', t, file=sys.stderr)

    ## write the star file
    f = sys.stdout
    if args.output is not None:
        f = open(args.output, 'w')

    print('data_images', file=f)
    print('loop_', file=f)
    for i,name in enumerate(particles.columns):
        print('_rln' + name + ' #' + str(i+1), file=f)

    particles.to_csv(f, sep='\t', index=False, header=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for thresholding particles in a star file')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


