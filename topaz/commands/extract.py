#!/usr/bin/env python
from __future__ import division, print_function

import argparse

from topaz.extract import extract_particles

name = 'extract'
help = 'extract particles from segmented images or segment and extract in one step with a trained classifier'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for extracting particles from segmented images or images processed with a trained model. Uses a non maximum suppression algorithm.')

    parser.add_argument('paths', nargs='*', help='paths to image files for processing, can also be streamed from stdin')

    parser.add_argument('-m', '--model', default='resnet16', help='path to trained subimage classifier. uses the pretrained resnet16 model by default. if micrographs have already been segmented (transformed to log-likelihood ratio maps), then this should be set to "none" (default: resnet16)')

    ## extraction parameter arguments
    parser.add_argument('-r', '--radius', type=int, help='radius of the regions to extract')
    parser.add_argument('-t', '--threshold', default=-6, type=float, help='log-likelihood score threshold at which to terminate region extraction, -6 is p>=0.0025 (default: -6)')

    
    ## coordinate scaling arguments
    parser.add_argument('-s', '--down-scale', type=float, default=1, help='DOWN-scale coordinates by this factor. output coordinates will be coord_out = (x/s)*coord. (default: 1)')
    parser.add_argument('-x', '--up-scale', type=float, default=1, help='UP-scale coordinates by this factor. output coordinates will be coord_out = (x/s)*coord. (default: 1)')

    parser.add_argument('--num-workers', type=int, default=0, help='number of processes to use for extracting in parallel, 0 uses main process, -1 uses all CPUs (default: 0)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')
    parser.add_argument('-p', '--patch-size', type=int, default=0, help='patch size for scoring micrographs in pieces (default: 0, no patching)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for scoring micrographs with model (default: 1)')


    ## radius selection arguments
    parser.add_argument('--assignment-radius', type=int, help='maximum distance between prediction and labeled target allowed for considering them a match (default: same as extraction radius)')
    parser.add_argument('--min-radius', type=int, default=5, help='minimum radius for region extraction when tuning radius parameter (default: 5)')
    parser.add_argument('--max-radius', type=int, default=100, help='maximum radius for region extraction when tuning radius parameters (default: 100)')
    parser.add_argument('--step-radius', type=int, default=5, help='grid size when searching for optimal radius parameter (default: 5)')


    parser.add_argument('--targets', help='path to file specifying particle coordinates. used to find extraction radius that maximizes the AUPRC') 
    parser.add_argument('--only-validate', action='store_true', help='flag indicating to only calculate validation metrics. does not report full prediction list')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU')

    parser.add_argument('-o', '--output', help='file path to write')
    parser.add_argument('--per-micrograph', action='store_true', help='write one particle file per micrograph at the location of the micrograph')
    parser.add_argument('--suffix', default='', help='optional suffix to add to particle file paths when using the --per-micrograph flag.')
    parser.add_argument('--format', choices=['coord', 'csv', 'star', 'json', 'box'], default='coord'
                    , help='file format of the OUTPUT files (default: coord)')
    parser.add_argument('--dims', type=int, default=2, choices=[2,3], help='image dimensionality (default: 2/micrographs), set to 3 for tomograms')
    parser.add_argument('-v','--verbose', action='store_true', help='report as each image is scored and picks are extracted')

    return parser


def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    extract_particles(args.paths, args.model, args.device, args.batch_size, args.threshold, args.radius, args.num_workers, 
                      args.targets, args.min_radius, args.max_radius, args.step_radius, args.assignment_radius, args.patch_size,
                      args.only_validate, args.output, args.per_micrograph, args.suffix, args.format, args.up_scale, args.down_scale, 
                      dims=args.dims, verbose=args.verbose)


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)