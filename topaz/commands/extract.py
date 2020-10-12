#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys

import numpy as np
import pandas as pd
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F

from topaz.utils.data.loader import load_image
import topaz.utils.files as file_utils
from topaz.algorithms import non_maximum_suppression, match_coordinates
from topaz.metrics import average_precision
import topaz.predict
import topaz.cuda

name = 'extract'
help = 'extract particles from segmented images or segment and extract in one step with a trained classifier'

def add_arguments(parser):

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


    return parser

class NonMaximumSuppression:
    def __init__(self, radius, threshold):
        self.radius = radius
        self.threshold = threshold

    def __call__(self, args):
        name,score = args
        score,coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        return name, score, coords

def nms_iterator(scores, radius, threshold, pool=None):
    process = NonMaximumSuppression(radius, threshold)
    if pool is not None:
        for name,score,coords in pool.imap_unordered(process, scores):
            yield name,score,coords
    else:
        for name,score in scores:
            score,coords = non_maximum_suppression(score, radius, threshold=threshold)
            yield name,score,coords

def iterate_score_target_pairs(scores, targets):
    for image_name,score in scores.items():
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values
        yield score,target

class ExtractMatches:
    def __init__(self, radius, threshold, match_radius):
        self.radius = radius
        self.threshold = threshold
        self.match_radius = match_radius

    def __call__(self, args):

        score,target = args 

        score,coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        if self.match_radius is None:
            assignment, dist = match_coordinates(target, coords, self.radius)
        else:
            assignment, dist = match_coordinates(target, coords, self.match_radius)

        mse = np.sum(dist[assignment==1]**2)
        
        return assignment, score, mse, len(target)

def extract_auprc(targets, scores, radius, threshold, match_radius=None, pool=None):
    N = 0
    mse = 0
    hits = []
    preds = []

    if pool is not None:
        process = ExtractMatches(radius, threshold, match_radius)
        iterator = iterate_score_target_pairs(scores, targets)
        for assignment,score,this_mse,n in pool.imap_unordered(process, iterator):
            mse += this_mse
            hits.append(assignment)
            preds.append(score)
            N += n
    else:
        for score,target in iterate_score_target_pairs(scores, targets):
            score,coords = non_maximum_suppression(score, radius, threshold=threshold)
            if match_radius is None:
                assignment, dist = match_coordinates(target, coords, radius)
            else:
                assignment, dist = match_coordinates(target, coords, match_radius)
            mse += np.sum(dist[assignment==1]**2)
            hits.append(assignment)
            preds.append(score)
            N += len(target)

    hits = np.concatenate(hits, 0)
    preds = np.concatenate(preds, 0)
    auprc = average_precision(hits, preds, N=N)

    rmse = np.sqrt(mse/hits.sum())

    return auprc, rmse, int(hits.sum()), N

class Process:
    def __init__(self, targets, target_scores, threshold, match_radius):
        self.targets = targets
        self.target_scores = target_scores
        self.threshold = threshold
        self.match_radius = match_radius

    def __call__(self, r):
        auprc, rmse, recall, n = extract_auprc(self.targets, self.target_scores, r, self.threshold
                                              , match_radius=self.match_radius)
        return r, auprc, rmse, recall, n

def find_opt_radius(targets, target_scores, threshold, lo=0, hi=200, step=10
                   , match_radius=None, pool=None):

    auprc = np.zeros(hi+1) - 1
    process = Process(targets, target_scores, threshold, match_radius)

    if pool is not None:
        for r,au,rmse,recall,n in pool.imap_unordered(process, range(lo, hi+1, step)):
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))
    else:
        for r in range(lo, hi+1, step):
            _,au,rmse,recall,n = process(r)
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))

    r = np.argmax(auprc)
    return r, auprc[r]


def stream_images(paths):
    for path in paths:
        image = load_image(path)
        image = np.array(image, copy=False)
        yield image


def score_images(model, paths, device=-1, batch_size=1):
    if model is not None and model != 'none': # score each image with the model
        ## set the device
        use_cuda = topaz.cuda.set_device(device)
        ## load the model
        from topaz.model.factory import load_model
        model = load_model(model)
        model.eval()
        model.fill()
        if use_cuda:
            model.cuda()
        scores = topaz.predict.score_stream(model, stream_images(paths), use_cuda=use_cuda
                                           , batch_size=batch_size)
    else: # load scores directly
        scores = stream_images(paths)
    for path,score in zip(paths, scores):
        yield path, score


def stream_inputs(f):
    for line in f:
        line = line.strip()
        if len(line) > 0:
            yield line


def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    # score the images lazily with a generator
    model = args.model
    device = args.device
    paths = args.paths
    batch_size = args.batch_size

    if len(paths) == 0: # no paths specified, so we read them from stdin
        paths = stream_inputs(sys.stdin)

    stream = score_images(model, paths, device=device, batch_size=batch_size)

    # extract coordinates from scored images
    threshold = args.threshold

    radius = args.radius
    if radius is None:
        radius = -1

    num_workers = args.num_workers
    pool = None
    if num_workers < 0:
        num_workers = multiprocessing.cpu_count()
    if num_workers > 0:
        pool = multiprocessing.Pool(num_workers)

    # if no radius is set, we choose the radius based on targets provided
    lo = args.min_radius
    hi = args.max_radius
    step = args.step_radius
    match_radius = args.assignment_radius

    if radius < 0 and args.targets is not None: # set the radius to optimize AUPRC of the targets
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(args.targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        ## find radius maximizing AUPRC
        radius, auprc = find_opt_radius(targets, target_scores, threshold, lo=lo, hi=hi, step=step
                                       , match_radius=match_radius, pool=pool)


    elif args.targets is not None:
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(args.targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        # calculate AUPRC for radius
        au, rmse, recall, n = extract_auprc(targets, target_scores, radius, threshold
                                           , match_radius=match_radius, pool=pool)
        print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(radius, au, rmse, recall, n))
    elif radius < 0:
        # must have targets if radius < 0
        raise Exception('Must specify targets for choosing the extraction radius if extraction radius is not provided')


    # now, extract all particles from scored images
    if not args.only_validate:
        per_micrograph = args.per_micrograph # store one file per micrograph rather than combining all files together
        suffix = args.suffix # optional suffix to add to particle file paths
        out_format = args.format

        f = sys.stdout
        if args.output is not None and not per_micrograph:
            f = open(args.output, 'w')

        scale = args.up_scale/args.down_scale

        if not per_micrograph:
            print('image_name\tx_coord\ty_coord\tscore', file=f)
        ## extract coordinates using radius 
        for path,score,coords in nms_iterator(stream, radius, threshold, pool=pool):
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            ## scale the coordinates
            if scale != 1:
                coords = np.round(coords*scale).astype(int)

            if per_micrograph:
                table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 'score': score})
                out_path,ext = os.path.splitext(path)
                out_path = out_path + suffix + '.' + out_format
                with open(out_path, 'w') as f:
                    file_utils.write_table(f, table, format=out_format, image_ext=ext)
            else:
                for i in range(len(score)):
                    print(name + '\t' + str(coords[i,0]) + '\t' + str(coords[i,1]) + '\t' + str(score[i]), file=f)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for extracting particles from segmented images or images processed with a trained model. Uses a non maximum suppression algorithm.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

















