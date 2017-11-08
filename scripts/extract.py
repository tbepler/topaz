#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from topaz.utils.data.loader import load_mrc, load_pil
from topaz.algorithms import non_maxima_suppression
from topaz.metrics import average_precision

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for extracting particles from segmented images or images processed with a trained model. Uses a non maxima suppression algorithm.')

    parser.add_argument('paths', nargs='+', help='paths to image files for processing')

    parser.add_argument('-m', '--model', help='path to trained subimage classifier, if no model is supplied input images must already be segmented')

    parser.add_argument('-r', '--radius', type=int, help='radius of the regions to extract')
    parser.add_argument('-t', '--threshold', default=0.5, type=float, help='score quantile giving threshold at which to terminate region extraction (default: 0.5)')


    parser.add_argument('--assignment-radius', type=int, help='maximum distance between prediction and labeled target allowed for considering them a match (default: same as extraction radius)')
    parser.add_argument('--min-radius', type=int, default=5, help='minimum radius for region extraction when tuning radius parameter (default: 5)')
    parser.add_argument('--max-radius', type=int, default=100, help='maximum radius for region extraction when tuning radius parameters (default: 100)')
    parser.add_argument('--step-radius', type=int, default=5, help='grid size when searching for optimal radius parameter (default: 5)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of processes to use for searching over radii (default: 0)')


    parser.add_argument('--targets', help='path to file specifying particle coordinates. used to find extraction radius that maximizes the AUPRC') 
    parser.add_argument('--only-validate', action='store_true', help='flag indicating to only calculate validation metrics. does not report full prediction list')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, <0 corresponds to CPU')

    parser.add_argument('-o', '--output', help='file path to write')

    return parser.parse_args()


def load_image(path):
    ext = os.path.splitext(path)[1]
    if ext == 'mrc':
        image = load_mrc(path)
    else:
        image = load_pil(path)
    return image 


def match_regions(targets, preds, radius):
    from scipy.optimize import linear_sum_assignment

    d2 = np.sum((preds[:,np.newaxis] - targets[np.newaxis])**2, 2)
    cost = d2 - radius*radius
    cost[cost > 0] = 0

    pred_index,target_index = linear_sum_assignment(cost)

    cost = cost[pred_index, target_index]
    pred_index = pred_index[cost < 0]

    assignment = np.zeros(len(preds), dtype=np.float32)
    assignment[pred_index] = 1

    return assignment, cost[cost < 0].mean()+radius*radius

def extract_auprc(targets, scores, radius, threshold, match_radius=None):
    N = 0
    rmse = 0
    hits = []
    preds = []
    for image_name,score in scores.items():
        score,coords = non_maxima_suppression(score, radius, threshold=threshold)
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values

        if match_radius is None:
            assignment, cost = match_regions(target, coords, radius)
        else:
            assignment, cost = match_regions(target, coords, match_radius)

        this_rmse = np.sqrt(cost)
        rmse += assignment.sum()*this_rmse

        hits.append(assignment)
        preds.append(score)
        N += len(target)

    hits = np.concatenate(hits, 0)
    preds = np.concatenate(preds, 0)
    auprc = average_precision(hits, preds, N=N)
    rmse = rmse/hits.sum()

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

def find_opt_radius(targets, target_scores, threshold, lo=0, hi=200, step=10, match_radius=None, num_workers=0):

    auprc = np.zeros(hi+1) - 1
    process = Process(targets, target_scores, threshold, match_radius)

    if num_workers > 0:
        from multiprocessing import Pool
        pool = Pool(num_workers)

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


if __name__ == '__main__':
    args = parse_args()

    if args.model is not None: ## load images and segment them with the model
        ## set the device
        use_cuda = False
        if args.device >= 0:
            use_cuda = torch.cuda.is_available()
            torch.cuda.set_device(args.device)

        ## load the model
        model = torch.load(args.model)
        model.eval()
        model.fill()

        if use_cuda:
            model.cuda()

        ## load the images and process with the model
        scores = {}
        for path in args.paths:
            basename = os.path.basename(path)
            image_name = os.path.splitext(basename)[0]
            image = load_image(path)

            ## process image with the model
            X = torch.from_numpy(np.array(image, copy=False)).unsqueeze(0).unsqueeze(0)
            if use_cuda:
                X = X.cuda()
            X = Variable(X, volatile=True)
            score = model(X).data[0,0].cpu().numpy()
            
            scores[image_name] = score

    else: # images are already segmented
        scores = {}
        for path in args.paths:
            basename = os.path.basename(path)
            image_name = os.path.splitext(basename)[0]
            image = load_image(path)
            scores[image_name] = np.array(image, copy=False)

    percentile = args.threshold*100
    scores_concat = np.concatenate([array.ravel() for array in scores.values()], 0)
    threshold = np.percentile(scores_concat, percentile)

    radius = args.radius
    if radius is None:
        radius = -1

    lo = args.min_radius
    hi = args.max_radius
    step = args.step_radius
    match_radius = args.assignment_radius

    if radius < 0 and args.targets is not None: # set the radius to optimize AUPRC of the targets
        targets = pd.read_csv(args.targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        ## find radius maximizing AUPRC
        num_workers = args.num_workers
        radius, auprc = find_opt_radius(targets, target_scores, threshold, lo=lo, hi=hi, step=step
                                       , match_radius=match_radius, num_workers=num_workers)
    elif args.targets is not None:
        targets = pd.read_csv(args.targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        # calculate AUPRC for radius
        au, rmse, recall, n = extract_auprc(targets, target_scores, radius, threshold, match_radius=match_radius)
        print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(radius, au, rmse, recall, n))
    elif radius < 0:
        # must have targets if radius < 0
        raise Exception('Must specify targets for choosing the extraction radius if none is provided')

    f = sys.stdout
    if args.output is not None:
        f = open(args.output, 'w')

    if not args.only_validate:
        print('image_name\tx_coord\ty_coord\tscore', file=f)
        ## extract coordinates using radius 
        for image_name,score in scores.items():
            score,coords = non_maxima_suppression(score, radius, threshold=threshold)
            for i in range(len(score)):
                print(image_name + '\t' + str(coords[i,0]) + '\t' + str(coords[i,1]) + '\t' + str(score[i]), file=f)






















