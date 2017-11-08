from __future__ import print_function,division

import numpy as np
import pandas as pd
import sys
import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for matching predicted particles with a test particle set')
    parser.add_argument('path', help='path to predictions')
    parser.add_argument('-r', '--radius', type=int, help='maximum match radius')
    parser.add_argument('--targets', help='path to test particles')
    parser.add_argument('-o', '--output', help='output path')
    return parser.parse_args()

def match_regions(targets, preds, radius):                                                          
    from scipy.optimize import linear_sum_assignment

    d2 = np.sum((preds[:,np.newaxis] - targets[np.newaxis])**2, 2)
    cost = d2 - radius*radius
    cost[cost > 0] = 0

    pred_index,target_index = linear_sum_assignment(cost)

    cost = cost[pred_index, target_index]
    dist = np.zeros(len(preds))
    dist[pred_index] = np.sqrt(d2[pred_index, target_index])

    pred_index = pred_index[cost < 0]
    assignment = np.zeros(len(preds), dtype=np.float32)
    assignment[pred_index] = 1


    return assignment, dist

if __name__ == '__main__':
    args = parse_args()

    try:
        predicts = pd.read_csv(args.path, sep='\t')
    except pd.errors.EmptyDataError:
        sys.exit(0)
    targets = pd.read_csv(args.targets, sep='\t')
    names = targets.image_name.unique()

    matches = []
    scores = []
    dists = []

    for name in names:
        target_coords = targets.loc[targets.image_name==name][['x_coord','y_coord']].values
        predict_coords = predicts.loc[predicts.image_name==name][['x_coord','y_coord']].values
        score = predicts.loc[predicts.image_name==name].score.values.astype(np.float32)
        match,dist = match_regions(target_coords, predict_coords, args.radius)
        
        matches.append(match)
        scores.append(score)
        dists.append(dist)

    matches = np.concatenate(matches, 0)
    scores = np.concatenate(scores, 0)
    dists = np.concatenate(dists, 0)

    df = pd.DataFrame({'score': scores, 'match': matches, 'dist': dists})

    if args.output is not None:
        df.to_csv(args.output, sep='\t', index=False)
    else:
        df.to_csv(sys.stdout, sep='\t', index=False)



