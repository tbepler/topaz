#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd

from topaz.metrics import precision_recall_curve

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for calculating the precision-recall curve for a set of predicted particle coordinates and a set of target coordinates.')

    parser.add_argument('--predicted', help='path to file containing predicted particle coordinates')
    parser.add_argument('--targets', help='path to file specifying target particle coordinates') 

    parser.add_argument('--assignment-radius', type=int, help='maximum distance between prediction and labeled target allowed for considering them a match')

    return parser.parse_args()


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
    hits = []
    preds = []
    for image_name,score in scores.items():
        score,coords = non_maxima_suppression(score, radius, threshold=threshold)
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values

        if match_radius is None:
            assignment, cost = match_regions(target, coords, radius)
        else:
            assignment, cost = match_regions(target, coords, match_radius)

        hits.append(assignment)
        preds.append(score)
        N += len(target)

    hits = np.concatenate(hits, 0)
    preds = np.concatenate(preds, 0)
    auprc = average_precision(hits, preds, N=N)

    return auprc, np.sqrt(cost), int(hits.sum()), N


if __name__ == '__main__':
    args = parse_args()

    match_radius = args.assignment_radius
    targets = pd.read_csv(args.targets, sep='\t')
    predicts = pd.read_csv(args.predicted, sep='\t', comment='#')

    image_list = set(targets.image_name.unique()) | set(predicts.image_name.unique())
    image_list = list(image_list)

    N = len(targets)

    matches = []
    scores = []

    for name in image_list:
        target = targets.loc[targets.image_name == name]
        predict = predicts.loc[predicts.image_name == name]

        target_coords = target[['x_coord', 'y_coord']].values
        predict_coords = predict[['x_coord', 'y_coord']].values
        score = predict.score.values.astype(np.float32)

        match,_ = match_regions(target_coords, predict_coords, match_radius)

        matches.append(match)
        scores.append(score)


    matches = np.concatenate(matches, 0)
    scores = np.concatenate(scores, 0)

    precision,recall,threshold,auprc = precision_recall_curve(matches, scores, N=N)

    print('# auprc={}'.format(auprc))     

    f1 = 2*precision*recall/(precision + recall)

    table = pd.DataFrame({'threshold': threshold})
    table['precision'] = precision
    table['recall'] = recall
    table['f1'] = f1

    table.to_csv(sys.stdout, sep='\t', index=False)


















