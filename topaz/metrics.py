from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import pandas as pd
from topaz.algorithms import match_coordinates



def precision_recall_curve(target, pred, N=None):
    if N is None:
        n = target.sum()
    else:
        n = N
    
    ## copy the target and prediction into matrix
    matrix = np.stack([-pred,target], 1)
    matrix.view('f4,f4').sort(order='f0', axis=0) # sort the rows

    # if we have duplicate predictions (i.e. targets with the same score)
    # compress these into one bucket with k entries and r hits
    
    # find where score[i] != score[i+1]
    mask = np.zeros(len(matrix), dtype=bool)
    mask[:-1] = (matrix[:-1,0] != matrix[1:,0])
    mask[-1] = True # last value is always true

    # how many blocks are there
    blocks = np.sum(mask)
    # count elements per block
    counts = np.zeros(blocks+1, dtype=int)
    counts[1:] = np.where(mask)[0] + 1 # last index of each block + 1
    k = np.diff(counts)
    pp = counts[1:] # number of predicted positives at each bucket

    # count true positives per block
    tp = np.cumsum(matrix[:,1])
    counts = np.zeros(blocks+1, dtype=int)
    counts[1:] = tp[mask]
    r = np.diff(counts)
    tp = counts[1:] # numer of true positives at each bucket
    
    pr = tp/pp # precision at each bucket
    pr[np.isnan(pr)] = 1
    avpr = np.sum(pr*r)/n # average-precision score = sum_buckets [precision(bucket)*recall(bucket)]
    
    re = tp/n # recall at each bucket 
    threshold = -matrix[:,0][mask] # threshold of each bucket

    return pr, re, threshold, avpr


def average_precision(target, pred, N=None):
    if N is None:
        n = target.sum()
    else:
        n = N
    
    ## copy the target and prediction into matrix
    matrix = np.stack([-pred,target], 1)
    matrix.view('f4,f4').sort(order='f0', axis=0) # sort the rows

    # if we have duplicate predictions (i.e. targets with the same score)
    # compress these into one bucket with k entries and r hits
    
    # find where score[i] != score[i+1]
    mask = np.zeros(len(matrix), dtype=bool)
    mask[:-1] = (matrix[:-1,0] != matrix[1:,0])
    mask[-1] = True # last value is always true

    # how many blocks are there
    blocks = np.sum(mask)
    # count elements per block
    counts = np.zeros(blocks+1, dtype=int)
    counts[1:] = np.where(mask)[0] + 1 # last index of each block + 1
    k = np.diff(counts)
    pp = counts[1:] # number of predicted positives at each bucket

    # count true positives per block
    tp = np.cumsum(matrix[:,1])
    counts = np.zeros(blocks+1, dtype=int)
    counts[1:] = tp[mask]
    r = np.diff(counts)
    tp = counts[1:] # numer of true positives at each bucket
    
    pr = tp/pp # precision at each bucket
    avpr = np.sum(pr*r)/n # average-precision score = sum_buckets [precision(bucket)*recall(bucket)]

    return avpr


def particle_prc(targets_path:str, predicted_path:str, match_radius:int, images:str,):
    '''Calculate precision-recall curve for particle coordinates
    '''
    targets = pd.read_csv(targets_path, sep='\t')
    predicts = pd.read_csv(predicted_path, sep='\t', comment='#')

    if images == 'union':
        image_list = set(targets.image_name.unique()) | set(predicts.image_name.unique())
    elif images == 'target':
        image_list = set(targets.image_name.unique())
    elif images == 'predicted':
        image_list = set(predicts.image_name.unique())
    else:
        raise Exception('Unknown image argument: ' + images)

    image_list = list(image_list)

    N = len(targets)

    matches = []
    scores = []

    count = 0
    mae = 0
    for name in image_list:
        target = targets.loc[targets.image_name == name]
        predict = predicts.loc[predicts.image_name == name]

        target_coords = target[['x_coord', 'y_coord']].values
        predict_coords = predict[['x_coord', 'y_coord']].values
        score = predict.score.values.astype(np.float32)

        match,dist = match_coordinates(target_coords, predict_coords, match_radius)

        this_mae = np.sum(dist[match==1])
        count += np.sum(match)
        delta = this_mae - np.sum(match)*mae
        mae += delta/count

        matches.append(match)
        scores.append(score)


    matches = np.concatenate(matches, 0)
    scores = np.concatenate(scores, 0)

    precision,recall,threshold,auprc = precision_recall_curve(matches, scores, N=N)

    print('# auprc={}, mae={}'.format(auprc,np.sqrt(mae)))     

    mask = (precision + recall) == 0
    f1 = 2*precision*recall
    f1[mask] = 0
    f1[~mask] /= (precision + recall)[~mask]

    table = pd.DataFrame({'threshold': threshold})
    table['precision'] = precision
    table['recall'] = recall
    table['f1'] = f1

    table.to_csv(sys.stdout, sep='\t', index=False)