from __future__ import absolute_import, division, print_function

import numpy as np

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

