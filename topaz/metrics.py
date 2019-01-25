from __future__ import division, print_function

import numpy as np

def precision_recall_curve(target, pred, N=None):
    if N is None:
        n = target.sum()
    else:
        n = N
    
    ## copy the target and prediction into matrix
    matrix = np.stack([-pred,target], 1)
    matrix.view('f4,f4').sort(order='f0', axis=0) # sort the rows
    
    precision = np.zeros(target.shape[0], dtype=matrix.dtype)
    recall = np.zeros(target.shape[0], dtype=matrix.dtype)
    threshold = np.zeros(target.shape[0], dtype=matrix.dtype)

    auprc = count = pr = relk = rel = 0
    j = 0
    
    for i in range(matrix.shape[0]):
        count += 1
        rel += matrix[i,1]
        relk += matrix[i,1] # target
        delta = matrix[i,1] - pr
        pr += delta/count
        if i >= matrix.shape[0] - 1 or matrix[i,0] != matrix[i+1,0]:
            precision[j] = pr
            recall[j] = rel/n
            threshold[j] = -matrix[i,0]
            j += 1
            auprc += pr*relk
            relk = 0
    auprc /= n
    
    return precision[:j], recall[:j], threshold[:j], auprc

def average_precision(target, pred, N=None):
    if N is None:
        n = target.sum()
    else:
        n = N
    
    ## copy the target and prediction into matrix
    matrix = np.stack([-pred,target], 1)
    matrix.view('f4,f4').sort(order='f0', axis=0) # sort the rows
    
    auprc = count = pr = relk = 0
    
    for i in range(matrix.shape[0]):
        count += 1
        relk += matrix[i,1] # target
        delta = matrix[i,1] - pr
        pr += delta/count
        if i >= matrix.shape[0] - 1 or matrix[i,0] != matrix[i+1,0]:
            auprc += pr*relk
            relk = 0
    auprc /= n
    
    return auprc

