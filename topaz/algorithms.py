from __future__ import absolute_import, print_function, division

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_coordinates(targets, preds, radius):
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


def non_maximum_suppression(x, r, threshold=-np.inf):
    ## enumerate coordinate deltas within d
    width = r
    ii,jj = np.meshgrid(np.arange(-width,width+1), np.arange(-width,width+1))
    mask = (ii**2 + jj**2) <= r*r
    ii = ii[mask]
    jj = jj[mask]
    major_axis = x.shape[1]
    coord_deltas = ii*major_axis + jj

    A = x.ravel()
    I = np.argsort(A, axis=None)[::-1] # reverse to sort in descending order

    S = set()
    #S = np.zeros(len(A), dtype=np.int8) # the set of suppressed coordinates
    #S = np.zeros(x.shape, dtype=np.int8)

    scores = np.zeros(len(A), dtype=np.float32)
    coords = np.zeros((len(A),2), dtype=np.int32)

    j = 0
    for i in I:
        if A[i] <= threshold:
            break
        if i not in S:
            ## coordinate i is next center
            xx = i % major_axis
            yy = i // major_axis
            scores[j] = A[i]
            coords[j,0] = xx
            coords[j,1] = yy
            j += 1
            ## add coordinates within d of i to the suppressed set
            y_coords = np.clip(yy + ii, 0, x.shape[0])
            x_coords = np.clip(xx + jj, 0, x.shape[1]) 
            for y_coord,x_coord in zip(y_coords, x_coords):
                S.add(y_coord*major_axis + x_coord)
    
    return scores[:j], coords[:j]


def non_maximum_suppression_3d(x, d, scale=1.0, threshold=-np.inf):
    ## enumerate coordinate deltas within d
    r = scale*d/2
    width = int(np.ceil(r))
    A = np.arange(-width,width+1)
    ii,jj,kk = np.meshgrid(A, A, A)
    mask = (ii**2 + jj**2 + kk**2) <= r*r
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    zstride = x.shape[1]*x.shape[2]
    ystride = x.shape[2]
    coord_deltas = ii*zstride + jj*ystride + kk
    
    A = x.ravel()
    I = np.argsort(A, axis=None)[::-1] # reverse to sort in descending order
    S = set() # the set of suppressed coordinates

    scores = np.zeros(len(A), dtype=np.float32)
    coords = np.zeros((len(A),3), dtype=np.int32)

    j = 0
    for i in I:
        if A[i] <= threshold:
            break
        if i not in S:
            ## coordinate i is next center
            zz,yy,xx = np.unravel_index(i, x.shape)
            scores[j] = A[i]
            coords[j,0] = xx
            coords[j,1] = yy
            coords[j,2] = zz
            j += 1
            ## add coordinates within d of i to the suppressed set
            for delta in coord_deltas:
                S.add(i + delta)
    
    return scores[:j], coords[:j]







