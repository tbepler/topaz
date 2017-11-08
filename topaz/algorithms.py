from __future__ import print_function, division

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def point_masked(x, mask, i, j, output):
    width = mask.size(2)
    M,N = x.size(1), x.size(2)
    imi = i - width//2
    ima = i + width//2 + 1
    jmi = j - width//2
    jma = j + width//2 + 1

    ci = (max(0,-imi), min(width,width-(ima-M)))
    cj = (max(0,-jmi), min(width,width-(jma-N)))
    xi = (max(0,imi), min(M,ima))
    xj = (max(0,jmi), min(N,jma))

    xc = x[:,xi[0]:xi[1],xj[0]:xj[1]] 
    mask = mask[0,:,ci[0]:ci[1],cj[0]:cj[1]]

    output[:,xi[0]:xi[1],xj[0]:xj[1]][mask > 0] = xc[mask > 0]

    return output

def point_value(value, mask, i, j, output):
    width = mask.size(2)
    M,N = output.size(1), output.size(2)
    imi = i - width//2
    ima = i + width//2 + 1
    jmi = j - width//2
    jma = j + width//2 + 1

    ci = (max(0,-imi), min(width,width-(ima-M)))
    cj = (max(0,-jmi), min(width,width-(jma-N)))
    xi = (max(0,imi), min(M,ima))
    xj = (max(0,jmi), min(N,jma))

    mask = mask[0,:,ci[0]:ci[1],cj[0]:cj[1]]
    output[:,xi[0]:xi[1],xj[0]:xj[1]][mask > 0] = value

    return output

def greedy_set_cover(x, d, scale=1.0, threshold=0, masked=False, value=None):
    ## form the shape filter
    r = scale*d/2
    width = int(np.ceil(d*scale))

    span = width//2
    I = np.arange(-span, span+1)[:,np.newaxis]
    J = np.arange(-span, span+1)[np.newaxis]
    dist = np.sqrt(I**2 + J**2)
    #mask = (dist <= r).astype(np.float32)
    ## to torch
    mask = x.new(*dist.shape)
    mask[:] = torch.from_numpy((dist <= r).view(np.uint8)).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask /= mask.sum()
    #Z = mask.sum()

    ndim = len(x.size())
    if ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif ndim == 3:
        x = x.unsqueeze(1)
       
    M,N = x.size(2), x.size(3)
    if not masked:
        sets = F.conv2d(Variable(x, volatile=True), Variable(mask, volatile=True), padding=span)
        sets = sets.data
    else:
        sets = x.clone()

    B = sets.size(0)
    L = sets.view(B,-1).size(1)
    y,k = sets.view(B,-1).max(1)
    #y /= Z
    y = y[:,0]
    k = k[:,0]
    y = y.cpu().numpy()
    k = k.cpu().numpy()

    scores = np.zeros((B,L), dtype=np.float32)
    coords = np.zeros((B,L,2), dtype=np.int32)

    ## iterate extracting sets in order
    i = 0
    xmasked = x.new(*x.size())
    while np.any(y > threshold) and i < L:
        scores[:,i] = y
        coords[:,i,0] = k % N  # X coord
        coords[:,i,1] = k // N # Y coord

        ## subtract these sets
        xmasked[:] = 0
        for j in range(B):
            ii = coords[j,i,0] ## column coordinate
            jj = coords[j,i,1] ## row coordinate
            if not masked:
                xj = point_masked(x[j], mask, jj, ii, xmasked[j])
            else:
                v = -np.inf # float(y[j])
                if value is not None:
                    v = value
                _ = point_value(v, mask, jj, ii, sets[j])
                #xmasked[j,:,jj,ii] = v
                #xj = point_value(float(y[j]), mask, jj, ii, xmasked[j])
        if not masked:
            delta = F.conv2d(Variable(xmasked, volatile=True), Variable(mask, volatile=True), padding=width//2).data
            sets -= delta

        ## extract next best regions
        y,k = sets.view(B,-1).max(1)
        #y /= Z
        y = y[:,0]
        k = k[:,0]
        y = y.cpu().numpy()
        k = k.cpu().numpy()
        i += 1

    return scores[:,:i], coords[:,:i]


def non_maxima_suppression(x, r, threshold=-np.inf):
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


def non_maxima_suppression_3d(x, d, scale=1.0, threshold=-np.inf):
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







