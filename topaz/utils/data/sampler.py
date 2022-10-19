from __future__ import division, print_function

import os
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from scipy.spatial.transform import Rotation
from topaz.utils.data.loader import LabeledImageCropDataset
from torchvision.transforms.functional import rotate as rotate2d


def enumerate_pn_coordinates(Y:List[np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
    """Given a list of arrays containing pixel labels, enumerate the positive,negative coordinates as 
    (index of array within list, index of coordinate within flattened array) pairs."""
    P_size = int(sum(array.sum() for array in Y)) # number of positive coordinates
    N_size = sum(array.size for array in Y) - P_size # number of negative coordinates

    #initialize arrays of shape (P_size,) and (N_size,) respectively
    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    N = np.zeros(N_size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # N index
    for image_idx in range(len(Y)):
        flat_array = Y[image_idx].ravel()
        for coord_idx in range(len(flat_array)):
            if flat_array[coord_idx]:
                P[i] = (image_idx, coord_idx)
                i += 1
            else:
                #N only accumulates 0/False coordinate pairs
                N[j] = (image_idx, coord_idx)
                j += 1
    return P, N


def enumerate_pu_coordinates(Y:List[np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
    """Given a list of arrays containing pixel labels, enumerate the positive,unlabeled(all) coordinates as 
    (index of array within list, index of coordinate within flattened array) pairs."""
    P_size = int(sum(array.sum() for array in Y)) # number of positive coordinates
    size = sum(array.size for array in Y)

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    U = np.zeros(size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # U index
    for image_idx in range(len(Y)):
        flat_array = Y[image_idx].ravel()
        for coord_idx in range(len(flat_array)):
            if flat_array[coord_idx]:
                P[i] = (image_idx, coord_idx)
                i += 1
            # U accumulates all image,coord pairs
            U[j] = (image_idx, coord_idx)
            j += 1
    return P, U


class ShuffledSampler(torch.utils.data.sampler.Sampler):
    '''Class for repeatedly shuffling and yielding from an array.
    WARNING: never returns None/StopIteration, do not attempt to convert to iterable.'''
    def __init__(self, x:np.ndarray, random=np.random):
        self.x = x
        self.random = random
        self.i = len(self.x)

    def __len__(self):
        return len(self.x)

    def __next__(self):
        if self.i >= len(self.x):
            #if consumed entire array, shuffle and reset to beginning
            self.random.shuffle(self.x)
            self.i = 0
        sample = self.x[self.i]
        self.i += 1
        return sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self


class StratifiedCoordinateSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels:List[List[np.ndarray]], balance:float=0.5, size:int=None, random=np.random, split:Literal['pn', 'pu']='pn'):

        groups = []
        weights = np.zeros(len(labels)*2)
        proportions = np.zeros((len(labels), 2))
        i = 0
        enum_method = enumerate_pn_coordinates if split == 'pn' else enumerate_pu_coordinates
        for group in labels:
            P,other = enum_method(group) #other is set of negatives if PN method, else unlabeled 
            P,other = ShuffledSampler(P, random=random), ShuffledSampler(other, random=random)
            groups.append(P)
            groups.append(other)      
            if split == 'pn':
                proportions[i//2,0] = len(other)/(len(other)+len(P))
                proportions[i//2,1] = len(P)/(len(other)+len(P))
            elif split  == 'pu':
                proportions[i//2,0] = (len(other) - len(P))/len(other)
                proportions[i//2,1] = len(P)/len(other)

            p = balance if balance is not None else proportions[i//2,1]
            weights[i] = p/len(labels)
            weights[i+1] = (1-p)/len(labels)
            i += 2

        if size is None:
            sizes = np.array([len(g) for g in groups]) #number micrographs in 
            size = int(np.round(np.min(sizes/weights)))

        self.groups = groups
        self.weights = weights
        self.proportions = proportions
        self.size = size

        self.history = np.zeros_like(self.weights)
        self.random = random

    def __len__(self):
        return self.size

    def __next__(self) -> int:
        n = self.history.sum()
        weights = self.weights
        if n > 0:
            weights = weights - self.history/n
            weights[weights < 0] = 0
            n = weights.sum()
            if n > 0:
                weights /= n
            else:
                weights = np.ones_like(weights)/len(weights)

        i = self.random.choice(len(weights), p=weights)
        self.history[i] += 1
        if np.all(self.history/self.history.sum() == self.weights):
            self.history[:] = 0

        g = self.groups[i]
        sample = next(g)

        i = i//2
        j,c = sample

        # code as integer; unfortunate hack required because pytorch converts index to integer...
        # allows storage of 3 integers in one int object
        h = i*2**56 + j*2**32 + c
        return h

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        for _ in range(self.size):
            yield next(self)


class RandomImageTransforms:
    """Container and iterator for image/label crops. Applies selected augmentations. Returns Torch Tensors."""
    def __init__(self, data:LabeledImageCropDataset, rotate:bool=True, flip:bool=True, crop:int=None,
                 resample=Image.BILINEAR, dims=2):
        self.data = data
        self.rotate = rotate
        self.flip = flip
        self.crop = crop
        self.resample = resample
        self.seeded = False
        self.dims = dims

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i:int):
        if not self.seeded:
            seed = (os.getpid()*31) % (2**32)
            self.random = np.random.RandomState(seed)
            self.seeded = True

        X, Y = self.data[i] # torch Tensors
        #below generally not used for training; Y should be 1D Tensor with 1 item
        if type(Y) is Image.Image: 
            Y = torch.from_numpy(np.array(Y, copy=False)).float()

        ## random rotation
        if self.rotate:
            angle = self.random.uniform(0, 360)
            if self.dims == 2:
                X = rotate2d(X, angle)
                Y = rotate2d(Y, angle) if Y.numel() > 1 else Y
            elif self.dims == 3:
                #array is ZYX so can directly rotate HW planes 
                X = rotate2d(X, angle)
                Y = rotate2d(Y, angle) if Y.numel() > 1 else Y
 
                #below spherical sampling mixes in missing wedge so don't use
                # rot_mat = torch.Tensor(Rotation.random().as_matrix()) # 3x3
                # rot_mat = torch.cat((rot_mat, torch.zeros(3,1)), axis=1) #append zero translation vector
                # rot_mat = rot_mat[None,...].type(torch.FloatTensor) #add singleton batch dimension
                # #grid is shape N x C x D x H x W
                # grid_shape = (1,1) + X.shape
                # grid = F.affine_grid(rot_mat, grid_shape, align_corners=False).type(torch.FloatTensor) 
                # X = F.grid_sample(X[None,None,...], grid, align_corners=False).squeeze()
                # Y = F.grid_sample(Y[None,None,...], grid, align_corners=False).squeeze() if Y.numel() > 1 else Y

        ## crop down (to model's receptive field) if requested
        if self.crop is not None:
            from topaz.utils.image import crop_image
            height,width,depth = X.shape if self.dims == 3 else (X.shape[0], X.shape[1], None)
            xmi = (width-self.crop)//2
            xma = xmi + self.crop
            ymi = (height-self.crop)//2
            yma = ymi + self.crop
            zmi, zma = None, None
            if depth:
                zmi = (depth - self.crop)//2
                zma = zmi + self.crop
            X = crop_image(X, xmi, xma, ymi, yma, zmi, zma)
            Y = crop_image(Y, xmi, ymi, xma, yma, zmi, zma)  if Y.numel() > 1 else Y
            
        ## random mirror of the image
        if self.flip:
            if self.random.uniform() > 0.5:
                #flip first dimension (Y axis)
                X = X.flipud()
                Y = Y.flipud() if len(Y.shape) >= 2 else Y 
            if self.random.uniform() > 0.5:
                #flip second dimension (X axis)
                X = X.fliplr()
                Y = Y.fliplr() if len(Y.shape) >= 2 else Y 
            if self.dims == 3 and self.random.uniform() > 0.5:
                #flip third dimension (Z axis) if 3D
                X = X.flip(2)
                Y = Y.flip(2) if len(Y.shape) >= 3 else Y

        return X, Y
