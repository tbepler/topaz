from __future__ import division, print_function

import os
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
from topaz.utils.data.loader import LabeledImageCropDataset
from torchvision.transforms.functional import rotate as rotate2d


def enumerate_coordinates(Y):
    """Given a list of arrays containing pixel labels, enumerate the positive and negative or unlabeled 
    (all) coordinates as (index of array within list, index of coordinate within flattened array) pairs."""
    Ps = []
    for image_idx in range(len(Y)):
        bool_array = Y[image_idx].ravel().to(bool)
        #get boolean mask
        indices = torch.arange(bool_array.numel(), device=bool_array.device).reshape(1,-1)
        img_idx = torch.zeros_like(indices).fill_(image_idx)
        indices = torch.cat([img_idx, indices], dim=0)
        #collect indices in order (follows from masking), format for return
        pos = indices[:,bool_array].T
        Ps.append(pos)
    return torch.cat(Ps, axis=0)

class ShuffledSampler(torch.utils.data.sampler.Sampler):
    '''Class for repeatedly shuffling and yielding from an Nx2 tensor.
    WARNING: never returns None/StopIteration, do not attempt to convert to iterable.'''
    def __init__(self, x:torch.Tensor):
        self.x = x
        self.i = len(self.x)

    def __len__(self):
        return len(self.x)

    def __next__(self):
        if self.i >= len(self.x):
            #if consumed entire array, shuffle and reset to beginning
            rand_idx = torch.randperm(len(self.x))
            self.x = self.x[rand_idx]
            self.i = 0
        sample = self.x[self.i]
        self.i += 1
        return sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self


class USampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_images:int, shape:tuple):
        self.num_images = num_images
        self.shape = tuple(shape) # currently assume all images are the same shape
        self.size = torch.IntTensor(self.shape).prod().item() # total pixels in image

    def __len__(self):
        # number of pixels in image
        return self.size

    def __next__(self):
        # sample a tomogram uniformly
        idx = np.random.randint(self.num_images)
        # sample random point, convert to coords
        point = np.random.randint(self.size)
        return idx, point

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self


class NSampler(torch.utils.data.sampler.Sampler):
    '''Given an Nx2 tensor of positive labels, sample unlabeled posts.'''
    def __init__(self, P:torch.Tensor, num_images:int, shape:tuple):
        self.P = P
        self.num_images = num_images
        self.shape = tuple(shape) # currently assume all images are the same shape
        self.size = torch.IntTensor(self.shape).prod().item() # total pixels in image
        self.trees = self._build_trees()  
        
    def _build_trees(self):
        trees = {}
        # move to CPU for unravel index function
        P = self.P.cpu()
        for img_idx in P[:,0].unique().tolist():
            coord_subset = P[P[:,0]==img_idx]
            coords = coord_subset[:,1] #1D raveled coordinates
            coords = np.stack(np.unravel_index(coords, self.shape), axis=1) #Nx3 spatial coords
            tree = KDTree(coords) # create spatial data structure to query against
            trees[img_idx] = tree
        return trees

    def __len__(self):
        # number of N pixels in image
        size = self.size - len(self.P)
        return size

    def __next__(self):
        while True:
            # sample a tomogram uniformly
            idx = np.random.randint(self.num_images)
            tree = self.trees[idx] if idx in self.trees.keys() else None
            # sample random point, convert to coord
            point = np.random.randint(self.size)
            # if no labels on a given image, return any pixel
            if tree is None:
                return idx,point
            unraveled = np.stack(np.unravel_index(point, self.shape)).reshape(1,-1)
            # check if point is in tree
            ind, dist = tree.query(unraveled)
            if dist > 0:
                return idx, point

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self


class StratifiedCoordinateSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels:List[List[torch.Tensor]], balance:float=0.5, size:int=None, random=np.random, split='pn'):
        # labels = List[List[Tensor]]
        groups = []
        weights = np.zeros(len(labels)*2)
        proportions = np.zeros((len(labels), 2))
        i = 0
        for group in labels:
            P = enumerate_coordinates(group) # P only
            other = USampler(len(group), group[0].shape) if split=='pu' else NSampler(P, len(group), group[0].shape)
            P = ShuffledSampler(P)
            groups.append(P)
            groups.append(other)
            
            if split == 'pn':
                total_len = (len(other)+len(P))
                proportions[i//2,0] = len(other)/total_len
                proportions[i//2,1] = len(P)/total_len
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
        h = h.item() if type(h) is torch.Tensor else h
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
            # 3D arrays are ZYX so can directly rotate HW planes, adding dim has no effect
            X = rotate2d(X[None,...], angle).squeeze()
            Y = rotate2d(Y[None,...], angle).squeeze() if Y.numel() > 1 else Y
 
        ## crop down (to model's receptive field) if requested
        if self.crop is not None:
            from topaz.utils.image import crop_image
            depth,height,width = X.shape if self.dims == 3 else (None, X.shape[0], X.shape[1])
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
