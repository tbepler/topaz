from __future__ import print_function, division

import os

import numpy as np
from PIL import Image

import torch
import torch.utils.data

def enumerate_pn_coordinates(Y):
    """
    Given a list of 2d arrays containing labels, enumerate the positive and negative coordinates as (image,coordinate) pairs.
    """

    P_size = int(sum(y.sum() for y in Y)) # number of positive coordinates
    N_size = sum(y.size for y in Y) - P_size # number of negative coordinates

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    N = np.zeros(N_size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # N index
    for image in range(len(Y)):
        y = Y[image].ravel()
        for coord in range(len(y)):
            if y[coord]:
                P[i] = (image, coord)
                i += 1
            else:
                N[j] = (image, coord)
                j += 1

    return P, N

def enumerate_pu_coordinates(Y):
    """
    Given a list of 2d arrays containing labels, enumerate the positive and unlabeled(all) coordinates as (image,coordinate) pairs.
    """

    P_size = int(sum(y.sum() for y in Y)) # number of positive coordinates
    size = sum(y.size for y in Y)

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    U = np.zeros(size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # U index
    for image in range(len(Y)):
        y = Y[image].ravel()
        for coord in range(len(y)):
            if y[coord]:
                P[i] = (image, coord)
                i += 1
            U[j] = (image, coord)
            j += 1

    return P, U

class ShuffledSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, x, random=np.random):
        self.x = x
        self.random = random
        self.i = len(self.x)

    def __len__(self):
        return len(self.x)

    def __next__(self):
        if self.i >= len(self.x):
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
    def __init__(self, labels, balance=0.5, size=None, random=np.random, split='pn'):

        groups = []
        weights = np.zeros(len(labels)*2)
        proportions = np.zeros((len(labels), 2))
        i = 0
        for group in labels:
            if split == 'pn':
                P,N = enumerate_pn_coordinates(group)
                P = ShuffledSampler(P, random=random)
                N = ShuffledSampler(N, random=random)
                groups.append(P)
                groups.append(N)

                proportions[i//2,0] = len(N)/(len(N)+len(P))
                proportions[i//2,1] = len(P)/(len(N)+len(P))
            elif split  == 'pu':
                P,U = enumerate_pu_coordinates(group)
                P = ShuffledSampler(P, random=random)
                U = ShuffledSampler(U, random=random)
                groups.append(P)
                groups.append(U)

                proportions[i//2,0] = (len(U) - len(P))/len(U)
                proportions[i//2,1] = len(P)/len(U)

            p = balance
            if balance is None:
                p = proportions[i//2,1]
            weights[i] = p/len(labels)
            weights[i+1] = (1-p)/len(labels)
            i += 2

        if size is None:
            sizes = np.array([len(g) for g in groups])
            size = int(np.round(np.min(sizes/weights)))

        self.groups = groups
        self.weights = weights
        self.proportions = proportions
        self.size = size

        self.history = np.zeros_like(self.weights)
        self.random = random

    def __len__(self):
        return self.size

    def __next__(self):
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

        # code as integer
        # unfortunate hack required because pytorch converts index to integer...
        h = i*2**56 + j*2**32 + c
        return h
        #return i//2, sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        for _ in range(self.size):
            yield next(self)


class RandomImageTransforms:
    def __init__(self, data, rotate=True, flip=True, crop=None, resample=Image.BILINEAR, to_tensor=False):
        self.data = data
        self.rotate = rotate
        self.flip = flip
        self.crop = crop
        self.resample = resample
        self.to_tensor = to_tensor
        self.seeded = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if not self.seeded:
            seed = (os.getpid()*31) % (2**32)
            self.random = np.random.RandomState(seed)
            self.seeded = True

        X, Y = self.data[i]

        ## random rotation
        if self.rotate:
            angle = self.random.uniform(0, 360)
            X = X.rotate(angle, resample=self.resample)
            if type(Y) is Image.Image:
                Y = Y.rotate(angle, resmaple=Image.NEAREST)

        ## crop down if requested
        if self.crop is not None:
            width,height = X.size
            xmi = (width-self.crop)//2
            xma = xmi+self.crop
            ymi = (height-self.crop)//2
            yma = ymi+self.crop
            
            X = X.crop((xmi,ymi,xma,yma))
            if type(Y) is Image.Image:
                Y = Y.crop((xmi,ymi,xma,yma))

        ## random mirror of the image
        if self.flip:
            if self.random.uniform() > 0.5:
                X = X.transpose(Image.FLIP_LEFT_RIGHT)
                if type(Y) is Image.Image:
                    Y = Y.transpose(Image.FLIP_LEFT_RIGHT)
            if self.random.uniform() > 0.5:
                X = X.transpose(Image.FLIP_TOP_BOTTOM)
                if type(Y) is Image.Image:
                    Y = Y.transpose(Image.FLIP_TOP_BOTTOM)

        if self.to_tensor:
            X = torch.from_numpy(np.array(X, copy=False))
            if type(Y) is Image.Image:
                Y = torch.from_numpy(np.array(Y, copy=False)).float()

        return X, Y





        


