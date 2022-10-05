from __future__ import division, print_function

import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from topaz.denoising.datasets import PatchDataset
from torch.utils.data import DataLoader


class LinearClassifier(nn.Module):
    '''A simple convolutional layer without non-linear activation.'''

    def __init__(self, features, dims=2, patch_size:int=None, padding:int=None, batch_size:int=1):
        '''
        Args:
            features (:obj:): the sizes associated with the layer

        Attributes:
            features (:obj:)
        '''
        super(LinearClassifier, self).__init__()
        self.features = features
        conv = nn.Conv3d if dims == 3 else nn.Conv2d
        self.classifier = conv(features.latent_dim, 1, 1)
        self.patch_size = patch_size
        self.padding = padding
        self.batch_size = batch_size

    @property
    def width(self):
        return self.features.width

    @property
    def latent_dim(self):
        return self.features.latent_dim

    def fill(self, stride=1):
        return self.features.fill(stride=stride)

    def unfill(self):
        self.features.unfill()

    def forward(self, x, top_level=True):
        '''Applies the classifier to an input.

        Args:
            x (np.ndarray): the image from which features are extracted and classified

        Returns:
            z (np.ndarray): output of the classifer
        '''
        use_patches = top_level and self.patch_size and self.padding
        if use_patches:
            exceeds_patch = all(size > self.patch_size for size in x.shape)
            if exceeds_patch:
                y = self.classify_patches(x)
        else:
            z = self.features(x)
            y = self.classifier(z)
        return y
       
    def classify_patches(self, tomo, volume_num:int=1, total_volumes:int=1, verbose:bool=True):
        '''Split tomogram into smaller 3D volumes for prediction.'''
        print('classifying patches')
        tomo = tomo.squeeze() #remove any channel or batch dims, 3D
        patch_data = PatchDataset(tomo=tomo, patch_size=self.patch_size, padding=self.padding)
        batch_iterator = DataLoader(patch_data, batch_size=self.batch_size)
        count, total = 0, len(patch_data)
        
        classified = torch.zeros_like(tomo)
        print('output shape: ', classified.shape)
        for index,x in batch_iterator:
            x = self(x, top_level=False) #need normalizing?

            # stitch into total volume
            for b in range(len(x)):
                #index into batch
                i,j,k = index[b]
                xb = x[b].squeeze()

                patch = classified[i:i+self.patch_size, j:j+self.patch_size, k:k+self.patch_size]
                pz,py,px = patch.shape

                xb = xb[self.padding:self.padding+pz, self.padding:self.padding+py, self.padding:self.padding+px]
                classified[i:i+self.patch_size, j:j+self.patch_size, k:k+self.patch_size] = xb

                count += 1
                if verbose:
                    print(f'# [{volume_num}/{total_volumes}] {round(count*100/total)}', file=sys.stderr, end='\r')
        print(' '*100, file=sys.stderr, end='\r')

        return classified