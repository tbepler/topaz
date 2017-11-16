from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from topaz.model.utils import insize_from_outsize

class BasicConv(nn.Module):
    def __init__(self, layers, units, unit_scaling=1, dropout=0, bn=True
                , pooling=None, activation=nn.PReLU):
        super(BasicConv, self).__init__()

        use_bias = (not bn)
        stride = 2
        if pooling == 'max':
            pooling = nn.MaxPool2d
            stride = 1
        elif pooling == 'avg':
            pooling = nn.AvgPool2d
            stride = 1

        sizes = layers
        layers = []
        strides = []

        nin = 1
        for size in sizes[:-1]:
            layers += [ nn.Conv2d(nin, units, size, stride=stride, bias=use_bias) ]
            strides += [stride]
            if bn:
                layers += [ nn.BatchNorm2d(units) ]
                strides += [1]
            layers += [ activation() ]
            strides += [1]
            if pooling is not None:
                layers += [ pooling(3, stride=2, padding=1) ]
                strides += [2]
            if dropout > 0:
                layers += [ nn.Dropout(p=dropout) ]
            nin = units
            units *= unit_scaling

        size = sizes[-1]
        layers += [ nn.Conv2d(nin, units, size, bias=use_bias) ]
        strides += [1]
        if bn:
            layers += [ nn.BatchNorm2d(units) ]
            strides += [1]
        layers += [ activation() ]
        if dropout > 0:
            layers += [ nn.Dropout(p=dropout) ]
        strides += [1]

        self.strides = strides
        self.width = insize_from_outsize(layers, 1)
        self.filled = False

        self.features = nn.Sequential(*layers)

        self.latent_dim = units

    def fill(self, stride=1):
        for mod,mod_stride in zip(self.features.children(), self.strides):
            if hasattr(mod, 'dilation'):
                mod.dilation = (stride, stride)
            if hasattr(mod, 'stride'):
                mod.stride = (1,1)
            # this is bugged in pytorch, padding size cannot be bigger than kernel despite dilation
            #if hasattr(mod, 'padding'):
            #    if type(mod.padding) is tuple and mod.padding[0] > 0:
            #        mod.padding = (stride,stride)
            #    elif type(mod.padding) is int and mod.padding > 0:
            #        mod.padding = stride
            stride *= mod_stride
        layers = list(self.features.modules())
        self.filled = True
        return stride


    def unfill(self):
        for mod,mod_stride in zip(self.features.children(), self.strides):
            if hasattr(mod, 'dilation'):
                mod.dilation = (1,1)
            if hasattr(mod, 'stride'):
                mod.stride = (mod_stride,mod_stride)
            #if hasattr(mod, 'padding'):
            #    if type(mod.padding) is tuple and mod.padding[0] > 0:
            #        mod.padding = (1,1)
            #    elif type(mod.padding) is int and mod.padding > 0:
            #        mod.padding = 1
        self.filled = False


    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1) # add channels dim
        if self.filled: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))
        z = self.features(x)
        return z


class Conv127(BasicConv):
    def __init__(self, units, **kwargs):
        super(Conv127, self).__init__(units, [7, 5, 5, 5, 5], **kwargs)

class Conv63(BasicConv):
    def __init__(self, units, **kwargs):
        super(Conv63, self).__init__(units, [7, 5, 5, 5], **kwargs)




