from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from topaz.model.utils import insize_from_outsize

class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()

        if 'pooling' in kwargs:
            pooling = kwargs['pooling']
            if pooling == 'max':
                kwargs['pooling'] = MaxPool

        modules = self.make_modules(**kwargs)
        self.features = nn.Sequential(*modules)

        self.width = insize_from_outsize(modules, 1)
        self.pad = False

    ## make property for num_features !!

    def fill(self, stride=1):
        for mod in self.features.children():
            if hasattr(mod, 'fill'):
                stride *= mod.fill(stride)
        self.pad = True
        return stride

    def unfill(self):
        for mod in self.features.children():
            if hasattr(mod, 'unfill'):
                mod.unfill()
        self.pad = False

    def set_padding(self, pad):
        self.pad = pad
        #for mod in self.features:
        #    if hasattr(mod, 'set_padding'):
        #        mod.set_padding(pad)

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1) # add channels dim
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))
        z = self.features(x)
        return z
        #return self.classifier(z)[:,0] # remove channels dim


class ResNet6(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0, activation=nn.ReLU, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]

        modules = [
                BasicConv2d(1, units[0], 5, bn=bn, activation=activation),
                ]
        modules.append(MaxPool(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[1], dilation=4, bn=bn, activation=activation),
                ]
        modules.append(MaxPool(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 5, bn=bn, activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        self.latent_dim = units[-1]

        return modules


class ResNet8(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0
                    , activation=nn.ReLU, pooling=None, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]
        self.stride = 1
        if pooling is None:
            self.stride = 2
        stride = self.stride

        modules = [
                BasicConv2d(1, units[0], 7, stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=2, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=2
                      , stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 5, bn=bn, activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        self.latent_dim = units[-1]

        return modules


class ResNet16(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0
                    , activation=nn.ReLU, pooling=None, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]
        self.stride = 1
        if pooling is None:
            self.stride = 2
        stride = self.stride

        modules = [
                BasicConv2d(1, units[0], 7, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[0]
                      , stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
                ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
                ResidA(units[0], units[0], units[0], bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1]
                      , stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], bn=bn, activation=activation),
                ResidA(units[1], units[1], units[1], bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 5, bn=bn, activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        self.latent_dim = units[-1]

        return modules
                                      

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.og_stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation*(self.kernel_size//2) # this is bugged in pytorch...
            #p = self.kernel_size//2
            self.pool.padding = (p, p)
            self.padding = p
        else:
            self.pool.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.pool.dilation = stride
        self.pool.padding = self.pool.padding*stride
        self.pool.stride = 1
        self.dilation = stride
        self.stride = 1
        return self.og_stride

    def unfill(self):
        self.pool.dilation = 1
        self.pool.padding = self.pool.padding//self.dilation
        self.pool.stride = self.og_stride
        self.dilation = 1
        self.stride = self.og_stride

    def forward(self, x):
        return self.pool(x)

class BasicConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size, dilation=1, stride=1
                , bn=False, activation=nn.ReLU):
        super(BasicConv2d, self).__init__()

        bias = not bn
        self.conv = nn.Conv2d(nin, nout, kernel_size, dilation=dilation
                             , stride=stride, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(nout)
        self.act = activation(inplace=True)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.og_dilation = dilation
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation*(self.kernel_size//2)
            self.conv.padding = (p, p)
            self.padding = p
        else:
            self.conv.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.conv.dilation = (self.og_dilation*stride, self.og_dilation*stride)
        self.conv.stride = (1,1)
        self.conv.padding = (self.conv.padding[0]*stride, self.conv.padding[1]*stride)
        self.dilation *= stride
        return self.stride

    def unfill(self):
        stride = self.dilation//self.og_dilation
        self.conv.dilation = (self.og_dilation, self.og_dilation)
        self.conv.stride = (self.stride,self.stride)
        self.conv.padding = (self.conv.padding[0]//stride, self.conv.padding[1]//stride)
        self.dilation = self.og_dilation

    def forward(self, x):
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        return self.act(y)

class ResidA(nn.Module):
    def __init__(self, nin, nhidden, nout, dilation=1, stride=1
                , activation=nn.ReLU, bn=False):
        super(ResidA, self).__init__()

        self.bn = bn
        bias = not bn

        if nin != nout:
            self.proj = nn.Conv2d(nin, nout, 1, stride=stride, bias=False)
        
        self.conv0 = nn.Conv2d(nin, nhidden, 3, bias=bias)
        if self.bn:
            self.bn0 = nn.BatchNorm2d(nhidden)
        self.act0 = activation(inplace=True)

        self.conv1 = nn.Conv2d(nhidden, nout, 3, dilation=dilation, stride=stride
                              , bias=bias)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(nout)
        self.act1 = activation(inplace=True)

        self.kernel_size = 2*dilation + 3
        self.stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            self.conv0.padding = (1,1)
            self.conv1.padding = self.conv1.dilation
            self.padding = self.kernel_size//2
        else:
            self.conv0.padding = (0,0)
            self.conv1.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.conv0.dilation = (stride, stride)
        self.conv0.padding = (self.conv0.padding[0]*stride, self.conv0.padding[1]*stride)
        self.conv1.dilation = (self.conv1.dilation[0]*stride, self.conv1.dilation[1]*stride)
        self.conv1.stride = (1,1)
        self.conv1.padding = (self.conv1.padding[0]*stride, self.conv1.padding[1]*stride)
        if hasattr(self, 'proj'):
            self.proj.stride = (1,1)
        self.dilation = self.dilation*stride
        return self.stride

    def unfill(self):
        self.conv0.dilation = (1,1)
        self.conv0.padding = (self.conv0.padding[0]//self.dilation, self.conv0.padding[1]//self.dilation)
        self.conv1.dilation = (self.conv1.dilation[0]//self.dilation, self.conv1.dilation[1]//self.dilation)
        self.conv1.stride = (self.stride,self.stride)
        self.conv1.padding = (self.conv1.padding[0]//self.dilation, self.conv1.padding[1]//self.dilation)
        if hasattr(self, 'proj'):
            self.proj.stride = (self.stride,self.stride)
        self.dilation = 1

    def forward(self, x):

        h = self.conv0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)

        y = self.conv1(h)

        #d2 = x.size(2) - y.size(2)
        #d3 = x.size(3) - y.size(3)
        #if d2 > 0 or d3 > 0:
        #    lb2 = d2//2
        #    ub2 = d2 - lb2
        #    lb3 = d3//2
        #    ub3 = d3 - lb3
        #    x = x[:,:,lb2:-ub2,lb3:-ub3]

        edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        x = x[:,:,edge:-edge,edge:-edge]

        if hasattr(self, 'proj'):
            x = self.proj(x)
        elif self.conv1.stride[0] > 1:
            x = x[:,:,::self.stride,::self.stride]
        

        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)

        return y



