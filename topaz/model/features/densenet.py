from __future__ import print_function,division

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, nin, ng):
        super(DenseBlock, self).__init__()

        self.nin = nin
        self.nout = nin + 3*ng

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nin, ng, 3, padding=1)
        self.conv2 = nn.Conv2d(nin+ng, ng, 3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(nin+2*ng, ng, 3, dilation=4, padding=4)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = torch.cat([x, h], 1)

        h2 = self.relu(self.conv2(h))
        h = torch.cat([h, h2], 1)

        h2 = self.relu(self.conv3(h))
        h = torch.cat([h, h2], 1)

        return h


class MultiscaleDenseNet(nn.Module):
    def __init__(self, base_units=64, ng=48, num_blocks=4):
        super(MultiscaleDenseNet, self).__init__()

        self.base_units = base_units
        self.ng = ng

        u = base_units
        layers = [ nn.Conv2d(1, u, 7, padding=3)
                 , nn.ReLU(inplace=True)
                 ]

        for _ in range(num_blocks):
            dense = DenseBlock(u, ng)
            o = u + 3*ng
            proj = nn.Conv2d(o, 2*u, 1)
            layers += [dense, proj, nn.ReLU(inplace=True)]
            u = 2*u

        # last conv layer
        conv = nn.Conv2d(u, u, 7, dilation=12)
        layers += [conv, nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*layers)

        self.latent_dim = u
        self.width = (7-1)*12 + 1
        self.pad = False


    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))
        h = self.layers(x)
        return h


class DenseLayer(nn.Sequential):
    def __init__(self, nin, ng, bn_size=4, dilation=1, dropout=0, bn=True):
        super(DenseLayer, self).__init__()

        self.nin = nin
        self.ng = ng

        bias = not bn

        self.add_module('conv1', nn.Conv2d(nin, ng*bn_size, kernel_size=3
                                          , dilation=dilation, bias=bias))
        if bn:
            self.add_module('bn1', nn.BatchNorm2d(ng*bn_size))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(ng*bn_size, ng, kernel_size=3, bias=bias))
        if bn:
            self.add_module('bn2', nn.BatchNorm2d(ng))
        self.add_module('relu2', nn.ReLU(inplace=True))
        if dropout > 0:
            self.add_module('dropout', nn.Dropout(dropout))

    def forward(self, x):
        h = super(DenseLayer, self).forward(x)
        n = (x.size(2) - h.size(2))//2
        m = (x.size(3) - h.size(3))//2
        x = x[:,:,n:-n,m:-m]
        return torch.cat([x, h], 1)


class DenseNet(nn.Module):
    def __init__(self, init_units=64, ng=32, bn_size=4, fc_units=1000, num_layers=12
                , dropout=0, bn=True):
        super(DenseNet, self).__init__()

        if bn:
            self.base = nn.Sequential( nn.Conv2d(1, init_units, kernel_size=7, bias=True)
                                     , nn.BatchNorm2d(init_units)
                                     , nn.ReLU(inplace=True)
                                     , nn.MaxPool2d(kernel_size=3, stride=1)
                                     )
        else:
            self.base = nn.Sequential( nn.Conv2d(1, init_units, kernel_size=7)
                                     , nn.ReLU(inplace=True)
                                     , nn.MaxPool2d(kernel_size=3, stride=1)
                                     )

        width = 9
        layers = []

        units = init_units
        for i in range(num_layers):
            if i % 3 == 0:
                d = 4
            elif i % 3 == 1:
                d = 2
            elif i % 3 == 2:
                d = 1

            l = DenseLayer(units, ng, bn_size=bn_size, dilation=d, bn=bn, dropout=dropout)
            layers.append(l)

            width += 2*d + 2 
            units += ng

        self.layers = nn.Sequential(*layers)

        
        self.fc = nn.Conv2d(units, fc_units, kernel_size=7)
        width += 6

        self.relu = nn.ReLU(inplace=True)

        self.latent_dim = fc_units
        self.width = width
        self.pad = False


    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))
        h = self.base(x)
        h = self.layers(h)
        h = self.relu(self.fc(h))
        return h

    def fill(self, stride=1):
        self.pad = True
        return stride

    def unfill(self):
        self.pad = False





