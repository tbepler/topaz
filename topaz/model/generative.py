from __future__ import print_function,division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
self.layers = nn.Sequential(
    # input is Z, going into a convolution
    nn.ConvTranspose2d(     nin, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8),
    activation(),
    # state size. (ngf*8) x 4 x 4
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4),
    activation(),
    # state size. (ngf*4) x 8 x 8
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    activation(),
    # state size. (ngf*2) x 16 x 16
    nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    activation(),
    # state size. (ngf) x 32 x 32
    nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
    # state size. (nc) x 64 x 64
)
"""
class ConvGenerator(nn.Module):
    def __init__(self, nin, units=32, depth=3, activation=nn.LeakyReLU):
        super(ConvGenerator, self).__init__()

        ngf = units

        scale = 2**depth # 8
        layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nin, ngf * scale, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * scale),
            activation(),
            # state size. (ngf*8) x 4 x 4
            ]

        for _ in range(depth):
            layers += [ 
                nn.ConvTranspose2d(ngf * scale, ngf * scale//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * scale//2),
                activation(),
                ]
            scale = scale//2

        layers += [
            nn.ConvTranspose2d(    ngf,      1, 3, 2, 1, bias=False),
            ]

        self.layers = nn.Sequential(*layers)
        self.width = 8*2**depth - 1


    def forward(self, z):
        if len(z.size()) < 4:
            z = z.view(-1,z.size(1),1,1)
        return self.layers(z)

