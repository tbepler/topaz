import datetime
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pkg_resources
import torch
from torch import nn
import torch.nn.functional as F
from topaz.filters import AffineDenoise
from torch.utils.data import DataLoader


class L0Loss:
    def __init__(self, eps=1e-8, gamma=2):
        self.eps = eps
        self.gamma = gamma

    def __call__(self, x, y):
        return torch.mean((torch.abs(x - y) + self.eps)**self.gamma)


class DenoiseNet(nn.Module):
    def __init__(self, base_filters):
        super(DenoiseNet, self).__init__()

        self.base_filters = base_filters
        nf = base_filters
        self.net = nn.Sequential( nn.Conv2d(1, nf, 11, padding=5)
                                , nn.LeakyReLU(0.1)
                                , nn.MaxPool2d(3, stride=1, padding=1)
                                , nn.Conv2d(nf, 2*nf, 3, padding=2, dilation=2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(2*nf, 2*nf, 3, padding=4, dilation=4)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(2*nf, 3*nf, 3, padding=1)
                                , nn.LeakyReLU(0.1)
                                , nn.MaxPool2d(3, stride=1, padding=1)
                                , nn.Conv2d(nf, 2*nf, 3, padding=2, dilation=2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(3*nf, 3*nf, 3, padding=4, dilation=4)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(3*nf, 1, 7, padding=3)
                                )

    def forward(self, x):
        return self.net(x)


class DenoiseNet2(nn.Module):
    def __init__(self, base_filters, width=11):
        super(DenoiseNet2, self).__init__()

        self.base_filters = base_filters
        nf = base_filters
        self.net = nn.Sequential( nn.Conv2d(1, nf, width, padding=width//2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(nf, nf, width, padding=width//2)
                                , nn.LeakyReLU(0.1)
                                , nn.Conv2d(nf, 1, width, padding=width//2)
                                )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class UDenoiseNet(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class UDenoiseNetSmall(nn.Module):
    def __init__(self, nf=48, width=11, top_width=3):
        super(UDenoiseNetSmall, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, width, padding=width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec3 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        h = self.enc4(p3)

        # upsampling with skip connections
        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class UDenoiseNet2(nn.Module):
    # modified U-net from noise2noise paper
    def __init__(self, nf=48):
        super(UDenoiseNet2, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf, 64, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, 3, padding=1)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')

        y = self.dec1(h)

        return y


class UDenoiseNet3(nn.Module):
    def __init__(self):
        super(UDenoiseNet3, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, 48, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(48, 48, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(144, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(96, 96, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(97, 64, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, 3, padding=1)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = x - self.dec1(h) # learn only noise component

        return y


class UDenoiseNet3D(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet3D, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv3d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool3d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv3d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv3d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv3d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        o = p4.size(4)
        #h = F.upsample(h, size=(n,m))
        #h = F.upsample(h, size=(n,m), mode='bilinear', align_corners=False)
        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        o = p3.size(4)
        
        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        o = p2.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        o = p1.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        o = x.size(4)

        h = F.interpolate(h, size=(n,m,o), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y



model_name_dict = {
    # 2D models
    'unet':'unet_L2_v0.2.2.sav',
    'unet-small':'unet_small_L1_v0.2.2.sav',
    'fcnn':'fcnn_L1_v0.2.2.sav',
    'affine':'affine_L1_v0.2.2.sav',
    'unet-v0.2.1':'unet_L2_v0.2.1.sav',
    # 3D models
    'unet-3d':'unet-3d-10a-v0.2.4.sav',
    'unet-3d-10a':'unet-3d-10a-v0.2.4.sav',
    'unet-3d-20a':'unet-3d-20a-v0.2.4.sav'
}

def load_model(name, base_kernel_width=11):
    ''' paths here should be ../pretrained/denoise
    '''
    log = sys.stderr
    
    # resolve model aliases 
    pretrained = (name in model_name_dict.keys())
    if pretrained:
        name = model_name_dict[name]

    # load model architecture
    if name == 'unet_L2_v0.2.1.sav':
        model = UDenoiseNet(base_width=7, top_width=3)
    elif name == 'unet_L2_v0.2.2.sav':
        model = UDenoiseNet(base_width=11, top_width=5)
    elif name == 'unet_small_L1_v0.2.2.sav':
        model = UDenoiseNetSmall(width=11, top_width=5)
    elif name == 'fcnn_L1_v0.2.2.sav':
        model = DenoiseNet2(64, width=11)
    elif name == 'affine_L1_v0.2.2.sav':
        model = AffineDenoise(max_size=31)
    elif name == 'unet-3d-10a-v0.2.4.sav': 
        model = UDenoiseNet3D(base_width=7)
    elif name == 'unet-3d-10a-v0.2.4.sav':
        model = UDenoiseNet3D(base_width=7)
    elif name == 'unet-3d-20a-v0.2.4.sav':
        model = UDenoiseNet3D(base_width=7)
    else:
        # if not set to a pretrained model, try loading path directly
        model = torch.load(name)

    # load model parameters/state
    if pretrained:
        print('# loading pretrained model:', name, file=log)
        pkg = __name__
        path = '../pretrained/denoise/' + name
        f = pkg_resources.resource_stream(pkg, path)
        state_dict = torch.load(f) # load the parameters
        model.load_state_dict(state_dict)
    elif type(model) is OrderedDict and '3d' in name:
        state = model
        model = UDenoiseNet3D(base_width=base_kernel_width)
        model.load_state_dict(state)
    
    model.eval()
    return model


def save_model(model, epoch, save_prefix, digits=3):
    if type(model) is nn.DataParallel:
        model = model.module
    path = save_prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
    #path = save_prefix + '_epoch{}.sav'.format(epoch)
    torch.save(model, path)


def __epoch(model, dataloader, loss_fn, optim, train=True, use_cuda=False) -> float:
    #set train or evaluate mode
    model.train(train)
    
    for (source,target) in dataloader:
        n = 0
        loss_accum = 0

        if use_cuda:
            source = source.cuda()
            target = target.cuda()

        # set input_channels to 1 (BW imgs) for Conv layers
        source = source.unsqueeze(1)
        pred = model(source).squeeze(1)

        loss = loss_fn(pred, target)
        
        if train:
            loss.backward()
            optim.step()
            optim.zero_grad()

        loss = loss.item()
        b = source.size(0)

        # percentage of image/tomogram
        n += b
        delta = b*(loss - loss_accum)
        loss_accum += delta/n
        
    return loss_accum


def train_model(model, train_dataset, val_dataset, loss_fn:str='L2', optim:str='adam', lr:float=0.001, weight_decay:float=0, batch_size:int=10, num_epochs:int=500, 
                shuffle:bool=True, use_cuda:bool=False, num_workers:int=1, verbose:bool=True, save_best:bool=False, save_interval:int=None, save_prefix:str=None):
    
    output = sys.stdout
    log = sys.stderr
    # num digits to hold epoch numbers
    digits = int(np.ceil(np.log10(num_epochs)))

    if save_prefix is not None:
        save_dir = os.path.dirname(save_prefix)
        if len(save_dir) > 0 and not os.path.exists(save_dir):
            print('# creating save directory:', save_dir, file=log)
            os.makedirs(save_dir)

    start_time = time.time()
    now = datetime.datetime.now()
    print('# starting time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s'.format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)
    
    # prepare data
    num_workers = min(num_workers, mp.cpu_count())
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # create loss function
    gamma = None
    if loss_fn == 'L0':
        gamma = 2
        eps = 1e-8
        loss_fn = L0Loss(eps=eps, gamma=gamma)
    elif loss_fn == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_fn == 'L2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f'Loss function: {loss_fn} not one of [L0, L1, L2].')
    
    # create optimizer
    params = [{'params': model.parameters(), 'weight_decay': weight_decay}]
    if optim == 'adagrad':
        optim = torch.optim.Adagrad(params, lr=lr)
    elif optim == 'adam':
        optim = torch.optim.Adam(params, lr=lr)
    elif optim == 'rmsprop':
        optim = torch.optim.RMSprop(params, lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(params, lr=lr, nesterov=True, momentum=0.9)
    else:
        raise ValueError('Unrecognized optim: ' + optim)

    ## Begin model training
    print('# training model...', file=log)
    if verbose:    
        print('\t'.join(['Epoch', 'Train Loss', 'Val Loss', 'Best Val Loss']), file=output)

    best_val_loss = np.inf
    for epoch in range(num_epochs):
        
        # anneal gamma to 0
        if gamma is not None:
            loss_fn.gamma = 2 - (epoch-1)*2/num_epochs
        
        train_loss = __epoch(model, train_data, loss_fn, train=True, use_cuda=use_cuda)
        with torch.no_grad():
            val_loss = __epoch(model, val_data, loss_fn, optim, train=False, use_cuda=use_cuda)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best and save_prefix is not None:
                    model.eval().cpu()
                    save_model(model, epoch+1, save_prefix, digits=digits)
                    if use_cuda:
                        model.cuda()
        
        if verbose:
            print('\t'.join([f'# [{epoch}/{num_epochs}]'] + [str(round(num, 5)) for num in (train_loss, val_loss, best_val_loss)]), file=output, end='\r')

        # periodically save model if desired
        if (save_prefix is not None) and (save_interval is not None) and (epoch+1)%save_interval == 0:
            model.eval().cpu()
            save_model(model, epoch+1, save_prefix, digits=digits)
            if use_cuda:
                model.cuda()
    
    print('# training completed!', file=log)
    end_time = time.time()
    now = datetime.datetime.now()
    print("# ending time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s".format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)
    print("# total time:", time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)), file=log)
            
    return model
