from __future__ import print_function,division

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidMultiScaleBlock(nn.Module):
    def __init__(self, nin, ng, nout, scales=[1,2,4], pooling='max', padding=False):
        super(ResidMultiScaleBlock, self).__init__()

        self.nin = nin
        self.nout = nout

        self.blocks = nn.ModuleList()
        for s in scales:
            block = []
            if s > 1:
                pool = nn.MaxPool2d(2*(s-1)+1, stride=1, padding=s-1) 
                block.append(pool)
            if padding:
                conv = nn.Conv2d(nin, ng, 3, dilation=s, padding=s)
            else:
                conv = nn.Conv2d(nin, ng, 3, dilation=s)
            block.append(conv)
            block.append(nn.ReLU(inplace=True))
            block.append(nn.Conv2d(ng, nout, 1))
            block = nn.Sequential(*block)
            self.blocks.append(block)

        self.relu = nn.ReLU(inplace=True)

        self.proj = None
        if nout != nin:
            self.proj = nn.Conv2d(nin, nout, 1, bias=False)


    def forward(self, x):
        h = x
        if self.proj is not None:
            h = self.proj(x)

        for block in self.blocks:
            z = block(x)
            if h.size(2) > z.size(2):
                n = h.size(2) - z.size(2)
                n = n//2
                h = h[:,:,n:-n]
            if h.size(3) > z.size(3):
                n = h.size(3) - z.size(3)
                n = n//2
                h = h[:,:,:,n:-n]
            h = h + z

        return self.relu(h)


class ResidMultiScaleNet(nn.Module):
    def __init__(self, units=[64, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256, 256], scales=[1,2,4]):
        super(ResidMultiScaleNet, self).__init__()

        u = units[0]
        self.base = nn.Sequential( nn.Conv2d(1, u, 11),
                                   nn.ReLU(inplace=True),
                                 )
        width = 10
        s = max(scales)

        layers = []
        for n_out in units[1:-1]:
            layer = ResidMultiScaleBlock(u, n_out, n_out, scales=scales)
            layers.append(layer)
            u = n_out
            width += 2*s

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Sequential( nn.Conv2d(u, units[-1], 5),
                                 nn.ReLU(inplace=True),
                               )
        width += 5
        self.width = width
        self.latent_dim = units[-1]
        self.pad = False

    def fill(self, *args, **kwargs):
        self.pad = True

    def unfill(self, *args, **kwargs):
        self.pad = False

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))

        h = self.base(x)
        h = self.layers(h)
        h = self.fc(h)

        return h


class RMSNet7(nn.Module):
    def __init__(self, units=[64, 64, 128, 128, 256]):
        super(RMSNet7, self).__init__()

        u = units[0]
        self.base = nn.Sequential( nn.Conv2d(1, u, 11),
                                   nn.ReLU(inplace=True),
                                 )
        layers = [
            # 1x 8 block
            ResidMultiScaleBlock(u, units[1], units[1]),        
            # 2x 16 blocks
            ResidMultiScaleBlock(units[1], units[2], units[2], scales=[1,2,8]),
            ResidMultiScaleBlock(units[2], units[2], units[2], scales=[1,2,8]),
            # 2x 32 blocks
            ResidMultiScaleBlock(units[2], units[3], units[3], scales=[1,4,16]),
            ResidMultiScaleBlock(units[3], units[3], units[3], scales=[1,4,16]),
        ]
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Sequential( nn.Conv2d(units[3], units[4], 11),
                                 nn.ReLU(inplace=True),
                               )
        self.width = 125
        self.latent_dim = units[-1]
        self.pad = False

    def fill(self, *args, **kwargs):
        self.pad = True

    def unfill(self, *args, **kwargs):
        self.pad = False

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))

        h = self.base(x)
        h = self.layers(h)
        h = self.fc(h)

        return h


class RMSNet10(nn.Module):
    def __init__(self, units=[64, 64, 128, 128, 256]):
        super(RMSNet10, self).__init__()

        u = units[0]
        self.base = nn.Sequential( nn.Conv2d(1, u, 11),
                                   nn.ReLU(inplace=True),
                                 )
        layers = [
            # 1x 8 block
            ResidMultiScaleBlock(u, units[1], units[1]),        
            ResidMultiScaleBlock(units[1], units[1], units[1], padding=True),        
            ResidMultiScaleBlock(units[1], units[1], units[1], padding=True),        
            # 2x 16 blocks
            ResidMultiScaleBlock(units[1], units[2], units[2], scales=[1,2,8]),
            ResidMultiScaleBlock(units[2], units[2], units[2], scales=[1,2,8]),
            ResidMultiScaleBlock(units[2], units[2], units[2], scales=[1,2,8], padding=True),
            # 2x 32 blocks
            ResidMultiScaleBlock(units[2], units[3], units[3], scales=[1,4,16]),
            ResidMultiScaleBlock(units[3], units[3], units[3], scales=[1,4,16]),
        ]
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Sequential( nn.Conv2d(units[3], units[4], 11),
                                 nn.ReLU(inplace=True),
                               )
        self.width = 125
        self.latent_dim = units[-1]
        self.pad = False

    def fill(self, *args, **kwargs):
        self.pad = True

    def unfill(self, *args, **kwargs):
        self.pad = False

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))

        h = self.base(x)
        h = self.layers(h)
        h = self.fc(h)

        return h



class PyramidNet(nn.Module):
    def __init__(self, base_units=64, scales=[2,4,8,16], pooling='max'):
        super(PyramidNet, self).__init__()

        u = base_units
        self.encoder_blocks = nn.ModuleList()
        
        base = nn.Sequential( nn.Conv2d(1, u, 11, padding=5)
                            , nn.ReLU(inplace=True)
                            )
        self.encoder_blocks.append(base)

        for s in scales:
            block = nn.Sequential(
                        nn.MaxPool2d(2*(s-1)+1, padding=s-1, stride=1),             
                        nn.Conv2d(u, u, 3, padding=s, dilation=s),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(u, u, 3, padding=s, dilation=s),
                        nn.ReLU(inplace=True),
                    )
            self.encoder_blocks.append(block)

        u_in = 1 + u + len(scales)*u # input to next section is concatenation of ^ blocks

        u_in = 0
        width = 0
        self.decoder_blocks = nn.ModuleList()
        # decoder blocks mirror the encoder blocks
        for s in scales[::-1][1:]:
            block = nn.Sequential(
                        nn.Conv2d(u + u_in, 2*u, 3, dilation=s),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(2*u, 2*u, 3, dilation=s),
                        nn.ReLU(inplace=True),
                    )
            self.decoder_blocks.append(block)
            u_in = 2*u
            width += 4*s

        # last block
        self.final_block = nn.Sequential(
                    nn.Conv2d(u_in + u, 2*u, 3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2*u, 4*u, 7),
                    nn.ReLU(inplace=True),
                )

        width += 9
        self.width = width
        self.pad = False
        self.latent_dim = 4*u

    def fill(self, *args, **kwargs):
        self.pad = True

    def unfill(self, *args, **kwargs):
        self.pad = False

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width//2
            x = F.pad(x, (p,p,p,p))

        h = x
        zs = []
        for block in self.encoder_blocks:
            h = block(h)
            zs.append(h)
        
        zs = zs[:-1][::-1]
        for z,block in zip(zs, self.decoder_blocks):
            h = block(h)

            if z.size(2) > h.size(2):
                n = z.size(2) - h.size(2)
                n = n//2
                z = z[:,:,n:-n]
            if z.size(3) > h.size(3):
                n = z.size(3) - h.size(3)
                n = n//2
                z = z[:,:,:,n:-n]

            h = torch.cat([z,h], 1)

        h = self.final_block(h)
        return h
        

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





