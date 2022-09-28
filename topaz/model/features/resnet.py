from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from topaz.model.utils import insize_from_outsize


# below are ResNet constituent components
class MaxPool(nn.Module):
    def __init__(self, kernel_size:int, stride:int=1, dims:int=2):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride) if dims == 3 \
            else nn.MaxPool2d(kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.og_stride = stride
        self.dilation = 1
        self.padding = 0
        self.dims = dims

    def set_padding(self, pad:bool):
        if pad:
            p = self.dilation*(self.kernel_size//2) # this is bugged in pytorch...
            self.pool.padding = tuple(p for _ in range(self.dims))
            self.padding = p
        else:
            self.pool.padding = tuple(0 for _ in range(self.dims))
            self.padding = 0

    def fill(self, stride:int):
        self.pool.dilation = stride
        self.pool.padding = self.pool.padding * stride
        self.pool.stride = 1
        self.dilation = stride
        self.stride = 1
        return self.og_stride

    def unfill(self):
        self.pool.dilation = 1
        self.pool.padding = self.pool.padding // self.dilation
        self.pool.stride = self.og_stride
        self.dilation = 1
        self.stride = self.og_stride

    def forward(self, x:torch.Tensor):
        return self.pool(x)


class BasicConv(nn.Module):
    '''Basic convolutional layer for use in ResNet architectures.
    Supports 2- and 3-dimensional inputs/kernels.'''
    def __init__(self, nin:int, nout:int, kernel_size:int, dilation:int=1, stride:int=1, bn:bool=False, activation:nn.Module=nn.ReLU, dims:int=2):
        super(BasicConv, self).__init__()

        if dims == 2:
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
        elif dims == 3:
            conv = nn.Conv3d
            batch_norm = nn.BatchNorm3d
        else:
            raise ValueError(f'Unsupported number of dimensions: {dims}. Try dims=2 or dims=3.')

        bias = (not bn)
        self.conv = conv(nin, nout, kernel_size, dilation=dilation, stride=stride, bias=bias)
        if bn:
            self.bn = batch_norm(nout)
            
        self.act = activation(inplace=True)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.og_dilation = dilation
        self.padding = 0
        self.dims = dims

    def set_padding(self, pad:bool):
        if pad:
            p = self.dilation * (self.kernel_size // 2)
            self.conv.padding = tuple(p for _ in range(self.dims))
            self.padding = p
        else:
            self.conv.padding = tuple(0 for _ in range(self.dims))
            self.padding = 0

    def fill(self, stride:int):
        self.conv.dilation = tuple(self.og_dilation*stride for _ in range(self.dims))
        self.conv.stride = tuple(1 for _ in range(self.dims))
        self.conv.padding = tuple(pad * stride for pad in self.conv.padding)
        self.dilation *= stride
        return self.stride

    def unfill(self):
        stride = self.dilation // self.og_dilation
        self.conv.dilation = tuple(self.og_dilation for _ in range(self.dims))
        self.conv.stride = tuple(self.stride for _ in range(self.dims))
        self.conv.padding = tuple(pad // stride for pad in self.conv.padding)
        self.dilation = self.og_dilation

    def forward(self, x:torch.Tensor):
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        return self.act(y)
        

class ResidA(nn.Module):
    '''Residual block primitive for ResNet architectures. 
    Supports 2- and 3-dimensional inputs/kernels.'''
    def __init__(self, nin, nhidden, nout, dilation=1, stride=1, activation=nn.ReLU, bn=False, dims=2):
        super(ResidA, self).__init__()

        self.dims = dims
        if dims == 2:
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
        elif dims == 3:
            conv = nn.Conv3d
            batch_norm = nn.BatchNorm3d
        else:
            raise ValueError(f'Unsupported number of dimensions: {dims}. Try dims=2 or dims=3.')

        self.bn = bn
        bias = (not bn)

        if nin != nout:
            self.proj = conv(nin, nout, 1, stride=stride, bias=False)  
        self.conv0 = conv(nin, nhidden, 3, bias=bias)
        if self.bn:
            self.bn0 = batch_norm(nhidden)
        self.act0 = activation(inplace=True)
        self.conv1 = conv(nhidden, nout, 3, dilation=dilation, stride=stride, bias=bias)
        if self.bn:
            self.bn1 = batch_norm(nout)
        self.act1 = activation(inplace=True)

        self.kernel_size = 2*dilation + 3
        self.stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            self.conv0.padding = tuple(1 for _ in range(self.dims))
            self.conv1.padding = self.conv1.dilation
            self.padding = self.kernel_size // 2
        else:
            self.conv0.padding = tuple(0 for _ in range(self.dims))
            self.conv1.padding = self.conv1.padding
            self.padding = 0

    def fill(self, stride):
        self.conv0.dilation = tuple(stride for _ in range(self.dims))
        self.conv0.padding = tuple(pad * stride for pad in self.conv0.padding)
        
        self.conv1.dilation = tuple(dil * stride for dil in self.conv1.dilation)
        self.conv1.padding = tuple(pad * stride for pad in self.conv1.padding)
        self.conv1.stride = tuple(1 for _ in range(self.dims))
        
        if hasattr(self, 'proj'):
            self.proj.stride = tuple(1 for _ in range(self.dims))
        self.dilation = self.dilation * stride
        return self.stride

    def unfill(self):
        self.conv0.dilation = tuple(1 for _ in range(self.dims))
        self.conv0.padding = tuple(pad // self.dilation for pad in self.conv0.padding)
        
        self.conv1.dilation = tuple(dil // self.dilation for dil in self.conv1.dilation)
        self.conv1.padding = tuple(pad // self.dilation for pad in self.conv1.padding)
        self.conv1.stride = tuple(self.stride for _ in range(self.dims))
        
        if hasattr(self, 'proj'):
            self.proj.stride = tuple(self.stride for _ in range(self.dims))
        self.dilation = 1

    def forward(self, x):
        h = self.conv0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)
        y = self.conv1(h)

        edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        if self.dims == 2:
            x = x[:,:,edge:-edge,edge:-edge]
        elif self.dims == 3:
            x = x[:,:,edge:-edge,edge:-edge, edge:-edge]

        if hasattr(self, 'proj'):
            x = self.proj(x)
        elif self.conv1.stride[0] > 1:
            if self.dims == 2:
                x = x[..., ::self.stride, ::self.stride]
            elif self.dims == 3:
                x = x[..., ::self.stride, ::self.stride, ::self.stride]

        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)

        return y
        

# Sample architectures
class ResNet(nn.Module):
    '''ResNet utility functions. Must be subclassed to define network architecture.'''
    def __init__(self, dims=2, **kwargs):
        super(ResNet, self).__init__()
        self.dims = dims

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

    def forward(self, x):
        if len(x.size()) < self.dims + 2:
            x = x.unsqueeze(1) # add channels dim
        if self.pad: ## add (width-1)//2 zeros to edges of x
            p = self.width // 2
            pad = tuple(p for _ in range(self.dims * 2))
            x = F.pad(x, pad)
        z = self.features(x)
        return z


class ResNet6(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0, activation=nn.ReLU, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]

        modules = [BasicConv(1, units[0], 5, bn=bn, activation=activation, dims=self.dims)]
        modules += [MaxPool(3, stride=2, dims=self.dims)]   
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        
        modules += [ResidA(units[0], units[0], units[1], dilation=4, bn=bn, activation=activation, dims=self.dims)]
        modules += [MaxPool(3, stride=2, dims=self.dims)]    
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        
        modules += [ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation, dims=self.dims)]
        modules += [BasicConv(units[1], units[2], 5, bn=bn, activation=activation, dims=self.dims)]
        modules += nn.Dropout(p=dropout) if dropout > 0 else []
        
        self.latent_dim = units[-1]
        return modules


class ResNet8(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0, activation=nn.ReLU, pooling:MaxPool=None, **kwargs):
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

        modules = [BasicConv(1, units[0], 7, stride=stride, bn=bn, activation=activation, dims=self.dims)]
        modules += [pooling(3, stride=2, dims=self.dims)] if pooling is not None else []
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        modules += [ResidA(units[0], units[0], units[0], dilation=2, bn=bn, activation=activation, dims=self.dims),
                    ResidA(units[0], units[0], units[1], dilation=2, stride=stride, bn=bn, activation=activation, dims=self.dims)]
        modules += [pooling(3, stride=2, dims=self.dims)] if pooling is not None else []
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        modules += [ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation, dims=self.dims),
                    BasicConv(units[1], units[2], 5, bn=bn, activation=activation, dims=self.dims)]
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []

        self.latent_dim = units[-1]
        return modules


class ResNet16(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0, activation=nn.ReLU, pooling=None, **kwargs):
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

        modules = [BasicConv(1, units[0], 7, bn=bn, activation=activation, dims=self.dims),
                   ResidA(units[0], units[0], units[0], stride=stride, bn=bn, activation=activation, dims=self.dims)]
        modules += [pooling(3, stride=2, dims=self.dims)] if pooling is not None else []
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        modules += [ResidA(units[0], units[0], units[0], bn=bn, activation=activation, dims=self.dims),
                    ResidA(units[0], units[0], units[0], bn=bn, activation=activation, dims=self.dims),
                    ResidA(units[0], units[0], units[0], bn=bn, activation=activation, dims=self.dims),
                    ResidA(units[0], units[0], units[1], stride=stride, bn=bn, activation=activation, dims=self.dims)]
        modules += [pooling(3, stride=2, dims=self.dims)] if pooling is not None else []
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []
        modules += [ResidA(units[1], units[1], units[1], bn=bn, activation=activation, dims=self.dims),
                    ResidA(units[1], units[1], units[1], bn=bn, activation=activation, dims=self.dims),
                    BasicConv(units[1], units[2], 5, bn=bn, activation=activation, dims=self.dims)]
        modules += [nn.Dropout(p=dropout)] if dropout > 0 else []

        self.latent_dim = units[-1]
        return modules