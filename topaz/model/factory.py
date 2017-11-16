from __future__ import print_function, division

from topaz.model.features.basic import BasicConv 
from topaz.model.features.resnet import ResNet8, ResNet6

resnet8 = ResNet8
resnet6 = ResNet6

def conv127(*args, **kwargs):
    layers = [7, 5, 5, 5, 5]
    return BasicConv(layers, *args, **kwargs)

def conv63(*args, **kwargs):
    layers = [7, 5, 5, 5]
    return BasicConv(layers, *args, **kwargs)

def conv31(*args, **kwargs):
    layers = [7, 5, 5]
    return BasicConv(layers, *args, **kwargs)


def get_feature_extractor(model, *args, **kwargs):
    constructor = eval(model)
    return constructor(*args, **kwargs)
     





