from __future__ import print_function, division

import torch

import topaz
from topaz.model.features.basic import BasicConv
from topaz.model.features.resnet import ResNet16, ResNet8, ResNet6
from topaz.model.classifier import LinearClassifier
from topaz.model.utils import load_state_dict_from_pkg

resnet16 = ResNet16
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
     

def load_model(path):
    if path == 'resnet16':
        name = 'resnet16_u64.sav'
        model = LinearClassifier(ResNet16(units=64, bn=False))
    elif path == 'resnet16_u64':
        name = 'resnet16_u64.sav'
        model = LinearClassifier(ResNet16(units=64, bn=False))
    elif path == 'resnet16_u32':
        name = 'resnet16_u32.sav'
        model = LinearClassifier(ResNet16(units=32, bn=False))
    elif path == 'resnet8':
        name = 'resnet8_u64.sav'
        model = LinearClassifier(ResNet8(units=64, bn=False))
    elif path == 'resnet8_u64':
        name = 'resnet8_u64.sav'
        model = LinearClassifier(ResNet8(units=64, bn=False))
    elif path == 'resnet8_u32':
        name = 'resnet8_u32.sav'
        model = LinearClassifier(ResNet8(units=32, bn=False))


    else: # load model using torch load
        model = torch.load(path, weights_only=False)
        return model

    # load the pretrained model
    pkg = topaz.__name__
    path = f'pretrained/detector/{name}'
    state_dict = load_state_dict_from_pkg(pkg, path)
    model.load_state_dict(state_dict)

    return model
