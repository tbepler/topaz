from __future__ import print_function, division

import os
import glob

import numpy as np
from PIL import Image
import torch

import topaz.mrc as mrc

class ImageDirectoryLoader:
    def __init__(self, rootdir, pathspec=os.path.join('{source}', '{image_name}'), format='tiff'
                , standardize=False):
        self.rootdir = rootdir
        self.pathspec = pathspec
        self.format = format
        self.standardize = standardize

    def get(self, *args, **kwargs):
        ext = self.pathspec.format(*args, **kwargs) + '.' + self.format
        path = os.path.join(self.rootdir, ext)
        if self.format == 'mrc':
            with open(path, 'rb') as f:
                content = f.read()
            image, header, extended_header = mrc.parse(content)
            if self.standardize:
                image = image - header.amean
                image /= header.rms
        else:
            image = Image.open(path)
            fp = image.fp
            image.load()
            fp.close()
            image = np.array(image, copy=False)
            if self.standardize:
                image = (image - image.mean())/image.std()
        return Image.fromarray(image)

class ImageTree:
    def __init__(self, images):
        self.images = images

    def get(self, source, name):
        return self.images[source][name]

def load_mrc(path, standardize=False):
    with open(path, 'rb') as f:
        content = f.read()
    image, header, extended_header = mrc.parse(content)
    if standardize:
        image = image - header.amean
        image /= header.rms
    return Image.fromarray(image)

def load_pil(path, standardize=False):
    image = Image.open(path)
    fp = image.fp
    image.load()
    fp.close()
    if standardize:
        image = np.array(image, copy=False)
        image = (image - image.mean())/image.std()
        image = Image.fromarray(image)
    return image

def load_image(source, name, rootdir, standardize=False):
    path = os.path.join(rootdir, source, name) + '.*'
    path = glob.glob(path)[0]
    ext = os.path.splitext(path)[1]
    if ext == 'mrc':
        im = load_mrc(path, standardize=standardize)
    else:
        im = load_pil(path, standardize=standardize)
    return im

def load_images(sources, names, rootdir, standardize=False):
    root = {}
    for source,name in zip(sources, names):
        branch = root.get(source, {})
        image = load_image(source, name, rootdir, standardize=standardize)
        branch[name] = image
        root[source] = branch
    return root

class LabeledImageCropDataset:
    def __init__(self, images, labels, crop):
        self.images = images
        self.labels = labels
        self.crop = crop

    def __getitem__(self, idx):
        g, (i, coord) = idx
        im = self.images[g][i]
        L = torch.from_numpy(self.labels[g][i].ravel()).unsqueeze(1)
        label = L[coord].float()

        ## crop the image
        x = coord % im.width
        y = coord // im.width
        xmi = x - self.crop//2
        xma = xmi + self.crop
        ymi = y - self.crop//2
        yma = ymi + self.crop
        im = im.crop((xmi, ymi, xma, yma))

        return im, label

class SegmentedImageDataset:
    def __init__(self, images, labels, to_tensor=False):
        self.images = images
        self.labels = labels
        self.size = sum(len(g) for g in images)
        self.to_tensor = to_tensor

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        j = 0
        while i >= len(self.images[j]):
            i -= len(self.images[j])
            j += 1
        im = self.images[j][i]
        label = self.labels[j][i]

        if self.to_tensor:
            im = torch.from_numpy(np.array(im, copy=False))
            label = torch.from_numpy(np.array(label, copy=False)).float()

        return im, label










