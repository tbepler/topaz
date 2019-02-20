#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

import topaz.utils.files as file_utils
from topaz.utils.printing import report
from topaz.utils.data.loader import load_images_from_list
from topaz.utils.data.coordinates import match_coordinates_to_images

name = 'pimax'
help = 'estimate an upper bound on pi using the alphamax algorithm'

def add_arguments(parser):

    # set GPU and number of worker threads
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')
    parser.add_argument('--num-workers', default=0, type=int, help='number of worker processes for data augmentation (default: 0)')

    # group arguments into sections

    data = parser.add_argument_group('training data arguments (required)')

    data.add_argument('--train-images', help='path to file listing the training images. also accepts directory path from which all images are loaded.')
    data.add_argument('--train-targets', help='path to file listing the training particle coordinates')

    data.add_argument('-k', '--k-fold', default=5, type=int, help='split the training set into K folds for cross validation part of pi estimation (default: 5)')
    data.add_argument('--cross-validation-seed', default=42, type=int, help='random seed for partitioning data into folds (default: 42)')
    
    
    data = parser.add_argument_group('data format arguments (optional)')
    ## optional format of the particle coordinates file
    data.add_argument('--format', dest='format_', choices=['auto', 'coord', 'csv', 'star', 'box'], default='auto'
                       , help='file format of the particle coordinates file (default: detect format automatically based on file extension)')
    data.add_argument('--image-ext', default='', help='sets the image extension if loading images from directory. should include "." before the extension (e.g. .tiff). (default: find all extensions)')

    
    training = parser.add_argument_group('training arguments (optional)')
    # training parameters
    training.add_argument('--radius', default=3, type=int, help='pixel radius around particle centers to consider positive (default: 3)')
    training.add_argument('--bagging', default=10, type=int, help='number of models to bag (ensemble) on each fold (default: 10)')

    training.add_argument('--l2', default=0.0, type=float, help='l2 regularizer on the model parameters (default: 0)')

    training.add_argument('--learning-rate', default=0.0003, type=float, help='learning rate for the optimizer (default: 0.0003)') 

    training.add_argument('--minibatch-size', default=1024, type=int, help='number of data points per minibatch (default: 1024)')
    training.add_argument('--num-steps', default=100000, type=int, help='number of SGD steps to train each model for (default: 100k)')


    model = parser.add_argument_group('model arguments (optional)')

    model.add_argument('--size', type=int, default=31, help='window size for the classifier (default: 31)')
    model.add_argument('--hidden-dim', type=int, default=200, help='hidden dimension of the classifier (default: 200)')


    outputs = parser.add_argument_group('output file arguments (optional)')
    outputs.add_argument('--save-prefix', help='path prefix to save trained models each epoch')
    outputs.add_argument('-o', '--output', help='destination to write the train/test curve')

    return parser

def match_images_targets(images, targets, radius):
    matched = match_coordinates_to_images(targets, images, radius=radius)
    ## unzip into matched lists
    images = []
    targets = []
    for key in matched:
        these_images,these_targets = zip(*list(matched[key].values()))
        images += these_images
        targets += these_targets

    return images, targets


def load_data(train_images, train_targets, radius, format_='auto', image_ext=''):

    # if train_images is a directory path, map to all images in the directory
    if train_images.endswith(os.sep):
        paths = glob.glob(train_images + '*' + image_ext)
        valid_paths = []
        image_names = []
        for path in paths:
            name = os.path.basename(path)
            name,ext = os.path.splitext(name)
            if ext in ['.mrc', '.tiff', '.png']:
                image_names.append(name)
                valid_paths.append(path)
        train_images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
    else:
        train_images = pd.read_csv(train_images, sep='\t') # training image file list
    #train_targets = pd.read_csv(train_targets, sep='\t') # training particle coordinates file
    train_targets = file_utils.read_coordinates(train_targets, format=format_)

    # check for source columns
    if 'source' not in train_images and 'source' not in train_targets:
        train_images['source'] = 0
        train_targets['source'] = 0
    # load the images and create target masks from the particle coordinates
    train_images = load_images_from_list(train_images.image_name, train_images.path
                                        , sources=train_images.source)

    # discard coordinates for micrographs not in the set of images
    # and warn the user if any are discarded
    names = set()
    for k,d in train_images.items():
        for name in d.keys():
            names.add(name)
    check = train_targets.image_name.apply(lambda x: x in names)
    missing = train_targets.image_name.loc[~check].unique().tolist()
    if len(missing) > 0:
        print('WARNING: {} micrographs listed in the coordinates file are missing from the training images. Image names are listed below.'.format(len(missing)), file=sys.stderr)
        print('WARNING: missing micrographs are: {}'.format(missing), file=sys.stderr)
    train_targets = train_targets.loc[check]

    # check that the particles roughly fit within the images
    # if they don't, the user may not have scaled the particles/images correctly
    width = 0
    height = 0
    for k,d in train_images.items():
        for image in d.values():
            w,h = image.size
            if w > width:
                width = w
            if h > height:
                height = h
    out_of_bounds = (train_targets.x_coord > width) | (train_targets.y_coord > height)
    count = out_of_bounds.sum()
    if count > int(0.1*len(train_targets)): # arbitrary cutoff of more than 10% of particles being out of bounds...
        print('WARNING: {} particle coordinates are out of the micrograph dimensions. Did you scale the micrographs and particle coordinates correctly?'.format(count), file=sys.stderr)
    #  also check that the coordinates fill most of the micrograph
    x_max = train_targets.x_coord.max()
    y_max = train_targets.y_coord.max()
    if x_max < 0.7*width and y_max < 0.7*height: # more arbitrary cutoffs
        print('WARNING: no coordinates are observed with x_coord > {} or y_coord > {}. Did you scale the micrographs and particle coordinates correctly?'.format(x_max, y_max), file=sys.stderr)

    num_micrographs = sum(len(train_images[k]) for k in train_images.keys())
    num_particles = len(train_targets)
    report('Loaded {} training micrographs with {} labeled particles'.format(num_micrographs, num_particles))

    train_images, train_targets = match_images_targets(train_images, train_targets, radius)
    return train_images, train_targets


def cross_validation_split(k, images, targets, seed=42):
    import topaz.utils.data.partition
    random = np.random.RandomState(seed)
    ## calculate number of positives per image for stratified split
    index = []
    count = []
    for i in range(len(targets)):
        index.append(i)
        count.append(targets[i].sum())
    counts_table = pd.DataFrame({'image_name': index, 'count': count})
    counts_table['source'] = 0
    partitions = list(topaz.utils.data.partition.kfold(k, counts_table))

    return partitions


def make_train_val(partitions, fold, images, targets):
    ## make the split from the partition indices
    train_table,validate_table = partitions[fold]

    test_images = []
    test_targets = []
    for _,row in validate_table.iterrows():
        i = row['image_name']
        test_images.append(images[i])
        test_targets.append(targets[i])

    train_images = []
    train_targets = []
    for _,row in train_table.iterrows():
        i = row['image_name']
        train_images.append(images[i])
        train_targets.append(targets[i])

    return train_images, train_targets, test_images, test_targets


def make_traindataset(X, Y, crop):
    from topaz.utils.data.loader import LabeledRegionsDataset
    from topaz.utils.data.sampler import RandomImageTransforms
    
    size = int(np.ceil(crop*np.sqrt(2)))
    if size % 2 == 0:
        size += 1
    dataset = LabeledRegionsDataset(X, Y, size)
    transformed = RandomImageTransforms(dataset, crop=crop, to_tensor=True)

    return transformed


def fit_steps(model, optim, data_iterator, num_steps, use_cuda=False):

    criteria = nn.BCEWithLogitsLoss()

    n = 0
    loss_accum = 0

    for step in range(1, num_steps+1):

        x,y = next(data_iterator)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        x = Variable(x).unsqueeze(1)
        y = Variable(y).view(-1).float()
        
        logits = model(x).view(-1)

        loss = criteria(logits, y)
        loss.backward()

        optim.step()
        optim.zero_grad()

        loss = loss.data[0]
        n += x.size(0)

        delta = x.size(0)*(loss - loss_accum)
        loss_accum += delta/n

        if step % 10 == 0:
            print('# [{}/{}] loss={:.5f}'.format(step, num_steps, loss_accum), end='\r', file=sys.stderr)

    print(' '*80, end='\r', file=sys.stderr)


    return loss_accum


def predict(models, images, use_cuda=False, padding=0):
    predicts = []
    for im in images:
        x = np.array(im, copy=False)
        x = Variable(torch.from_numpy(x), requires_grad=False)
        if use_cuda:
            x = x.cuda()
        x = x.view(1, 1, x.size(0), x.size(1))
        if padding > 0:
            x = F.pad(x, (padding,padding,padding,padding))
        p = 0
        for model in models:
            p = torch.sigmoid(model(x)).data + p

        p /= len(models)
        predicts.append(p.cpu().numpy())

    return predicts


def batch_iterator(dataset, batch_size, num_workers=0):
    iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    while True:
        for batch in iterator:
            yield batch


def main(args):
    ## set the device
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.device)
    report('Using device={} with cuda={}'.format(args.device, use_cuda))
    
    ## load the data
    radius = args.radius # number of pixels around coordinates to label as positive
    images, targets = \
            load_data(args.train_images,
                      args.train_targets,
                      radius,
                      format_=args.format_,
                      image_ext=args.image_ext
                     )

    batch_size = args.minibatch_size
    num_steps = args.num_steps
    num_bags = args.bagging
    lr = args.learning_rate

    size = args.size
    hidden_dim = args.hidden_dim
    
    ## fit classifiers with cross validation and bagging
    partitions = cross_validation_split(args.k_fold, images, targets, seed=args.cross_validation_seed)

    labels = [] # these are the observed labels per region
    logits = [] # these are the predicted logits per region

    for fold in range(args.k_fold):
        train_images,train_targets,test_images,test_targets = make_train_val(partitions, fold, images, targets)

        # make the iterator for the training images
        train_dset = make_traindataset(train_images, train_targets, size)
        train_iterator = batch_iterator(train_dset, batch_size, num_workers=args.num_workers)

        # fit the models to ensemble
        models = []
        for i in range(num_bags):
            ## initialize the model
            model = nn.Sequential(
                        nn.Conv2d(1, hidden_dim, kernel_size=size),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden_dim, 1, kernel_size=1)
            )
            if use_cuda:
                model = model.cuda()

            ## initialize the optimizer
            optim = torch.optim.Adam(model.parameters(), lr=lr)

            ## fit the model
            loss = fit_steps(model, optim, train_iterator, num_steps, use_cuda=use_cuda)
            report('Fold: {}, model: {}, loss = {}'.format(fold, i+1, loss))

            models.append(model)

        # classify the heldout data with the bag of classifiers
        p = size//2
        y_list = predict(models, test_images, use_cuda=use_cuda, padding=p)

        for i in range(len(test_targets)):
            t = test_targets[i]
            y = y_list[i]
            labels.append(t.ravel())
            logits.append(y.ravel())
    
    ## concatenate the labels and logits
    ## then proceed with alphamax algorithm

    labels = np.concatenate(labels, 0)
    logits = np.concatenate(logits, 0)

    print(len(labels), len(logits))
    """
    # subsample data if too big
    ns = 500
    if len(labels) > ns:
        index = np.random.choice(len(labels), size=ns, replace=False)
        labels = labels[index]
        logits = logits[index]
    """

    ## we need to estimate the density of the logits
    ## using a non-parametric model (in this case, histogram based)
    x_1 = logits[labels == 1] # these are our positive data points
    x = logits # these are all data points

    # make kernel density estimate based on histogram
    mi = np.min(x)
    ma = np.max(x)
    bins = np.linspace(mi, ma, 201)

    index = np.digitize(x, bins[1:-1])
    index_1 = index[labels == 1]

    # find the weight of each bin
    c,_ = np.histogram(x, bins=bins)
    w = c/c.sum()
    c,_ = np.histogram(x_1, bins=bins)
    w_1 = c/c.sum()

    # now, make kernel density estimate
    k = w[index]
    k_1 = k[labels == 1]

    f_hat_1 = w_1[index]

    index = torch.from_numpy(index)
    index_1 = torch.from_numpy(index_1)
    w = torch.from_numpy(w)
    w_1 = torch.from_numpy(w_1)
    k = torch.from_numpy(k)
    k_1 = torch.from_numpy(k_1)
    f_hat_1 = torch.from_numpy(f_hat_1)

    index = Variable(index, requires_grad=False)
    index_1 = Variable(index_1, requires_grad=False)
    w = Variable(w, requires_grad=False)
    w_1 = Variable(w_1, requires_grad=False)
    k = Variable(k, requires_grad=False)
    k_1 = Variable(k_1, requires_grad=False)
    f_hat_1 = Variable(f_hat_1, requires_grad=False)

    # define the optimization problem
    def nll(beta):
        beta = torch.from_numpy(beta)
        beta = Variable(beta, requires_grad=False)

        b = beta[index]
        b_1 = beta[index_1]

        h_1 = torch.sum(torch.log(b_1*k_1 + 1e-10))

        alpha = torch.sum(w*beta)
        h_0 = (1-alpha)*(1-b)*k/torch.sum((1-beta)*w)
        f_1 = alpha*b*f_hat_1/torch.sum(beta*w_1)

        h = f_1 + h_0
        h = torch.sum(torch.log(h + 1e-10))

        loss = -h_1 - h
        return loss.data[0]

    def nll_jac(beta): # Jacobian of the NLL
        beta = torch.from_numpy(beta)
        beta = Variable(beta, requires_grad=True)

        b = beta[index]
        b_1 = beta[index_1]

        h_1 = torch.sum(torch.log(b_1*k_1 + 1e-10))

        alpha = torch.sum(w*beta)
        h_0 = (1-alpha)*(1-b)*k/torch.sum((1-beta)*w)
        f_1 = alpha*b*f_hat_1/torch.sum(beta*w_1)

        h = f_1 + h_0
        h = torch.sum(torch.log(h + 1e-10))

        nll = -h_1 - h

        nll.backward()
        beta_grad = beta.grad.data.numpy()
        return beta_grad


    # for each alpha slice, find optimal beta
    from scipy.optimize import minimize

    # what values do we want to try?
    alphas = np.linspace(0, 0.5, 101)
    alphas = alphas[1:] # discard pi=0
    loglike = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha = alphas[i]
        beta_init = alpha*np.ones(len(bins)-1)
        # define constraint
        constraint = {'type': 'eq'
                     ,'fun': lambda a: np.mean(a) - alpha
                     ,'jac': lambda a: np.ones_like(a)/len(a)
                     }
        bounds = [(0,1)]*len(beta_init)
        result = minimize(nll, jac=nll_jac, x0=beta_init, bounds=bounds, constraints=constraint)
        loglike[i] = -result.fun
        print(alpha, loglike[i])

    """
    # make kernel density estimate based on Gaussian KDE
    s2 = 0.1
    k = np.exp(-(x - x[:,np.newaxis])**2/s2)/len(x)
    k_1 = k[labels == 1]
    log_k = -(x - x[:,np.newaxis])**2/s2 - np.log(len(x))
    log_k_1 = log_k[labels == 1]
    log_k_hat_1 = -(x_1 - x[:,np.newaxis])**2/s2 - np.log(len(x_1)) # estimate for all x using only x_1
    k_hat_1 = np.exp(log_k_hat_1) # estimate for all x using only x_1
    f_hat_1 = np.sum(k_hat_1, 1) # probability of x given x_1

    def logsumexp(a):
        ma = np.max(a, axis=1, keepdims=True)
        s = np.exp(a - ma).sum(axis=1, keepdims=True)
        ls = ma + np.log(s)
        return ls[:,0]

    def logsumexp2(a, b):
        ma = np.maximum(a, b)
        s = np.exp(a - ma) + np.exp(b - ma)
        return ma + np.log(s)


    # define the optimization problem
    def nll(beta):
        log_beta = np.log(beta)
        # first term is over only positives
        log_h_1 = np.sum(logsumexp(log_beta + log_k_1))

        alpha = np.mean(beta)
        log_1_m_beta = np.log(1-beta)

        log_h_0 = logsumexp(log_1_m_beta + log_k) + np.log(1-alpha)
        log_f_1 = np.log(f_hat_1) + np.log(alpha)
        log_h = np.sum(logsumexp2(log_f_1, log_h_0))

        return -log_h_1 - log_h

    # for each alpha slice, find optimal beta
    from scipy.optimize import minimize

    # what values do we want to try?
    alphas = np.linspace(0.001, 0.5, 100)
    loglike = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha = alphas[i]
        beta_init = alpha*np.ones(len(x))
        # define constraint
        constraint = {'type': 'eq'
                     ,'fun': lambda a: np.mean(a) - alpha
                     }
        bounds = [(0,1)]*len(beta_init)
        result = minimize(nll, x0=beta_init, bounds=bounds, constraints=constraint)
        loglike[i] = -result.fun
        print(alpha, loglike[i])
    """
        
    print(alphas)
    print(loglike)


    report('Done!')


if __name__ == '__main__':
    import argparse
    parser = ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)





