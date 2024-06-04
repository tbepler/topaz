#!/usr/bin/env python
from __future__ import division, print_function

import glob
import multiprocessing as mp
import os
import sys
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import topaz.methods as methods
import topaz.model.classifier as C
import topaz.utils.data.partition
import topaz.utils.files as file_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from topaz.metrics import average_precision
from topaz.model.classifier import classify_patches
from topaz.model.factory import get_feature_extractor, load_model
from topaz.model.generative import ConvGenerator
from topaz.stats import calculate_pi
from topaz.utils.data.coordinates import match_coordinates_to_images
from topaz.utils.data.loader import (LabeledImageCropDataset,
                                     SegmentedImageDataset,
                                     load_images_from_list)
from topaz.utils.data.sampler import (RandomImageTransforms,
                                      StratifiedCoordinateSampler)
from topaz.utils.printing import report
from torch.utils.data.dataloader import DataLoader


def match_images_targets(images:dict, targets:pd.DataFrame, radius:float, dims:int=2, use_cuda:bool=False) \
    -> Tuple[List[List[Union[Image.Image,np.ndarray]]], List[List[np.ndarray]]]:
    '''Given names mapped to images and a DataFrame of coordinates, returns coordinates as mask of the same shape 
    as corresponding image. Returns lists of lists of arrays, each list of arrays corresponding to an input source.'''
    matched = match_coordinates_to_images(targets, images, radius=radius, dims=dims, use_cuda=use_cuda)
    ## unzip into matched lists
    images = []
    targets = []
    for key in matched:
        these_images,these_targets = zip(*list(matched[key].values()))
        images.append(list(these_images))
        targets.append(list(these_targets))
    return images, targets


def filter_targets_missing_images(images:pd.DataFrame, targets:pd.DataFrame, mode:str='training'):
    '''Discard target coordinates for micrographs not in the set of images. Warn the user if any are discarded.'''
    names = set()
    for k,d in images.items():
        for name in d.keys():
            names.add(name)
    check = targets.image_name.apply(lambda x: x in names)
    missing = targets.image_name.loc[~check].unique().tolist()
    if len(missing) > 0:
        print(f'WARNING: {len(missing)} micrographs listed in the coordinates file are missing from the {mode} images. Image names are listed below.', file=sys.stderr)
        print(f'WARNING: missing micrographs are: {missing}', file=sys.stderr)
    targets = targets.loc[check]    
    return targets


def load_image_set(images_path, targets_path, image_ext, radius, format_, as_images=True, mode='training', 
                   dims=2, use_cuda=False) -> Tuple[List[List[Union[Image.Image,np.ndarray]]], List[List[np.ndarray]]]:
    # if train_images is a directory path, map to all images in the directory
    if os.path.isdir(images_path):
        paths = glob.glob(images_path + os.sep + '*' + image_ext)
        valid_paths, image_names = [], []
        for path in paths:
            name = os.path.basename(path)
            name,ext = os.path.splitext(name)
            if ext in ['.mrc', '.tiff', '.png']:
                image_names.append(name)
                valid_paths.append(path)
        images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
    else:
        images = pd.read_csv(images_path, sep='\t') # training image file list
    targets = file_utils.read_coordinates(targets_path, format=format_)

    # check for source columns
    if 'source' not in images and 'source' not in targets:
        images['source'] = 0
        targets['source'] = 0
        
    # load the images and create target masks from the particle coordinates
    images = load_images_from_list(images.image_name, images.path, sources=images.source, as_images=as_images)

    # remove target coordinates missing corresponding images
    targets = filter_targets_missing_images(images, targets, mode=mode)

    # check that particles roughly fit in images; if don't, may not have scaled particles/images correctly
    check_particle_image_bounds(images, targets, dims=dims)
    
    num_micrographs = sum(len(images[k]) for k in images.keys())
    num_particles = len(targets)
    report(f'Loaded {num_micrographs} {mode} micrographs with {num_particles} labeled particles')
    if num_particles == 0 and mode == 'training':
        print('ERROR: no training particles specified. Check that micrograph names in the particles file match those in the micrographs file/directory.', file=sys.stderr)
        raise Exception('No training particles.')

    #convert targets to masks of the same shape as their image
    images, targets = match_images_targets(images, targets, radius, dims=dims, use_cuda=use_cuda)
    report(f'Created target binary masks for {mode} micrographs.')
    return images, targets


def check_particle_image_bounds(images:pd.DataFrame, targets:pd.DataFrame, dims=2):
    '''Check that the target particles roughly fit within the images/micrographs. If they don't, 
    prints a warning that images/particle coordinates may not have been scaled correctly.'''
    width, height, depth = 0, 0, 0
    #set maximum bounds from image shapes
    for k,d in images.items():
        for image in d.values():
            if dims == 2:
                # if numpy array (H, W), reverse height and width order to (W,H)
                w,h = image.size if (type(image) == Image.Image) else (image.shape[1], image.shape[0])
            elif dims == 3:
                d, h, w = image.shape #3D arrays can only be read as numpy arrays             
            width, height = max(w, width), max(h, height)
            depth = max(d, depth) if dims==3 else 0        
    out_of_bounds = (targets.x_coord > width) | (targets.y_coord > height) | (dims==3 and targets.z_coord > depth)
    count = out_of_bounds.sum()
    
    # arbitrary cutoff of more than 10% of particles being out of bounds...
    if count > int(0.1*len(targets)): 
        print(f'WARNING: {count} particle coordinates are out of the micrograph dimensions. Did you scale the micrographs and particle coordinates correctly?', file=sys.stderr)
    # also check that the coordinates fill most of the micrograph, cutoffs arbitrary
    x_max, y_max = targets.x_coord.max(), targets.y_coord.max()
    z_max = targets.z_coord.max() if dims==3 else None
    xy_below_cutoff = (x_max < 0.7 * width) and (y_max < 0.7 * height)
    if xy_below_cutoff:        
        z_output = f'or z_coord > {z_max}' if (dims == 3) and (z_max < 0.7 * depth) else ''
        output = f'WARNING: no coordinates are observed with x_coord > {x_max} or y_coord > {y_max} {z_output}. \
                Did you scale the micrographs and particle coordinates correctly?'
        print(output, file=sys.stderr)


def make_traindataset(X:List[List[Union[Image.Image, np.ndarray]]], Y:List[List[np.ndarray]], crop:int, 
                      dims:int=2) -> Union[LabeledImageCropDataset, RandomImageTransforms]:
    '''Extract and augment (via rotation, mirroring, and cropping) crops from the input arrays.'''
    size = int(np.ceil(crop*np.sqrt(2))) #multiply square side by hypotenuse to ensure rotations dont remove corners
    size += 1 if size % 2 == 0 else 0
    dataset = LabeledImageCropDataset(X, Y, size, dims=dims)
    if dims == 3: #don't augment 3D volumes
        transformed = RandomImageTransforms(dataset, crop=crop, dims=dims, flip=False, rotate=False)
    else:
        transformed = RandomImageTransforms(dataset, crop=crop, dims=dims, flip=True, rotate=True)
    return transformed


def calculate_positive_fraction(targets):
    per_source = []
    for source_targets in targets:
        positives = sum(target.sum() for target in source_targets)
        total = sum(target.size for target in source_targets)
        per_source.append(positives/total)
    return np.mean(per_source)


def cross_validation_split(k:int, fold:int, images:List[Union[Image.Image, np.ndarray]], targets:List[np.ndarray], random=np.random):
    ## calculate number of positives per image for stratified split
    source = []
    index = []
    count = []
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            source.append(i)
            index.append(j)
            count.append(targets[i][j].sum())
    counts_table = pd.DataFrame({'source': source, 'image_name': index, 'count': count})
    partitions = list(topaz.utils.data.partition.kfold(k, counts_table))

    ## make the split from the partition indices
    train_table,validate_table = partitions[fold]

    test_images = [[]*len(images)]
    test_targets = [[]*len(targets)]
    for _,row in validate_table.iterrows():
        i = row['source']
        j = row['image_name']
        test_images[i].append(images[i][j])
        test_targets[i].append(targets[i][j])

    train_images = [[]*len(images)]
    train_targets = [[]*len(targets)]
    for _,row in train_table.iterrows():
        i = row['source']
        j = row['image_name']
        train_images[i].append(images[i][j])
        train_targets[i].append(targets[i][j])

    return train_images, train_targets, test_images, test_targets


def load_data(train_images_path:str, train_targets_path:str, test_images_path:str, test_targets_path:str, radius:float, k_fold:int=0, fold:int=0, 
              cross_validation_seed:int=42, format_:str='auto', image_ext:str='', as_images:bool=True, dims:int=2, use_cuda:bool=False):
    '''Load training and testing (if available) images and picked particles. May split training data for cross-validation if no testing data are given.'''
    #load training images and target particles
    train_images, train_targets = load_image_set(train_images_path, train_targets_path, image_ext=image_ext, radius=radius, 
                                                 format_=format_, as_images=as_images, mode='training', dims=dims, use_cuda=use_cuda)
    #load test images and target particles or split training
    if test_images_path is not None:
        test_images, test_targets = load_image_set(test_images_path, test_targets_path, image_ext=image_ext, radius=radius, 
                                                   format_=format_, as_images=as_images, mode='test', dims=dims, use_cuda=use_cuda)
    elif k_fold > 1:
        ## seed for partitioning the data
        random = np.random.RandomState(cross_validation_seed)
        ## make the split
        train_images, train_targets, test_images, test_targets = cross_validation_split(k_fold, fold, train_images, train_targets, random=random)

        n_train = sum(len(images) for images in train_images)
        n_test = sum(len(images) for images in test_images)
        report('Split into {} train and {} test micrographs'.format(n_train, n_test))       
    else:
        test_images, test_targets = None, None
        
    return train_images, train_targets, test_images, test_targets


def report_data_stats(train_images, train_targets, test_images, test_targets):
    '''Assumes data are given as torch Tensors.'''
    report('source\tsplit\tp_observed\tnum_positive_regions\ttotal_regions')
    num_positive_regions = 0
    total_regions = 0
    for i in range(len(train_images)):
        p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
        p = int(p)
        total = sum(train_targets[i][j].numel() for j in range(len(train_targets[i])))
        num_positive_regions += p
        total_regions += total
        p_observed = p/total
        p_observed = '{:.3g}'.format(p_observed)
        report(str(i)+'\t'+'train'+'\t'+p_observed+'\t'+str(p)+'\t'+str(total))
        if test_targets is not None:
            p = sum(test_targets[i][j].sum() for j in range(len(test_targets[i])))
            p = int(p)
            total = sum(test_targets[i][j].numel() for j in range(len(test_targets[i])))
            p_observed = p/total
            p_observed = '{:.3g}'.format(p_observed)
            report(str(i)+'\t'+'test'+'\t'+p_observed+'\t'+str(p)+'\t'+str(total))
    return num_positive_regions, total_regions


def make_model(args):
    '''Load or create 2D models.'''
    report('Loading model:', args.model)
    if args.model.endswith('.sav'): # loading pretrained model
        model = torch.load(args.model)
        model.train()
        return model

    report('Model parameters: units={}, dropout={}, bn={}'.format(args.units, args.dropout, args.bn))

    # initialize the model 
    units = args.units
    dropout = args.dropout
    bn = args.bn == 'on'
    pooling = args.pooling
    unit_scaling = args.unit_scaling

    arch = args.model
    flag = None
    if args.pretrained:
        # check if model parameters match an available pretrained model
        if arch == 'resnet8' and units == 32:
            flag = 'resnet8_u32'
        elif arch == 'resnet8' and units == 64:
            flag = 'resnet8_u64'
        elif arch == 'resnet16' and units == 32:
            flag = 'resnet16_u32'
        elif arch == 'resnet16' and units == 64:
            flag = 'resnet16_u64'

    if flag is not None:
        report('Loading pretrained model:', flag)
        classifier = load_model(flag)
        classifier.train()
    else:
        feature_extractor = get_feature_extractor(args.model, units, dropout=dropout, bn=bn
                                                 , unit_scaling=unit_scaling, pooling=pooling)
        classifier = C.LinearClassifier(feature_extractor, dims=2, patch_size=args.patch_size, padding=args.patch_padding, batch_size=args.minibatch_size)

    ## if the method is generative, create the generative model as well
    generative = None
    if args.autoencoder > 0:
        ngf = args.ngf
        depth = int(np.log2(classifier.width+1)-3)
        generative = ConvGenerator(classifier.latent_dim, units=ngf, depth=depth)
        ## attach the generative model
        classifier.generative = generative
        report('Generator: units={}, size={}'.format(ngf, generative.width))

    report('Receptive field:', classifier.width)

    return classifier


def make_training_step_method(classifier, num_positive_regions, positive_fraction
                             , lr=1e-3, l2=0, method='GE-binomial', pi=0, slack=-1
                             , autoencoder=0):
    criteria = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam

    # pi sets the expected fraction of positives
    # but during training, we iterate over unlabeled data with labeled positives removed
    # therefore, we expected the fraction of positives in the unlabeled data
    # to be pi - fraction of labeled positives
    # if we are using the 'GE-KL' or 'GE-binomial' loss functions
    p_observed = positive_fraction
    if pi <= p_observed and method in ['GE-KL', 'GE-binomial']:
        # if pi <= p_observed, then we think the unlabeled data is all negatives
        # report this to the user and switch method to 'PN' if it isn't already
        print(f'WARNING: pi={pi} but the observed fraction of positives is {p_observed} and method is set to {method}.', file=sys.stderr) 
        print(f'WARNING: setting method to PN with pi={p_observed} instead.', file=sys.stderr)
        print(f'WARNING: if you meant to use {method}, please set pi > {p_observed}.', file=sys.stderr)
        pi = p_observed
        method = 'PN'
    elif method in ['GE-KL', 'GE-binomial']:
        pi = pi - p_observed

    split = 'pn'
    if method == 'PN':
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PN(classifier, optim, criteria, pi=pi, l2=l2, autoencoder=autoencoder)

    elif method == 'GE-KL':
        if slack < 0:
            slack = 10
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_KL(classifier, optim, criteria, pi, l2=l2, slack=slack)

    elif method == 'GE-binomial':
        if slack < 0:
            slack = 1
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_binomial(classifier, optim, criteria, pi, l2=l2, slack=slack, autoencoder=autoencoder)

    elif method == 'PU':
        split = 'pu'
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PU(classifier, optim, criteria, pi, l2=l2, autoencoder=autoencoder)

    else:
        raise Exception('Invalid method: ' + method)

    return trainer, criteria, split


def make_data_iterators(train_images:List[List[Union[Image.Image,np.ndarray]]], train_targets:List[List[np.ndarray]], 
                        test_images:List[List[Union[Image.Image,np.ndarray]]], test_targets:List[List[np.ndarray]], 
                        crop:int, split:Literal['pn','pu'], args, dims:int=2, to_tensor=True):
    ## training parameters
    minibatch_size = args.minibatch_size
    epoch_size = args.epoch_size
    num_epochs = args.num_epochs
    num_workers = mp.cpu_count() if args.num_workers < 0 else args.num_workers # set num workers to use all CPUs 
    testing_batch_size = args.test_batch_size
    balance = None if args.natural else args.minibatch_balance # ratio of positive to negative in minibatch
    report(f'minibatch_size={minibatch_size}, epoch_size={epoch_size}, num_epochs={num_epochs}')

    ## create augmented training dataset
    train_dataset = make_traindataset(train_images, train_targets, crop, dims=dims)
    test_dataset = SegmentedImageDataset(test_images, test_targets, to_tensor=to_tensor) if test_targets is not None else None

    ## create minibatch iterators
    labels = train_dataset.data.labels
    sampler = StratifiedCoordinateSampler(labels, size=epoch_size*minibatch_size, balance=balance, split=split)
    
    train_iterator = DataLoader(train_dataset, batch_size=minibatch_size, sampler=sampler, num_workers=num_workers)
    test_iterator = DataLoader(test_dataset, batch_size=testing_batch_size, num_workers=0) if test_dataset is not None else None
    return train_iterator, test_iterator


def evaluate_model(classifier, criteria, data_iterator, use_cuda=False):
    classifier.eval()
    classifier.fill()

    n = 0
    loss = 0
    scores = []
    Y_true = []

    with torch.no_grad():
        for X,Y in data_iterator:
            Y = Y.view(-1)
            Y_true.append(Y.cpu().numpy())
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()

            if classifier.dims == 2:
                score = classifier(X).view(-1)
            elif classifier.dims == 3:
                score = classify_patches(classifier, X, batch_size=data_iterator.batch_size, 
                                        patch_size=classifier.patch_size, padding=classifier.padding).view(-1)

            scores.append(score.data.cpu().numpy())
            this_loss = criteria(score, Y).item()

            n += Y.size(0)
            delta = Y.size(0)*(this_loss - loss)
            loss += delta/n

    scores = np.concatenate(scores, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    y_hat = 1.0/(1.0 + np.exp(-scores))
    precision = y_hat[Y_true == 1].sum()/y_hat.sum()
    tpr = y_hat[Y_true == 1].mean()
    fpr = y_hat[Y_true == 0].mean()
    
    auprc = average_precision(Y_true, scores)

    classifier.unfill()

    return loss, precision, tpr, fpr, auprc


def fit_epoch(step_method, data_iterator, epoch=1, it=1, use_cuda=False, output=sys.stdout):
    for X,Y in data_iterator:
        Y = Y.view(-1)
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        metrics = step_method.step(X, Y)
        line = '\t'.join([str(epoch), str(it), 'train'] + [str(metric) for metric in metrics] + ['-'])
        print(line, file=output)
        #output.flush()
        it += 1
    return it


def fit_epochs(classifier, criteria, step_method, train_iterator, test_iterator, num_epochs
              , save_prefix=None, use_cuda=False, output=sys.stdout):
    ## fit the model, report train/test stats, save model if required
    header = step_method.header
    line = '\t'.join(['epoch', 'iter', 'split'] + header + ['auprc'])
    print(line, file=output)

    it = 1
    for epoch in range(1,num_epochs+1):
        ## update the model
        classifier.train()
        it = fit_epoch(step_method, train_iterator, epoch=epoch, it=it, use_cuda=use_cuda, output=output)

        ## measure validation performance
        if test_iterator is not None:
            loss,precision,tpr,fpr,auprc = evaluate_model(classifier, criteria, test_iterator, use_cuda=use_cuda)
            line = '\t'.join([str(epoch), str(it), 'test', str(loss)] + ['-']*(len(header)-4) + [str(precision), str(tpr), str(fpr), str(auprc)])
            print(line, file=output)
            output.flush()

        ## save the model
        if save_prefix is not None:
            prefix = save_prefix
            digits = int(np.ceil(np.log10(num_epochs)))
            path = prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
            classifier.cpu()
            torch.save(classifier, path)
            if use_cuda:
                classifier.cuda()


def train_model(classifier, train_images, train_targets, test_images, test_targets, use_cuda, save_prefix, output, args, dims:int=2, to_tensor=True):
    num_positive_regions, total_regions = report_data_stats(train_images, train_targets, test_images, test_targets)

    ## make the training step method
    if args.num_particles > 0:
        num_micrographs = sum(len(images) for images in train_images)
        # expected particles in training set rather than per micrograph
        expected_num_particles = args.num_particles * num_micrographs
        
        pi = calculate_pi(expected_num_particles, args.radius, total_regions, dims)

        report('Specified expected number of particle per micrograph = {}'.format(args.num_particles))
        report('With radius = {}'.format(args.radius))
        report('Setting pi = {}'.format(pi))
    else: 
        pi = args.pi
        report('pi = {}'.format(pi))
    
    trainer, criteria, split = make_training_step_method(classifier, num_positive_regions,
                                                         num_positive_regions/total_regions,
                                                         lr=args.learning_rate, l2=args.l2,
                                                         method=args.method, pi=pi, slack=args.slack,
                                                         autoencoder=args.autoencoder)
    ## training parameters
    train_iterator,test_iterator = make_data_iterators(train_images, train_targets,
                                                       test_images, test_targets,
                                                       classifier.width, split, args, dims=dims, to_tensor=to_tensor)
    
    fit_epochs(classifier, criteria, trainer, train_iterator, test_iterator, args.num_epochs,
               save_prefix=save_prefix, use_cuda=use_cuda, output=output)

    return classifier