#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from topaz.utils.printing import report
from topaz.utils.data.loader import load_images_from_list
from topaz.utils.data.coordinates import match_coordinates_to_images

name = 'train'
help = 'train region classifier from images with labeled coordinates'

def add_arguments(parser):

    parser.add_argument('--train-images', help='path to file listing the training images')
    parser.add_argument('--train-targets', help='path to file listing the training particle coordinates')
    parser.add_argument('--test-images', help='path to file listing the test images, optional')
    parser.add_argument('--test-targets', help='path to file listing the testing particle coordinates, optional')

    ## optional format of the particle coordinates file
    parser.add_argument('--format', dest='format_', choices=['auto', 'coord', 'csv', 'star', 'box'], default='auto'
                       , help='file format of the INPUT file (default: detect format automatically based on file extension)')

    ## cross-validation k-fold split options
    parser.add_argument('-k', '--k-fold', default=0, type=int, help='option to split the training set into K folds for cross validation (default: not used)')
    parser.add_argument('--fold', default=0, type=int, help='when using K-fold cross validation, sets which fold is used as the heldout test set (default: 0)')
    parser.add_argument('--cross-validation-seed', default=42, type=int, help='random seed for partitioning data into folds (default: 42)')

    # training parameters
    parser.add_argument('--radius', default=0, type=int, help='pixel radius around particle centers to consider positive (default: 0)')

    parser.add_argument('-m', '--model', default='resnet8', help='model type to fit (default: resnet8)')
    parser.add_argument('--units', default=32, type=int, help='number of units model parameter (default: 32)')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate model parameter(default: 0.0)')
    parser.add_argument('--bn', default='on', choices=['on', 'off'], help='use batch norm in the model (default: on)')
    parser.add_argument('--pooling', help='pooling method to use (default: none)')
    parser.add_argument('--unit-scaling', default=2, type=int, help='scale the number of units up by this factor every pool/stride layer (default: 2)')
    parser.add_argument('--ngf', default=32, type=int, help='scaled number of units per layer in generative model if used (default: 32)')

    methods = ['PN', 'GE-KL', 'GE-binomial', 'PU']
    parser.add_argument('--method', choices=methods, default='GE-binomial', help='objective function to use for learning the region classifier (default: GE-binomial)')

    parser.add_argument('--autoencoder', default=0, type=float, help='option to augment method with autoencoder. weight on reconstruction error (default: 0)')

    parser.add_argument('--pi', type=float, help='parameter specifying fraction of data that is expected to be positive')
    parser.add_argument('--slack', default=-1, type=float, help='weight on GE penalty (default: 10 for GE-KL, 1 for GE-binomial)')


    parser.add_argument('--l2', default=0.0, type=float, help='l2 regularizer on the model parameters (default: 0)')

    parser.add_argument('--learning-rate', default=0.0002, type=float, help='learning rate for the optimizer (default: 0.0002)') 

    parser.add_argument('--natural', action='store_true', help='sample unbiasedly from the data to form minibatches rather than sampling particles and not particles at ratio given by minibatch-balance parameter')

    parser.add_argument('--minibatch-size', default=256, type=int, help='number of data points per minibatch (default: 256)')
    parser.add_argument('--minibatch-balance', default=0.0625, type=float, help='fraction of minibatch that is positive data points (default: 1/16)')
    parser.add_argument('--epoch-size', default=5000, type=int, help='number of parameter updates per epoch (default: 5000)')
    parser.add_argument('--num-epochs', default=10, type=int, help='maximum number of training epochs (default: 10)')

    parser.add_argument('--num-workers', default=0, type=int, help='number of worker processes for data augmentation (default: 0)')
    parser.add_argument('--test-batch-size', default=1, type=int, help='batch size for calculating test set statistics (default: 1)')


    ## device and output files
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')

    parser.add_argument('--save-prefix', help='path prefix to save trained models each epoch')
    parser.add_argument('-o', '--output', help='destination to write the train/test curve')

    ## only describe the model
    parser.add_argument('--describe', action='store_true', help='only prints a description of the model, does not train')

    return parser

def match_images_targets(images, targets, radius):
    matched = match_coordinates_to_images(targets, images, radius=radius)
    ## unzip into matched lists
    images = []
    targets = []
    for key in matched:
        these_images,these_targets = zip(*list(matched[key].values()))
        images.append(list(these_images))
        targets.append(list(these_targets))

    return images, targets

def make_traindataset(X, Y, crop):
    from topaz.utils.data.loader import LabeledImageCropDataset
    from topaz.utils.data.sampler import RandomImageTransforms
    
    size = int(np.ceil(crop*np.sqrt(2)))
    if size % 2 == 0:
        size += 1
    dataset = LabeledImageCropDataset(X, Y, size)
    transformed = RandomImageTransforms(dataset, crop=crop, to_tensor=True)

    return transformed


def make_trainiterator(dataset, minibatch_size, epoch_size, balance=0.5, num_workers=0):
    """ epoch_size in data points not minibatches """

    from topaz.utils.data.sampler import StratifiedCoordinateSampler
    from torch.utils.data.dataloader import DataLoader

    labels = dataset.labels
    sampler = StratifiedCoordinateSampler(labels, size=epoch_size, balance=balance)
    loader = DataLoader(dataset, batch_size=minibatchsize, sampler=sampler
                       , num_workers=num_workers)

    return loader


def make_testdataset(X, Y):
    from topaz.utils.data.loader import SegmentedImageDataset

    dataset = SegmentedImageDataset(X, Y, to_tensor=True)

    return dataset

def calculate_positive_fraction(targets):
    per_source = []
    for source_targets in targets:
        positives = sum(target.sum() for target in source_targets)
        total = sum(target.size for target in source_targets)
        per_source.append(positives/total)
    return np.mean(per_source)


def cross_validation_split(k, fold, images, targets, random=np.random):
    import topaz.utils.data.partition
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

def load_data(train_images, train_targets, test_images, test_targets, radius
             , k_fold=0, fold=0, cross_validation_seed=42, format_='auto'):

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
    train_images, train_targets = match_images_targets(train_images, train_targets, radius)
    
    if test_images is not None:
        test_images = pd.read_csv(test_images, sep='\t')
        #test_targets = pd.read_csv(test_targets, sep='\t')
        test_targets = file_utils.read_coordinates(test_targets, format=format_)
        # check for source columns
        if 'source' not in test_images and 'source' not in test_targets:
            test_images['source'] = 0
            test_targets['source'] = 0
        test_images = load_images_from_list(test_images.image_name, test_images.path
                                           , sources=test_images.source)
        test_images, test_targets = match_images_targets(test_images, test_targets, radius)
    elif k_fold > 1:
        ## seed for partitioning the data
        random = np.random.RandomState(cross_validation_seed)
        ## make the split
        train_images, train_targets, test_images, test_targets = cross_validation_split(k_fold, fold, train_images, train_targets, random=random)
       
    return train_images, train_targets, test_images, test_targets

def report_data_stats(train_images, train_targets, test_images, test_targets):
    report('source\tsplit\tp\ttotal')
    num_positive_regions = 0
    total_regions = 0
    for i in range(len(train_images)):
        p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
        total = sum(train_targets[i][j].size for j in range(len(train_targets[i])))
        num_positive_regions += p
        total_regions += total
        p = p/total
        report(str(i)+'\t'+'train'+'\t'+str(p)+'\t'+str(total))
        if test_targets is not None:
            p = sum(test_targets[i][j].sum() for j in range(len(test_targets[i])))
            total = sum(test_targets[i][j].size for j in range(len(test_targets[i])))
            p = p/total
            report(str(i)+'\t'+'test'+'\t'+str(p)+'\t'+str(total))
    return num_positive_regions, total_regions

def make_model(args):
    from topaz.model.factory import get_feature_extractor
    import topaz.model.classifier as C
    from topaz.model.classifier import LinearClassifier

    report('Loading model:', args.model)
    report('Model parameters: units={}, dropout={}, bn={}'.format(args.units, args.dropout, args.bn))
   
    # initialize the model 
    units = args.units
    dropout = args.dropout
    bn = args.bn == 'on'
    pooling = args.pooling
    unit_scaling = args.unit_scaling

    feature_extractor = get_feature_extractor(args.model, units, dropout=dropout, bn=bn
                                             , unit_scaling=unit_scaling, pooling=pooling)
    classifier = C.LinearClassifier(feature_extractor)

    ## if the method is generative, create the generative model as well
    generative = None
    if args.autoencoder > 0:
        from topaz.model.generative import ConvGenerator
        ngf = args.ngf
        depth = int(np.log2(classifier.width+1)-3)
        generative = ConvGenerator(classifier.latent_dim, units=ngf, depth=depth)
        ## attach the generative model
        classifier.generative = generative
        report('Generator: units={}, size={}'.format(ngf, generative.width))

    report('Receptive field:', classifier.width)

    return classifier

def make_training_step_method(classifier, num_positive_regions, positive_fraction, args):
    import topaz.methods as methods

    criteria = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam
    lr = args.learning_rate
    l2 = args.l2
    pi = args.pi
    slack = args.slack
    split = 'pn'

    if args.method == 'PN':
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PN(classifier, optim, criteria, pi=pi, l2=l2
                            , autoencoder=args.autoencoder)

    elif args.method == 'GE-KL':
        #split = 'pu'
        if slack < 0:
            slack = 10
        assert positive_fraction <= pi
        pi = pi - positive_fraction
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_KL(classifier, optim, criteria, pi, l2=l2, slack=slack)

    elif args.method == 'GE-binomial':
        #split = 'pu'
        if slack < 0:
            slack = 1
        assert positive_fraction <= pi
        pi = pi - positive_fraction
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_binomial(classifier, optim, criteria, pi
                                     , l2=l2, slack=slack
                                     , autoencoder=args.autoencoder
                                     )

    elif args.method == 'PU':
        split = 'pu'
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PU(classifier, optim, criteria, pi, l2=l2, autoencoder=args.autoencoder)

    else:
        raise Exception('Invalid method: ' + args.method)

    return trainer, criteria, split

def make_data_iterators(train_images, train_targets, test_images, test_targets
                       , crop, split, args):
    from topaz.utils.data.sampler import StratifiedCoordinateSampler
    from torch.utils.data.dataloader import DataLoader

    ## training parameters
    minibatch_size = args.minibatch_size
    epoch_size = args.epoch_size
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    testing_batch_size = args.test_batch_size
    balance = args.minibatch_balance # ratio of positive to negative in minibatch
    if args.natural:
        balance = None
    report('minibatch_size={}, epoch_size={}, num_epochs={}'.format(
           minibatch_size, epoch_size, num_epochs))

    ## create augmented training dataset
    train_dataset = make_traindataset(train_images, train_targets, crop)
    test_dataset = None
    if test_targets is not None:
        test_dataset = make_testdataset(test_images, test_targets)

    ## create minibatch iterators
    labels = train_dataset.data.labels
    sampler = StratifiedCoordinateSampler(labels, size=epoch_size*minibatch_size
                                         , balance=balance, split=split)
    train_iterator = DataLoader(train_dataset, batch_size=minibatch_size, sampler=sampler
                               , num_workers=num_workers)

    test_iterator = None
    if test_dataset is not None:
        test_iterator = DataLoader(test_dataset, batch_size=testing_batch_size, num_workers=0)

    return train_iterator, test_iterator


def evaluate_model(classifier, criteria, data_iterator, use_cuda=False):
    from topaz.metrics import average_precision

    classifier.eval()
    classifier.fill()

    n = 0
    loss = 0
    scores = []
    Y_true = []

    for X,Y in data_iterator:
        Y = Y.view(-1)
        Y_true.append(Y.numpy())
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        X = Variable(X, volatile=True)
        Y = Variable(Y, volatile=True)

        score = classifier(X).view(-1)

        scores.append(score.data.cpu().numpy())
        this_loss = criteria(score, Y).data[0]

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
        it = fit_epoch(step_method, train_iterator, epoch=epoch, it=it
                      , use_cuda=use_cuda, output=output)

        ## measure validation performance
        if test_iterator is not None:
            loss,precision,tpr,fpr,auprc = evaluate_model(classifier, criteria, test_iterator
                                                         , use_cuda=use_cuda)
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


def main(args):
    ## initialize the model
    classifier = make_model(args)

    if args.describe: 
        ## only print a description of the model and terminate
        print(classifier)
        sys.exit()

    ## set the device
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.device)
    report('Using device={} with cuda={}'.format(args.device, use_cuda))

    if use_cuda:
        classifier.cuda()
    
    ## load the data
    radius = args.radius # number of pixels around coordinates to label as positive
    train_images, train_targets, test_images, test_targets = \
            load_data(args.train_images,
                      args.train_targets,
                      args.test_images,
                      args.test_targets,
                      radius,
                      format_=args.format_,
                      k_fold=args.k_fold,
                      fold=args.fold,
                      cross_validation_seed=args.cross_validation_seed,
                     )
    num_positive_regions, total_regions = report_data_stats(train_images, train_targets
                                                           , test_images, test_targets)

    ## make the training step method
    trainer, criteria, split = make_training_step_method(classifier
                                                        , num_positive_regions
                                                        , num_positive_regions/total_regions
                                                        , args)

    ## training parameters
    train_iterator,test_iterator = make_data_iterators(train_images, train_targets,
                                                       test_images, test_targets,
                                                       classifier.width, split, args)
    
    ## fit the model, report train/test stats, save model if required
    output = sys.stdout if args.output is None else open(args.output, 'w')
    save_prefix = args.save_prefix
    #if not os.path.exists(os.path.dirname(save_prefix)):
    #    os.makedirs(os.path.dirname(save_prefix))
    fit_epochs(classifier, criteria, trainer, train_iterator, test_iterator, args.num_epochs
              , save_prefix=save_prefix, use_cuda=use_cuda, output=output)

    report('Done!')


if __name__ == '__main__':
    import argparse
    parser = ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)





