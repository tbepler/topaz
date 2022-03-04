#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
import glob

import numpy as np
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import topaz.utils.files as file_utils
from topaz.utils.printing import report
from topaz.utils.data.loader import load_images_from_list
from topaz.utils.data.coordinates import match_coordinates_to_images
import topaz.cuda

name = 'train'
help = 'train region classifier from images with labeled coordinates'

def add_arguments(parser):

    ## only describe the model
    parser.add_argument('--describe', action='store_true', help='only prints a description of the model, does not train')
    # set GPU and number of worker threads
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')
    parser.add_argument('--num-workers', default=0, type=int, help='number of worker processes for data augmentation, if set to <0, automatically uses all CPUs available (default: 0)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')

    # group arguments into sections

    data = parser.add_argument_group('training data arguments (required)')

    data.add_argument('--train-images', help='path to file listing the training images. also accepts directory path from which all images are loaded.')
    data.add_argument('--train-targets', help='path to file listing the training particle coordinates')



    data = parser.add_argument_group('test data arguments (optional)')

    data.add_argument('--test-images', help='path to file listing the test images. also accepts directory path from which all images are loaded.')
    data.add_argument('--test-targets', help='path to file listing the testing particle coordinates.')

    
    data = parser.add_argument_group('data format arguments (optional)')
    ## optional format of the particle coordinates file
    data.add_argument('--format', dest='format_', choices=['auto', 'coord', 'csv', 'star', 'box'], default='auto'
                       , help='file format of the particle coordinates file (default: detect format automatically based on file extension)')
    data.add_argument('--image-ext', default='', help='sets the image extension if loading images from directory. should include "." before the extension (e.g. .tiff). (default: find all extensions)')

    
    data = parser.add_argument_group('cross validation arguments (optional)')
    ## cross-validation k-fold split options
    data.add_argument('-k', '--k-fold', default=0, type=int, help='option to split the training set into K folds for cross validation (default: not used)')
    data.add_argument('--fold', default=0, type=int, help='when using K-fold cross validation, sets which fold is used as the heldout test set (default: 0)')
    data.add_argument('--cross-validation-seed', default=42, type=int, help='random seed for partitioning data into folds (default: 42)')


    training = parser.add_argument_group('training arguments (required)')
    training.add_argument('-n', '--num-particles', type=float, default=-1, help='instead of setting pi directly, pi can be set by giving the expected number of particles per micrograph (>0). either this parameter or pi must be set.')
    training.add_argument('--pi', type=float, help='parameter specifying fraction of data that is expected to be positive')

    
    training = parser.add_argument_group('training arguments (optional)')
    # training parameters
    training.add_argument('-r', '--radius', default=3, type=int, help='pixel radius around particle centers to consider positive (default: 3)')

    methods = ['PN', 'GE-KL', 'GE-binomial', 'PU']
    training.add_argument('--method', choices=methods, default='GE-binomial', help='objective function to use for learning the region classifier (default: GE-binomial)')
    training.add_argument('--slack', default=-1, type=float, help='weight on GE penalty (default: 10 for GE-KL, 1 for GE-binomial)')

    training.add_argument('--autoencoder', default=0, type=float, help='option to augment method with autoencoder. weight on reconstruction error (default: 0)')

    training.add_argument('--l2', default=0.0, type=float, help='l2 regularizer on the model parameters (default: 0)')

    training.add_argument('--learning-rate', default=0.0002, type=float, help='learning rate for the optimizer (default: 0.0002)') 

    training.add_argument('--natural', action='store_true', help='sample unbiasedly from the data to form minibatches rather than sampling particles and not particles at ratio given by minibatch-balance parameter')

    training.add_argument('--minibatch-size', default=256, type=int, help='number of data points per minibatch (default: 256)')
    training.add_argument('--minibatch-balance', default=0.0625, type=float, help='fraction of minibatch that is positive data points (default: 0.0625)')
    training.add_argument('--epoch-size', default=1000, type=int, help='number of parameter updates per epoch (default: 1000)')
    training.add_argument('--num-epochs', default=10, type=int, help='maximum number of training epochs (default: 10)')


    model = parser.add_argument_group('model arguments (optional)')

    model.add_argument('--pretrained', dest='pretrained', action='store_true', help='by default, topaz train will initialize model parameters from the pretrained parameters if a pretrained model with the same configuration is available (e.g. resnet8 with 64 units). disable this behaviour by setting the --no-pretrained flag')
    model.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    model.set_defaults(pretrained=True)

    model.add_argument('-m', '--model', default='resnet8', help='model type to fit (default: resnet8)')
    model.add_argument('--units', default=32, type=int, help='number of units model parameter (default: 32)')
    model.add_argument('--dropout', default=0.0, type=float, help='dropout rate model parameter(default: 0.0)')
    model.add_argument('--bn', default='on', choices=['on', 'off'], help='use batch norm in the model (default: on)')
    model.add_argument('--pooling', help='pooling method to use (default: none)')
    model.add_argument('--unit-scaling', default=2, type=int, help='scale the number of units up by this factor every pool/stride layer (default: 2)')
    model.add_argument('--ngf', default=32, type=int, help='scaled number of units per layer in generative model, only used if autoencoder > 0 (default: 32)')

    outputs = parser.add_argument_group('output file arguments (optional)')
    outputs.add_argument('--save-prefix', help='path prefix to save trained models each epoch')
    outputs.add_argument('-o', '--output', help='destination to write the train/test curve')


    misc = parser.add_argument_group('miscellaneous arguments (optional)')
    misc.add_argument('--test-batch-size', default=1, type=int, help='batch size for calculating test set statistics (default: 1)')

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
    loader = DataLoader(dataset, batch_size=minibatch_size, sampler=sampler
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
             , k_fold=0, fold=0, cross_validation_seed=42, format_='auto', image_ext=''):

    # if train_images is a directory path, map to all images in the directory
    if os.path.isdir(train_images):
        paths = glob.glob(train_images + os.sep + '*' + image_ext)
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
    if num_particles == 0:
        print('ERROR: no training particles specified. Check that micrograph names in the particles file match those in the micrographs file/directory.', file=sys.stderr)
        raise Exception('No training particles.')


    train_images, train_targets = match_images_targets(train_images, train_targets, radius)

    
    if test_images is not None:
        if os.path.isdir(test_images):
            paths = glob.glob(test_images + os.sep + '*' + image_ext)
            valid_paths = []
            image_names = []
            for path in paths:
                name = os.path.basename(path)
                name,ext = os.path.splitext(name)
                if ext in ['.mrc', '.tiff', '.png']:
                    image_names.append(name)
                    valid_paths.append(path)
            test_images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
        else:
            test_images = pd.read_csv(test_images, sep='\t')
        #test_targets = pd.read_csv(test_targets, sep='\t')
        test_targets = file_utils.read_coordinates(test_targets, format=format_)
        # check for source columns
        if 'source' not in test_images and 'source' not in test_targets:
            test_images['source'] = 0
            test_targets['source'] = 0
        test_images = load_images_from_list(test_images.image_name, test_images.path
                                           , sources=test_images.source)

        # discard coordinates for micrographs not in the set of images
        # and warn the user if any are discarded
        names = set()
        for k,d in test_images.items():
            for name in d.keys():
                names.add(name)
        check = test_targets.image_name.apply(lambda x: x in names)
        missing = test_targets.image_name.loc[~check].unique().tolist()
        if len(missing) > 0:
            print('WARNING: {} micrographs listed in the coordinates file are missing from the test images. Image names are listed below.'.format(len(missing)), file=sys.stderr)
            print('WARNING: missing micrographs are: {}'.format(missing), file=sys.stderr)
        test_targets = test_targets.loc[check]

        num_micrographs = sum(len(test_images[k]) for k in test_images.keys())
        num_particles = len(test_targets)
        report('Loaded {} test micrographs with {} labeled particles'.format(num_micrographs, num_particles))

        test_images, test_targets = match_images_targets(test_images, test_targets, radius)
    elif k_fold > 1:
        ## seed for partitioning the data
        random = np.random.RandomState(cross_validation_seed)
        ## make the split
        train_images, train_targets, test_images, test_targets = cross_validation_split(k_fold, fold, train_images, train_targets, random=random)

        n_train = sum(len(images) for images in train_images)
        n_test = sum(len(images) for images in test_images)
        report('Split into {} train and {} test micrographs'.format(n_train, n_test))

    return train_images, train_targets, test_images, test_targets

def report_data_stats(train_images, train_targets, test_images, test_targets):
    report('source\tsplit\tp_observed\tnum_positive_regions\ttotal_regions')
    num_positive_regions = 0
    total_regions = 0
    for i in range(len(train_images)):
        p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
        p = int(p)
        total = sum(train_targets[i][j].size for j in range(len(train_targets[i])))
        num_positive_regions += p
        total_regions += total
        p_observed = p/total
        p_observed = '{:.3g}'.format(p_observed)
        report(str(i)+'\t'+'train'+'\t'+p_observed+'\t'+str(p)+'\t'+str(total))
        if test_targets is not None:
            p = sum(test_targets[i][j].sum() for j in range(len(test_targets[i])))
            p = int(p)
            total = sum(test_targets[i][j].size for j in range(len(test_targets[i])))
            p_observed = p/total
            p_observed = '{:.3g}'.format(p_observed)
            report(str(i)+'\t'+'test'+'\t'+p_observed+'\t'+str(p)+'\t'+str(total))
    return num_positive_regions, total_regions

def make_model(args):
    from topaz.model.factory import get_feature_extractor
    import topaz.model.classifier as C
    from topaz.model.classifier import LinearClassifier

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
        from topaz.model.factory import load_model
        report('Loading pretrained model:', flag)
        classifier = load_model(flag)
        classifier.train()
    else:
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

def make_training_step_method(classifier, num_positive_regions, positive_fraction
                             , lr=1e-3, l2=0, method='GE-binomial', pi=0, slack=-1
                             , autoencoder=0):
    import topaz.methods as methods

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
        print('WARNING: pi={} but the observed fraction of positives is {} and method is set to {}.'.format(pi, p_observed, method)
             , file=sys.stderr) 
        print('WARNING: setting method to PN with pi={} instead.'.format(p_observed), file=sys.stderr)
        print('WARNING: if you meant to use {}, please set pi > {}.'.format(method, p_observed), file=sys.stderr)
        pi = p_observed
        method = 'PN'
    elif method in ['GE-KL', 'GE-binomial']:
        pi = pi - p_observed

    split = 'pn'
    if method == 'PN':
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PN(classifier, optim, criteria, pi=pi, l2=l2
                            , autoencoder=autoencoder)

    elif method == 'GE-KL':
        if slack < 0:
            slack = 10
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_KL(classifier, optim, criteria, pi, l2=l2, slack=slack)

    elif method == 'GE-binomial':
        if slack < 0:
            slack = 1
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_binomial(classifier, optim, criteria, pi
                                     , l2=l2, slack=slack
                                     , autoencoder=autoencoder
                                     )

    elif method == 'PU':
        split = 'pu'
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PU(classifier, optim, criteria, pi, l2=l2, autoencoder=autoencoder)

    else:
        raise Exception('Invalid method: ' + method)

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
    if num_workers < 0: # set num workers to use all CPUs
        num_workers = mp.cpu_count()

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

    with torch.no_grad():
        for X,Y in data_iterator:
            Y = Y.view(-1)
            Y_true.append(Y.numpy())
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()

            score = classifier(X).view(-1)

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
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    ## initialize the model
    classifier = make_model(args)

    if args.describe: 
        ## only print a description of the model and terminate
        print(classifier)
        sys.exit()

    ## set the device
    """
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.device)
        else:
            print('WARNING: you specified GPU (device={}) but no GPUs were detected. This may mean there is a mismatch between your system CUDA version and your pytorch CUDA version.'.format(args.device), file=sys.stderr)
    """

    use_cuda = topaz.cuda.set_device(args.device)
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
                      image_ext=args.image_ext
                     )
    num_positive_regions, total_regions = report_data_stats(train_images, train_targets
                                                           , test_images, test_targets)

    ## make the training step method
    if args.num_particles > 0:
        expected_num_particles = args.num_particles
        # make this expected particles in training set rather than per micrograph
        num_micrographs = sum(len(images) for images in train_images)
        expected_num_particles *= num_micrographs
        
        # given the expected number of particles and the radius
        # calculate what pi should be
        # pi = pixels_per_particle*expected_number_of_particles/pixels_in_dataset
        grid = np.linspace(-radius, radius, 2*radius+1)
        xx = np.zeros((2*radius+1, 2*radius+1)) + grid[:,np.newaxis]
        yy = np.zeros((2*radius+1, 2*radius+1)) + grid[np.newaxis]
        d2 = xx**2 + yy**2
        mask = (d2 <= radius**2).astype(int)
        pixels_per_particle = mask.sum()

        # total_regions is number of regions in the data
        pi = pixels_per_particle*expected_num_particles/total_regions

        report('Specified expected number of particle per micrograph = {}'.format(args.num_particles))
        report('With radius = {}'.format(radius))
        report('Setting pi = {}'.format(pi))
    else: 
        pi = args.pi
        report('pi = {}'.format(pi))

    trainer, criteria, split = make_training_step_method(classifier
                                                        , num_positive_regions
                                                        , num_positive_regions/total_regions
                                                        , lr=args.learning_rate
                                                        , l2=args.l2
                                                        , method=args.method
                                                        , pi=pi
                                                        , slack=args.slack
                                                        , autoencoder=args.autoencoder
                                                        )

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
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)





