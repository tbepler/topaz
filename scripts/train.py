#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def report(*args):
    print('#', *args, file=sys.stderr)

def load_images(table):
    from topaz.utils.data.loader import load_mrc, load_pil
    images = {}
    for source,name,path in zip(table.source, table.image_name, table.path):
        ext = os.path.splitext(path)[1]
        if ext == 'mrc':
            image = load_mrc(path)
        else:
            image = load_pil(path)
        images.setdefault(source, {})[name] = image
    return images

def match_images_targets(images, targets, radius):
    from topaz.utils.picks import as_mask
    
    image_list = []
    target_list = []
    
    for source,source_images in images.items():
        source_image_list = []
        source_target_list = []
        for name,image in source_images.items():
            these_targets = targets.loc[(targets.source == source) & (targets.image_name == name)]
            xcoord = these_targets.x_coord.values.astype(np.int32)
            ycoord = these_targets.y_coord.values.astype(np.int32)
            radii = np.array([radius]*len(xcoord), dtype=np.int32)

            target_mask = as_mask((image.height, image.width), xcoord, ycoord, radii)

            source_image_list.append(image)
            source_target_list.append(target_mask)

        image_list.append(source_image_list)
        target_list.append(source_target_list)

    return image_list, target_list


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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-images', help='path to file listing the training images')
    parser.add_argument('--train-targets', help='path to file listing the training particle coordinates')
    parser.add_argument('--test-images', help='path to file listing the test images, optional')
    parser.add_argument('--test-targets', help='path to file listing the testing particle coordinates, optional')

    ## cross-validation k-fold split options
    parser.add_argument('-k', '--k-fold', default=0, type=int, help='option to split the training set into K folds for cross validation (default: not used)')
    parser.add_argument('--fold', default=0, type=int, help='when using K-fold cross validation, sets which fold is used as the heldout test set (default: 0)')
    parser.add_argument('--cross-validation-seed', default=42, type=int, help='random seed for partitioning data into folds (default: 42)')

    parser.add_argument('--radius', default=0, type=int, help='pixel radius around particle centers to consider positive (default: 0)')

    parser.add_argument('-m', '--model', default='resnet8', help='model type to fit (default: resnet8)')
    parser.add_argument('--units', default=32, type=int, help='number of units model parameter (default: 32)')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate model parameter(default: 0.0)')
    parser.add_argument('--bn', default='on', choices=['on', 'off'], help='use batch norm in the model (default: on)')
    parser.add_argument('--pooling', help='pooling method to use (default: none)')
    parser.add_argument('--unit-scaling', default=1, type=int, help='scale the number of units up by this factor every layer (default: 1)')
    parser.add_argument('--ngf', default=32, type=int, help='scaled number of units per layer in generative model if used (default: 32)')

    methods = ['PN', 'GE-KL', 'GE-binomial', 'PU']
    parser.add_argument('--method', choices=methods, default='GE-binomial', help='objective function to use for learning the region classifier (default: GE-binomial)')

    parser.add_argument('--autoencoder', default=0, type=float, help='option to augment method with autoencoder. weight on reconstruction error (default: 0)')

    parser.add_argument('--pi', type=float, help='parameter specifying fraction of data that is expected to be positive')
    parser.add_argument('--slack', default=-1, type=float, help='weight on GE penalty (default: 10 x number of particles for GE-KL, 1 for GE-binomial)')


    parser.add_argument('--l2', default=0.0, type=float, help='l2 regularizer on the model parameters (default: 0)')

    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)') 

    parser.add_argument('--natural', action='store_true', help='sample unbiasedly from the data to form minibatches rather than sampling particles and not particles at ratio given by minibatch-balance parameter')

    parser.add_argument('--minibatch-size', default=256, type=int, help='number of data points per minibatch (default: 256)')
    parser.add_argument('--minibatch-balance', default=0.0625, type=float, help='fraction of minibatch that is positive data points (default: 1/16)')
    parser.add_argument('--epoch-size', default=5000, type=int, help='number of parameter updates per epoch (default: 5000)')
    parser.add_argument('--num-epochs', default=10, type=int, help='maximum number of training epochs (default: 10)')

    parser.add_argument('--num-workers', default=0, type=int, help='number of worker processes for data augmentation (default: 0)')
    parser.add_argument('--test-batch-size', default=1, type=int, help='batch size for calculating test set statistics (default: 1)')

    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')

    parser.add_argument('--save-prefix', help='path prefix to save trained models each epoch')
    parser.add_argument('--output', help='destination to write the train/test curve')

    parser.add_argument('--describe', action='store_true', help='only prints a description of the model, does not train')

    return parser.parse_args()

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


if __name__ == '__main__':
    args = parse_args()

    ## initialize the model and get the training step method
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

    if args.describe: 
        ## only print a description of the model and terminate
        print(classifier)
        sys.exit()

    ## set the device
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        torch.cuda.set_device(args.device)
    report('Using device={} with cuda={}'.format(args.device, use_cuda))

    if use_cuda:
        classifier.cuda()
    
    # get the misclassification loss function
    criteria = nn.BCEWithLogitsLoss()

    # data parameters
    radius = args.radius # number of pixels around coordinates to label as positive

    ## load the data
    train_images = pd.read_csv(args.train_images, sep='\t') # training image file list
    train_targets = pd.read_csv(args.train_targets, sep='\t') # training particle coordinates file
    # check for source columns
    if 'source' not in train_images and 'source' not in train_targets:
        train_images['source'] = 0
        train_targets['source'] = 0
    # load the images and create target masks from the particle coordinates
    train_images = load_images(train_images)
    train_images, train_targets = match_images_targets(train_images, train_targets, radius)
    
    test_images = test_targets = None
    if args.test_images is not None:
        test_images = pd.read_csv(args.test_images, sep='\t')
        test_targets = pd.read_csv(args.test_targets, sep='\t')
        # check for source columns
        if 'source' not in test_images and 'source' not in test_targets:
            test_images['source'] = 0
            test_targets['source'] = 0
        test_images = load_images(test_images)
        test_images, test_targets = match_images_targets(test_images, test_targets, radius)
    elif args.k_fold > 1:
        k = args.k_fold
        fold = args.fold # which fold is being used as train/test
        ## seed for partitioning the data
        random = np.random.RandomState(args.cross_validation_seed)
        ## make the split
        train_images, train_targets, test_images, test_targets = cross_validation_split(k, fold, train_images, train_targets, random=random)
        

    report('source\tsplit\tp\ttotal')
    num_positive_regions = 0
    for i in range(len(train_images)):
        p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
        total = sum(train_targets[i][j].size for j in range(len(train_targets[i])))
        num_positive_regions += total
        p = p/total
        report(str(i)+'\t'+'train'+'\t'+str(p)+'\t'+str(total))
        if test_targets is not None:
            p = sum(test_targets[i][j].sum() for j in range(len(test_targets[i])))
            total = sum(test_targets[i][j].size for j in range(len(test_targets[i])))
            p = p/total
            report(str(i)+'\t'+'test'+'\t'+str(p)+'\t'+str(total))


    ## make the training step method
    optim = torch.optim.Adam
    lr = args.learning_rate
    l2 = args.l2
    pi = args.pi
    slack = args.slack
    split = 'pn'

    import topaz.methods as methods
    if args.method == 'PN':
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PN(classifier, optim, criteria, pi=pi, l2=l2
                            , autoencoder=args.autoencoder)
    elif args.method == 'GE-KL':
        #split = 'pu'
        if slack < 0:
            slack = 10*num_positive_regions
        momentum = 1 # args.ge_momentum
        positive_fraction = calculate_positive_fraction(train_targets)
        assert positive_fraction <= pi
        pi = pi - positive_fraction
        entropy_penalty = 0 # args.ge_entropy_penalty
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_KL(classifier, optim, criteria, pi, l2=l2, slack=slack, momentum=momentum
                                #, labeled_fraction=positive_fraction
                                , entropy_penalty=entropy_penalty)
    elif args.method == 'GE-binomial':
        #split = 'pu'
        if slack < 0:
            slack = 1
        positive_fraction = calculate_positive_fraction(train_targets)
        assert positive_fraction <= pi
        pi = pi - positive_fraction
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_binomial(classifier, optim, criteria, pi
                                     , l2=l2, slack=slack
                                     #, entropy_penalty=args.ge_entropy_penalty
                                     #, posterior_L1 = args.posterior_l1
                                     , autoencoder=args.autoencoder
                                     #, labeled_fraction=positive_fraction
                                     )
    elif args.method == 'PU':
        split = 'pu'
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PU(classifier, optim, criteria, pi, l2=l2, autoencoder=args.autoencoder)
    else:
        raise Exception('Invalid method: ' + args.method)


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
    crop = classifier.width
    train_dataset = make_traindataset(train_images, train_targets, crop)
    test_dataset = None
    if test_targets is not None:
        test_dataset = make_testdataset(test_images, test_targets)

    ## create minibatch iterators
    from topaz.utils.data.sampler import StratifiedCoordinateSampler
    from torch.utils.data.dataloader import DataLoader

    labels = train_dataset.data.labels
    sampler = StratifiedCoordinateSampler(labels, size=epoch_size*minibatch_size
                                         , balance=balance, split=split)
    train_iterator = DataLoader(train_dataset, batch_size=minibatch_size, sampler=sampler
                               , num_workers=num_workers)

    test_iterator = None
    if test_dataset is not None:
        test_iterator = DataLoader(test_dataset, batch_size=testing_batch_size, num_workers=0)

    
    ## fit the model, report train/test stats, save model if required
    from topaz.metrics import average_precision

    output = sys.stdout if args.output is None else open(args.output, 'w')

    header = trainer.header
    #line = '\t'.join(['epoch', 'iter', 'split'] + header + ['precision', 'tpr', 'fpr', 'auprc'])
    line = '\t'.join(['epoch', 'iter', 'split'] + header + ['auprc'])
    print(line, file=output)

    it = 1
    for epoch in range(1,num_epochs+1):
        
        ## update the model
        classifier.train()
        for X,Y in train_iterator:
            Y = Y.view(-1)
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
            
            metrics = trainer.step(X, Y)

            line = '\t'.join([str(epoch), str(it), 'train'] + [str(metric) for metric in metrics] + ['-'])
            print(line, file=output)
            #output.flush()

            it += 1

        if test_iterator is not None:
            ## measure performance on test set
            classifier.eval()
            classifier.fill()
        
            n = 0
            loss = 0
            scores = []
            Y_true = []

            for X,Y in test_iterator:
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

            del X
            del Y
            del score

            scores = np.concatenate(scores, axis=0)
            Y_true = np.concatenate(Y_true, axis=0)

            y_hat = 1.0/(1.0 + np.exp(-scores))
            precision = y_hat[Y_true == 1].sum()/y_hat.sum()
            tpr = y_hat[Y_true == 1].mean()
            fpr = y_hat[Y_true == 0].mean()
            
            auprc = average_precision(Y_true, scores)

            line = '\t'.join([str(epoch), str(it), 'test', str(loss)] + ['-']*(len(header)-4) + [str(precision), str(tpr), str(fpr), str(auprc)])
            print(line, file=output)
            output.flush()

            classifier.unfill()

            ## save the model
            if args.save_prefix is not None:
                prefix = args.save_prefix
                digits = int(np.ceil(np.log10(num_epochs)))
                path = prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
                classifier.cpu()
                torch.save(classifier, path)
                if use_cuda:
                    classifier.cuda()



    report('Done!')


