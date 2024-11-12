#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import sys

import topaz.cuda
from topaz.training import load_data,make_model,train_model
from topaz.utils.printing import report


name = 'train'
help = 'train 2D region classifier from images with labeled coordinates'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(help)

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
    model.add_argument('-s', '--patch-size', type=int, default=96, help='classify micrographs in patches of this size. not used if < 1 (default: 96)')
    model.add_argument('-p', '--patch-padding', type=int, default=48, help='padding around each patch to remove edge artifacts (default: 48)')

    outputs = parser.add_argument_group('output file arguments (optional)')
    outputs.add_argument('--save-prefix', help='path prefix to save trained models each epoch')
    outputs.add_argument('-o', '--output', help='destination to write the train/test curve')


    misc = parser.add_argument_group('miscellaneous arguments (optional)')
    misc.add_argument('--test-batch-size', default=1, type=int, help='batch size for calculating test set statistics (default: 1)')

    return parser


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
    use_cuda = topaz.cuda.set_device(args.device)
    report('Using device={} with cuda={}'.format(args.device, use_cuda))
    if use_cuda:
        classifier.cuda()
        if args.num_workers != 0: 
            report('When using GPU to load data, we only load in this process. Setting num_workers = 0.')
            args.num_workers = 0
    
    ## fit the model, report train/test stats, save model if required
    output = sys.stdout if args.output is None else open(args.output, 'w')
    save_prefix = args.save_prefix

    report('Training...')
    classifier = train_model(classifier, args.train_images, args.train_targets, args.test_images, args.test_targets, 
                            use_cuda, save_prefix, output, args, dims=2)
    report('Done!')
    return classifier


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
