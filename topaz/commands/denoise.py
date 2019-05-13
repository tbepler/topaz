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

from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample
import topaz.mrc as mrc

name = 'denoise'
help = 'denoise micrographs using different denoising algorithms'

def add_arguments(parser):

    ## only describe the model
    # set GPU and number of worker threads
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')

    parser.add_argument('micrographs', nargs='*', help='micrographs to denoise')

    parser.add_argument('-o', '--output', help='directory to save denoised micrographs')
    parser.add_argument('--format', dest='format_', default='mrc', help='output format for the images (default: mrc)')
    parser.add_argument('--normalize', action='store_true', help='normalize the micrographs')

    parser.add_argument('--stack', action='store_true', help='denoise a MRC stack rather than list of micorgraphs')

    parser.add_argument('--save-prefix', help='path prefix to save denoising model')
    parser.add_argument('-m', '--model', default='L0', help='use pretrained denoising model, use L0, L1, or L2 for different pretrained models (default: L0)')

    parser.add_argument('-a', '--dir-a', help='directory of training images part A')
    parser.add_argument('-b', '--dir-b', help='directory of training images part B')

    parser.add_argument('--bin', type=int, default=1)
    parser.add_argument('-s', '--patch-size', type=int, default=-1, help='denoises micrographs in patches of this size. not used if <1 (default: -1)')

    parser.add_argument('-n', '--noise', default=1.0, type=float, help='standard deviation of the noise (default: 1.0)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--criteria', default='L2', choices=['L0', 'L1', 'L2'], help='training criteria (default: L2)')

    parser.add_argument('-c', '--crop', type=int, default=800, help='training crop size (default: 800)')
    parser.add_argument('--batch-size', type=int, default=10, help='training batch size (default: 10)')

    parser.add_argument('--num-epochs', default=100, type=int, help='number of training epochs (default: 100)') 

    return parser


import topaz.denoise as dn
from topaz.utils.image import save_image


def main(args):

    ## set the device
    use_cuda = False
    if args.device >= 0:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.device)
    print('# using device={} with cuda={}'.format(args.device, use_cuda), file=sys.stderr)

    if args.model is None:
        # train denoising model
        # make the dataset
        sigma = args.noise
        crop = args.crop
        #dataset = dn.GaussianNoise(images, sigma=sigma, crop=crop, xform=True)

        # get paired images
        dir_a = args.dir_a
        dir_b = args.dir_b

        A = []
        B = []
        for path in glob.glob(dir_a + os.sep + '*.mrc'):
            name = os.path.basename(path)
            A.append(path)
            B.append(dir_b + os.sep + name)
        print('# training with', len(A), 'image pairs', file=sys.stderr)

        dataset = dn.PairedImages(A, B, crop=crop, xform=True)

        # initialize the model
        #model = dn.DenoiseNet(32)
        model = dn.UDenoiseNet()
        if use_cuda:
            model = model.cuda()

        # train
        lr = args.lr
        batch_size = args.batch_size
        num_epochs = args.num_epochs

        num_workers = 8

        print('epoch', 'loss')
        #criteria = nn.L1Loss()
        criteria = args.criteria

        for epoch,loss in dn.train_noise2noise(model, dataset, lr=lr, batch_size=batch_size
                                              , criteria=criteria
                                              , num_epochs=num_epochs, use_cuda=use_cuda
                                              , num_workers=num_workers
                                              ):
            print(epoch, loss)

            # save the model
            if args.save_prefix is not None:
                path = args.save_prefix + '_epoch{}.sav'.format(epoch)
                model.cpu()
                model.eval()
                torch.save(model, path)
                if use_cuda:
                    model.cuda()

    else: # load the saved model
        if args.model in ['L0', 'L1', 'L2']:
            model = dn.load_model(args.model)
        else:
            model = torch.load(args.model)
        print('# using model:', args.model, file=sys.stderr)
        model.eval()
        if use_cuda:
            model.cuda()

    if args.stack:
        # we are denoising a single MRC stack
        with open(args.micrographs[0], 'rb') as f:
            content = f.read()
        stack,_,_ = mrc.parse(content)
        print('# denoising stack with shape:', stack.shape, file=sys.stderr)

        denoised = dn.denoise_stack(model, stack, use_cuda=use_cuda)

        # write the denoised stack
        path = args.output
        print('# writing', path, file=sys.stderr)
        with open(path, 'wb') as f:
            mrc.write(f, denoised)

    else:
        # using trained model
        # stream the micrographs and denoise as we go

        normalize = args.normalize
        if args.format_ == 'png':
            # always normalize png format
            normalize = True

        format_ = args.format_

        count = 0
        total = len(args.micrographs)

        bin_ = args.bin
        ps = args.patch_size

        # now, stream the micrographs and denoise them
        for path in args.micrographs:
            name,_ = os.path.splitext(os.path.basename(path))
            mic = np.array(load_image(path), copy=False)
            if bin_ > 1:
                mic = downsample(mic, bin_)
            mu = mic.mean()
            std = mic.std()

            # denoise
            mic = (mic - mu)/std
            mic = dn.denoise(model, mic, patch_size=ps, use_cuda=use_cuda)

            if normalize:
                mic = (mic - mic.mean())/mic.std()
            else:
                # add back std. dev. and mean
                mic = std*mic + mu

            # write the micrograph
            outpath = args.output + os.sep + name + '.' + format_
            save_image(mic, outpath)

            count += 1
            print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')

        print('', file=sys.stderr)



if __name__ == '__main__':
    import argparse
    parser = ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)





