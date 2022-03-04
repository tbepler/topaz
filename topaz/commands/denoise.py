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
import topaz.cuda

name = 'denoise'
help = 'denoise micrographs with various denoising algorithms'

def add_arguments(parser):

    ## only describe the model
    # set GPU and number of worker threads
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')

    parser.add_argument('micrographs', nargs='*', help='micrographs to denoise')

    parser.add_argument('-o', '--output', help='directory to save denoised micrographs')
    parser.add_argument('--suffix', default='', help='add this suffix to each output file name. if no output directory is specified, denoised micrographs are written to the same location as the input with a default suffix of ".denoised" (default: none)')
    parser.add_argument('--format', dest='format_', default='mrc', help='output format for the images (default: mrc)')
    parser.add_argument('--normalize', action='store_true', help='normalize the micrographs')

    parser.add_argument('--stack', action='store_true', help='denoise a MRC stack rather than list of micorgraphs')

    parser.add_argument('--save-prefix', help='path prefix to save denoising model')
    parser.add_argument('-m', '--model', nargs='+', default=['unet'], help='use pretrained denoising model(s). can accept arguments for multiple models the outputs of which will be averaged. pretrained model options are: unet, unet-small, fcnn, affine. to use older unet version specify unet-v0.2.1 (default: unet)')

    parser.add_argument('-a', '--dir-a', nargs='+', help='directory of training images part A')
    parser.add_argument('-b', '--dir-b', nargs='+', help='directory of training images part B')
    parser.add_argument('--hdf', help='path to HDF5 file containing training image stack as an alternative to dirA/dirB')
    parser.add_argument('--preload', action='store_true', help='preload micrographs into RAM')
    parser.add_argument('--holdout', type=float, default=0.1, help='fraction of training micrograph pairs to holdout for validation (default: 0.1)')

    parser.add_argument('--lowpass', type=float, default=1, help='lowpass filter micrographs by this amount (in pixels) before applying the denoising filter. uses a hard lowpass filter (i.e. sinc) (default: no lowpass filtering)')
    parser.add_argument('--gaussian', type=float, default=0, help='Gaussian filter micrographs with this standard deviation (in pixels) before applying the denoising filter (default: 0)')
    parser.add_argument('--inv-gaussian', type=float, default=0, help='Inverse Gaussian filter micrographs with this standard deviation (in pixels) before applying the denoising filter (default: 0)')

    parser.add_argument('--deconvolve', action='store_true', help='apply optimal Gaussian deconvolution filter to each micrograph before denoising')
    parser.add_argument('--deconv-patch', type=int, default=1, help='apply spatial covariance correction to micrograph to this many patches (default: 1)')

    parser.add_argument('--pixel-cutoff', type=float, default=0, help='set pixels >= this number of standard deviations away from the mean to the mean. only used when set > 0 (default: 0)')
    parser.add_argument('-s', '--patch-size', type=int, default=1024, help='denoises micrographs in patches of this size. not used if < 1 (default: 1024)')
    parser.add_argument('-p', '--patch-padding', type=int, default=500, help='padding around each patch to remove edge artifacts (default: 500)')

    parser.add_argument('--method', choices=['noise2noise', 'masked'], default='noise2noise', help='denoising training method (default: noise2noise)')
    parser.add_argument('--arch', choices=['unet', 'unet-small', 'unet2', 'unet3', 'fcnet', 'fcnet2', 'affine'], default='unet', help='denoising model architecture (default: unet)')


    parser.add_argument('--optim', choices=['adam', 'adagrad', 'sgd'], default='adagrad', help='optimizer (default: adagrad)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--criteria', default='L2', choices=['L0', 'L1', 'L2'], help='training criteria (default: L2)')

    parser.add_argument('-c', '--crop', type=int, default=800, help='training crop size (default: 800)')
    parser.add_argument('--batch-size', type=int, default=4, help='training batch size (default: 4)')

    parser.add_argument('--num-epochs', default=100, type=int, help='number of training epochs (default: 100)') 

    parser.add_argument('--num-workers', default=16, type=int, help='number of threads to use for loading data during training (default: 16)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')

    return parser


import topaz.denoise as dn
from topaz.utils.image import save_image


def make_paired_images_datasets(dir_a, dir_b, crop, random=np.random, holdout=0.1, preload=False, cutoff=0):
    # train denoising model
    # make the dataset
    A = []
    B = []
    for path in glob.glob(dir_a + os.sep + '*.mrc'):
        name = os.path.basename(path)
        A.append(path)
        B.append(dir_b + os.sep + name)

    # randomly hold out some image pairs for validation
    n = int(holdout*len(A))
    order = random.permutation(len(A))

    A_train = []
    A_val = []
    B_train = []
    B_val = []
    for i in range(n):
        A_val.append(A[order[i]])
        B_val.append(B[order[i]])
    for i in range(n, len(A)):
        A_train.append(A[order[i]])
        B_train.append(B[order[i]])

    print('# training with', len(A_train), 'image pairs', file=sys.stderr)
    print('# validating on', len(A_val), 'image pairs', file=sys.stderr)

    dataset_train = dn.PairedImages(A_train, B_train, crop=crop, xform=True, preload=preload, cutoff=cutoff)
    dataset_val = dn.PairedImages(A_val, B_val, crop=crop, preload=preload, cutoff=cutoff)

    return dataset_train, dataset_val


def make_images_datasets(dir_a, dir_b, crop, random=np.random, holdout=0.1, cutoff=0):
    # train denoising model
    # make the dataset
    paths = []
    for path in glob.glob(dir_a + os.sep + '*.mrc'):
        paths.append(path)

    if dir_b is not None:
        for path in glob.glob(dir_b + os.sep + '*.mrc'):
            paths.append(path)

    # randomly hold out some image pairs for validation
    n = int(holdout*len(paths))
    order = random.permutation(len(paths))

    path_train = []
    path_val = []
    for i in range(n):
        path_val.append(paths[order[i]])
    for i in range(n, len(paths)):
        path_train.append(paths[order[i]])

    print('# training with', len(path_train), 'image pairs', file=sys.stderr)
    print('# validating on', len(path_val), 'image pairs', file=sys.stderr)

    dataset_train = dn.NoiseImages(path_train, crop=crop, xform=True, cutoff=cutoff)
    dataset_val = dn.NoiseImages(path_val, crop=crop, cutoff=cutoff)

    return dataset_train, dataset_val


class HDFPairedDataset:
    def __init__(self, dataset, start=0, end=None, xform=False, cutoff=0):
        self.dataset = dataset
        self.start = start
        self.end = end
        if end is None:
            self.end = len(dataset)
        self.n = (self.end - self.start)//2
        self.xform = xform
        self.cutoff = cutoff

    def __len__(self):
        return self.n

    def __getitem__(self, i): # retrieve the i'th image pair
        i = self.start + i*2
        j = i + 1

        x = self.dataset[i]
        y = self.dataset[j]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
                y = np.flip(y, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)
                y = np.flip(y, 1)


            k = np.random.randint(4)
            x = np.rot90(x, k=k)
            y = np.rot90(y, k=k)

            # swap x and y
            if np.random.rand() > 0.5:
                t = x
                x = y
                y = t

            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
            y[(y < -self.cutoff) | (y > self.cutoff)] = 0

        return x,y


class HDFDataset:
    def __init__(self, dataset, start=0, end=None, xform=False, cutoff=0):
        self.dataset = dataset
        self.start = start
        self.end = end
        if end is None:
            self.end = len(dataset)
        self.n = self.end - self.start
        self.xform = xform
        self.cutoff = cutoff

    def __len__(self):
        return self.n

    def __getitem__(self, i): # retrieve the i'th image pair
        i = self.start + i
        x = self.dataset[i]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)


            k = np.random.randint(4)
            x = np.rot90(x, k=k)

            x = np.ascontiguousarray(x)

        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0

        return x


def make_hdf5_datasets(path, paired=True, preload=False, holdout=0.1, cutoff=0):

    # open the hdf5 dataset
    import h5py
    f = h5py.File(path, 'r')
    dataset = f['images']
    if preload:
        dataset = dataset[:]

    # split into train/validate
    N = len(dataset) # number of image pairs
    if paired:
        N = N//2
    n = int(holdout*N)
    split = 2*(N-n)

    if paired:
        dataset_train = HDFPairedDataset(dataset, end=split, xform=True, cutoff=cutoff)
        dataset_val = HDFPairedDataset(dataset, start=split, cutoff=cutoff)
    else:
        dataset_train = HDFDataset(dataset, end=split, xform=True, cutoff=cutoff)
        dataset_val = HDFDataset(dataset, start=split, cutoff=cutoff)

    print('# training with', len(dataset_train), 'image pairs', file=sys.stderr)
    print('# validating on', len(dataset_val), 'image pairs', file=sys.stderr)

    return dataset_train, dataset_val


def denoise_image(mic, models, lowpass=1, cutoff=0, gaus=None, inv_gaus=None, deconvolve=False
                 , deconv_patch=1, patch_size=-1, padding=0, normalize=False
                 , use_cuda=False):
    if lowpass > 1:
        mic = dn.lowpass(mic, lowpass)

    mic = torch.from_numpy(mic)
    if use_cuda:
        mic = mic.cuda()

    # normalize and remove outliers
    mu = mic.mean()
    std = mic.std()
    x = (mic - mu)/std
    if cutoff > 0:
        x[(x < -cutoff) | (x > cutoff)] = 0

    # apply guassian/inverse gaussian filter
    if gaus is not None:
        x = dn.denoise(gaus, x)
    elif inv_gaus is not None:
        x = dn.denoise(inv_gaus, x)
    elif deconvolve:
        # estimate optimal filter and correct spatial correlation
        x = dn.correct_spatial_covariance(x, patch=deconv_patch)

    # denoise
    mic = 0
    for model in models:
        mic += dn.denoise(model, x, patch_size=patch_size, padding=padding)
    mic /= len(models)

    # restore pixel scaling
    if normalize:
        mic = (mic - mic.mean())/mic.std()
    else:
        # add back std. dev. and mean
        mic = std*mic + mu

    # back to numpy/cpu
    mic = mic.cpu().numpy()

    return mic


def main(args):

    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    ## set the device
    use_cuda = topaz.cuda.set_device(args.device)
    print('# using device={} with cuda={}'.format(args.device, use_cuda), file=sys.stderr)

    cutoff = args.pixel_cutoff # pixel truncation limit

    do_train = (args.dir_a is not None and args.dir_b is not None) or (args.hdf is not None)
    if do_train:

        method = args.method
        paired = (method == 'noise2noise')
        preload = args.preload
        holdout = args.holdout # fraction of image pairs to holdout for validation

        if args.hdf is None: #use dirA/dirB
            crop = args.crop
            dir_as = args.dir_a
            dir_bs = args.dir_b

            dset_train = []
            dset_val = []

            for dir_a, dir_b in zip(dir_as, dir_bs): 
                random = np.random.RandomState(44444)
                if paired:
                    dataset_train, dataset_val = make_paired_images_datasets(dir_a, dir_b, crop
                                                                            , random=random
                                                                            , holdout=holdout
                                                                            , preload=preload 
                                                                            , cutoff=cutoff
                                                                            )
                else:
                    dataset_train, dataset_val = make_images_datasets(dir_a, dir_b, crop
                                                                     , cutoff=cutoff
                                                                     , random=random
                                                                     , holdout=holdout)
                dset_train.append(dataset_train)
                dset_val.append(dataset_val)

            dataset_train = dset_train[0]
            for i in range(1, len(dset_train)):
                dataset_train.x += dset_train[i].x
                if paired:
                    dataset_train.y += dset_train[i].y

            dataset_val = dset_val[0]
            for i in range(1, len(dset_val)):
                dataset_val.x += dset_val[i].x
                if paired:
                    dataset_val.y += dset_val[i].y

            shuffle = True
        else: # make HDF datasets
            dataset_train, dataset_val = make_hdf5_datasets(args.hdf, paired=paired
                                                           , cutoff=cutoff
                                                           , holdout=holdout
                                                           , preload=preload)
            shuffle = preload

        # initialize the model
        arch = args.arch
        if arch == 'unet':
            model = dn.UDenoiseNet()
        elif arch == 'unet-small':
            model = dn.UDenoiseNetSmall()
        elif arch == 'unet2':
            model = dn.UDenoiseNet2()
        elif arch == 'unet3':
            model = dn.UDenoiseNet3()
        elif arch == 'fcnet':
            model = dn.DenoiseNet(32)
        elif arch == 'fcnet2':
            model = dn.DenoiseNet2(64)
        elif arch == 'affine':
            model = dn.AffineDenoise()
        else:
            raise Exception('Unknown architecture: ' + arch)

        if use_cuda:
            model = model.cuda()

        # train
        optim = args.optim
        lr = args.lr
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        digits = int(np.ceil(np.log10(num_epochs)))

        num_workers = args.num_workers

        print('epoch', 'loss_train', 'loss_val')
        #criteria = nn.L1Loss()
        criteria = args.criteria
        

        if method == 'noise2noise':
            iterator = dn.train_noise2noise(model, dataset_train, lr=lr
                                           , optim=optim
                                           , batch_size=batch_size
                                           , criteria=criteria
                                           , num_epochs=num_epochs
                                           , dataset_val=dataset_val
                                           , use_cuda=use_cuda
                                           , num_workers=num_workers
                                           , shuffle=shuffle
                                           )
        elif method == 'masked':
            iterator = dn.train_mask_denoise(model, dataset_train, lr=lr
                                            , optim=optim
                                            , batch_size=batch_size
                                            , criteria=criteria
                                            , num_epochs=num_epochs
                                            , dataset_val=dataset_val
                                            , use_cuda=use_cuda
                                            , num_workers=num_workers
                                            , shuffle=shuffle
                                            )



        for epoch,loss_train,loss_val in iterator:
            print(epoch, loss_train, loss_val)
            sys.stdout.flush()

            # save the model
            if args.save_prefix is not None:
                path = args.save_prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
                #path = args.save_prefix + '_epoch{}.sav'.format(epoch)
                model.cpu()
                model.eval()
                torch.save(model, path)
                if use_cuda:
                    model.cuda()
                    
        models = [model]

    else: # load the saved model(s)
        models = []
        for arg in args.model:
            if arg == 'none':
                print('# Warning: no denoising model will be used', file=sys.stderr)
            else:
                print('# Loading model:', arg, file=sys.stderr)
            model = dn.load_model(arg)

            model.eval()
            if use_cuda:
                model.cuda()

            models.append(model)

    # using trained model
    # denoise the images

    normalize = args.normalize
    if args.format_ == 'png' or args.format_ == 'jpg':
        # always normalize png and jpg format
        normalize = True

    format_ = args.format_
    suffix = args.suffix

    lowpass = args.lowpass
    gaus = args.gaussian
    if gaus > 0:
        gaus = dn.GaussianDenoise(gaus)
        if use_cuda:
            gaus.cuda()
    else:
        gaus = None
    inv_gaus = args.inv_gaussian
    if inv_gaus > 0:
        inv_gaus = dn.InvGaussianFilter(inv_gaus)
        if use_cuda:
            inv_gaus.cuda()
    else:
        inv_gaus = None
    deconvolve = args.deconvolve
    deconv_patch = args.deconv_patch

    ps = args.patch_size
    padding = args.patch_padding

    count = 0

    # we are denoising a single MRC stack
    if args.stack:
        with open(args.micrographs[0], 'rb') as f:
            content = f.read()
        stack,_,_ = mrc.parse(content)
        print('# denoising stack with shape:', stack.shape, file=sys.stderr)
        total = len(stack)

        denoised = np.zeros_like(stack)
        for i in range(len(stack)):
            mic = stack[i]
            # process and denoise the micrograph
            mic = denoise_image(mic, models, lowpass=lowpass, cutoff=cutoff, gaus=gaus
                               , inv_gaus=inv_gaus, deconvolve=deconvolve
                               , deconv_patch=deconv_patch
                               , patch_size=ps, padding=padding, normalize=normalize
                               , use_cuda=use_cuda
                               )
            denoised[i] = mic

            count += 1
            print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')

        print('', file=sys.stderr)
        # write the denoised stack
        path = args.output
        print('# writing', path, file=sys.stderr)
        with open(path, 'wb') as f:
            mrc.write(f, denoised)
    
    else:
        # stream the micrographs and denoise them
        total = len(args.micrographs)

        # make the output directory if it doesn't exist
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        for path in args.micrographs:
            name,_ = os.path.splitext(os.path.basename(path))
            mic = np.array(load_image(path), copy=False).astype(np.float32)

            # process and denoise the micrograph
            mic = denoise_image(mic, models, lowpass=lowpass, cutoff=cutoff, gaus=gaus
                               , inv_gaus=inv_gaus, deconvolve=deconvolve
                               , deconv_patch=deconv_patch
                               , patch_size=ps, padding=padding, normalize=normalize
                               , use_cuda=use_cuda
                               )

            # write the micrograph
            if not args.output:
                if suffix == '' or suffix is None:
                    suffix = '.denoised'
                # write the file to the same location as input
                no_ext,ext = os.path.splitext(path)
                outpath = no_ext + suffix + '.' + format_
            else:
                outpath = args.output + os.sep + name + suffix + '.' + format_
            save_image(mic, outpath) #, mi=None, ma=None)

            count += 1
            print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')
        print('', file=sys.stderr)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)





