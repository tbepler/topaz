#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import sys

import numpy as np
import topaz.denoise as dn
from topaz.cuda import set_device
from topaz.denoise import Denoise, denoise_stack, denoise_stream
from topaz.denoising.datasets import (make_hdf5_datasets,
                                      make_paired_images_datasets)

name = 'denoise'
help = 'denoise micrographs with various denoising algorithms'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(help)

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

    parser.add_argument('--lowpass', type=float, default=1, help='lowpass filter micrographs by this a0mount (in pixels) before applying the denoising filter. uses a hard lowpass filter (i.e. sinc) (default: no lowpass filtering)')
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



def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)
    
    ## set the device
    use_cuda = set_device(args.device)
    print(f'# using device={args.device} with cuda={use_cuda}', file=sys.stderr)
    
    do_train = (args.dir_a is not None and args.dir_b is not None) or (args.hdf is not None)
    if do_train:
        #create denoiser and send model to GPU if using cuda
        denoiser = Denoise(args.arch, use_cuda)
        
        # create paired datasets for noise2noise training
        if args.hdf is None: #use dirA/dirB
            train_data, val_data = make_paired_images_datasets(args.dir_a, args.dir_b, crop=args.crop, random=np.random, holdout=args.holdout, preload=args.preload, cutoff=args.pixel_cutoff)
        else:
            train_data, val_data = make_hdf5_datasets(args.hdf, paired=True, preload=args.preload, holdout=args.holdout, cutoff=args.pixel_cutoff)

        # train
        denoiser.train(train_data, val_data, loss_fn=args.criteria, optim=args.optim, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, shuffle=True, 
                       num_workers=args.num_workers, verbose=True, save_best=True, save_interval=args.save_interval, save_prefix=args.save_prefix)
        models = [denoiser]
    else: # load the saved model(s)
        models = []
        for arg in args.model:
            out_string = '# Warning: no denoising model will be used' if arg == 'none' else '# Loading model:'+str(arg)
            print(out_string, file=sys.stderr)
            
            denoiser = Denoise(args.arch, use_cuda)
            models.append(denoiser)    

    # always normalize png and jpg format
    normalize = True if args.format_ in ['png', 'jpg'] else args.normalize

    gaus = args.gaussian
    gaus = dn.GaussianDenoise(gaus, use_cuda=use_cuda) if gaus > 0 else None
    
    inv_gaus = args.inv_gaussian
    inv_gaus = dn.InvGaussianFilter(inv_gaus, use_cuda=use_cuda) if inv_gaus > 0 else None
    
    #terminate if no micrographs given
    if len(args.micrographs) < 1:
        return
    if args.stack:
        # we are denoising a single MRC stack
        denoised = denoise_stack(args.micrographs[0], args.output, models, args.lowpass, args.pixel_cutoff, gaus, inv_gaus,
                                 args.deconvolve, args.deconv_patch, args.patch_size, args.patch_padding,
                                 normalize, use_cuda)
    else:
        # stream the micrographs and denoise them
        denoised = denoise_stream(args.micrographs, args.output, args.format_, args.suffix, models, args.lowpass, args.pixel_cutoff, 
                                  gaus, inv_gaus, args.deconvolve, args.deconv_patch, args.patch_size, args.patch_padding,
                                  normalize, use_cuda)
    return denoised

if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
