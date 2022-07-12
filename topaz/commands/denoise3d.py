#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import sys

from topaz.cuda import set_device
from topaz.denoise import Denoise3D, denoise_tomogram_stream
from topaz.denoising.datasets import make_tomogram_datasets

name = 'denoise3d'
help = 'denoise 3D volumes with various denoising algorithms'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(help)

    parser.add_argument('volumes', nargs='*', help='volumes to denoise')
    parser.add_argument('-o', '--output', help='directory to save denoised volumes')
    parser.add_argument('--suffix', help='optional suffix to append to file paths. if not output is specfied, denoised volumes are written to the same location as the input with the suffix appended to the name (default .denoised)')

    parser.add_argument('-m', '--model', default='unet-3d', help='use pretrained denoising model. accepts path to a previously saved model or one of the provided pretrained models. pretrained model options are: unet-3d, unet-3d-10a, unet-3d-20a (default: unet-3d)')

    ## training parameters
    parser.add_argument('-a', '--even-train-path', help='path to even training data')
    parser.add_argument('-b', '--odd-train-path', help='path to odd training data')

    parser.add_argument('--N-train', type=int, default=1000, help='Number of train points per volume (default: 1000)')
    parser.add_argument('--N-test', type=int, default=200, help='Number of test points per volume (default: 200)')

    parser.add_argument('-c', '--crop', type=int, default=96, help='training tile size (default: 96)')
    parser.add_argument('--base-kernel-width', type=int, default=11, help='width of the base convolutional filter kernel in the U-net model (default: 11)')

    parser.add_argument('--optim', choices=['adam', 'adagrad', 'sgd'], default='adagrad', help='optimizer (default: adagrad)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--criteria', default='L2', choices=['L1', 'L2'], help='training criteria (default: L2)')
    parser.add_argument('--momentum', type=float, default=0.8, help='momentum parameter for SGD optimizer (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=10, help='minibatch size (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=500, help='number of training epochs (default: 500)')


    parser.add_argument('-w', '--weight_decay', type=float, default=0, help='L2 regularizer on the generative network (default: 0)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--save-prefix', help='path prefix to save denoising model')

    parser.add_argument('--num-workers', type=int, default=1, help='number of workers for dataloader (default: 1)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')


    ## denoising parameters
    parser.add_argument('-g', '--gaussian', type=float, default=0, help='standard deviation of Gaussian filter postprocessing, 0 means no postprocessing (default: 0)')
    parser.add_argument('-s', '--patch-size', type=int, default=96, help='denoises volumes in patches of this size. not used if <1 (default: 96)')
    parser.add_argument('-p', '--patch-padding', type=int, default=48, help='padding around each patch to remove edge artifacts (default: 48)')

    ## other parameters
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device/s to use (default: -2, multi gpu), set to >= 0 for single gpu, set to -1 for cpu')

    return parser



def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    ## set the device
    use_cuda = set_device(args.device)
    print(f'# using device={args.device} with cuda={use_cuda}', file=sys.stderr)
    
    do_train = (args.even_train_path is not None) or (args.odd_train_path is not None)
    if do_train:
        #create denoiser and send model to GPU if using cuda
        denoiser = Denoise3D(args.arch, use_cuda)
        
        # create paired datasets for noise2noise training
        train_data, val_data = make_tomogram_datasets(args.even_train_path, args.odd_train_path, 
                                                      args.patch_size, args.N_train, args.N_test)

        # train
        denoiser.train(train_data, val_data, loss_fn=args.criteria, optim=args.optim, lr=args.lr, batch_size=args.batch_size, 
                       num_epochs=args.num_epochs, shuffle=True, num_workers=args.num_workers, verbose=True, save_best=True,
                       save_interval=args.save_interval, save_prefix=args.save_prefix)
    else: # load the saved model(s)
        out_string = '# Warning: no denoising model will be used' if args.model == 'none' else '# Loading model:'+str(args.model)
        print(out_string, file=sys.stderr)
        denoiser = Denoise3D(args.arch, use_cuda) if args.model != 'none' else None

    total = len(args.volumes)
    #terminate if no tomograms given
    if total < 1:
        return
    
    print(f'# denoising {total} tomograms with patch size={args.patch_size} and padding={args.padding}', file=sys.stderr)
    # denoise the volumes
    denoised = denoise_tomogram_stream(volumes=args.volumes, model=denoiser, output_path=args.output, suffix=args.suffix, gaus=args.gaus, 
                                       patch_size=args.patch_size, padding=args.patch_padding, verbose=True, use_cuda=use_cuda)
    return denoised


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
