#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
import glob
import time
import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample
import topaz.mrc as mrc
import topaz.cuda

from topaz.denoise import UDenoiseNet3D, GaussianDenoise3d

name = 'denoise3d'
help = 'denoise 3D volumes with various denoising algorithms'

def add_arguments(parser):

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


def train_epoch(iterator, model, cost_func, optim, epoch=1, num_epochs=1, N=1, use_cuda=False):
    
    c = 0
    loss_accum = 0    
    model.train()

    for batch_idx , (source,target), in enumerate(iterator):
        
        b = source.size(0)        
        loss_mb = 0
        if use_cuda:
            source = source.cuda()
            target = target.cuda()
            
        denoised_source = model(source)
        loss = cost_func(denoised_source,target)
        
        loss.backward()
        optim.step()
        optim.zero_grad()

        loss = loss.item()

        c += b
        delta = b*(loss - loss_accum)
        loss_accum += delta/c

        template = '# [{}/{}] training {:.1%}, Error={:.5f}'
        line = template.format(epoch+1, num_epochs, c/N, loss_accum)
        print(line, end='\r', file=sys.stderr)
    
    print(' '*80, end='\r', file=sys.stderr)    
    return loss_accum


def eval_model(iterator, model, cost_func, epoch=1, num_epochs=1, N=1, use_cuda=False):
    
    c = 0
    loss_accum = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx , (source,target), in enumerate(iterator):
            
            b = source.size(0)        
            loss_mb = 0
            if use_cuda:
                source = source.cuda()
                target = target.cuda()
                
            denoised_source = model(source)
            loss = cost_func(denoised_source,target)
            
            loss = loss.item()
    
            c += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/c
    
            template = '# [{}/{}] testing {:.1%}, Error={:.5f}'
            line = template.format(epoch+1, num_epochs, c/N, loss_accum)
            print(line, end='\r', file=sys.stderr)
            
            
    print(' '*80, end='\r', file=sys.stderr)    
    return loss_accum

class TrainingDataset3D(torch.utils.data.Dataset):
    
    def __init__(self,even_path,odd_path,tilesize,N_train,N_test):

        self.tilesize = tilesize
        self.N_train = N_train
        self.N_test = N_test
        self.mode = 'train'
        
        self.even_paths = []
        self.odd_paths = []

        if os.path.isfile(even_path) and os.path.isfile(odd_path):
            self.even_paths.append(even_path)
            self.odd_paths.append(odd_path)
        elif os.path.isdir(even_path) and os.path.isdir(odd_path):
            for epath in glob.glob(even_path + os.sep + '*'):
                name = os.path.basename(epath)
                opath = odd_path + os.sep + name 
                if not os.path.isfile(opath):
                    print('# Error: name mismatch between even and odd directory,', name, file=sys.stderr)
                    print('# Skipping...', file=sys.stderr)
                else:
                    self.even_paths.append(epath)
                    self.odd_paths.append(opath)

        self.means = []
        self.stds = []
        self.even = []
        self.odd = []
        self.train_idxs = []
        self.test_idxs = []

        for i,(f_even,f_odd) in enumerate(zip(self.even_paths, self.odd_paths)):
            even = self.load_mrc(f_even)
            odd = self.load_mrc(f_odd)

            if even.shape != odd.shape:
                print('# Error: shape mismatch:', f_even, f_odd, file=sys.stderr)
                print('# Skipping...', file=sys.stderr)
            else:
                even_mean,even_std = self.calc_mean_std(even)
                odd_mean,odd_std = self.calc_mean_std(odd)
                self.means.append((even_mean,odd_mean))
                self.stds.append((even_std,odd_std))  

                self.even.append(even)
                self.odd.append(odd)

                mask = np.ones(even.shape, dtype=np.uint8)
                train_idxs, test_idxs = self.sample_coordinates(mask, N_train, N_test, vol_dims=(tilesize, tilesize, tilesize))

                        
                self.train_idxs += train_idxs
                self.test_idxs += test_idxs

        if len(self.even) < 1:
            print('# Error: need at least 1 file to proceeed', file=sys.stderr)
            sys.exit(2)

    def load_mrc(self, path):
        with open(path, 'rb') as f:
            content = f.read()
        tomo,_,_ = mrc.parse(content)
        tomo = tomo.astype(np.float32)
        return tomo
    
    def get_train_test_idxs(self,dim):
        assert len(dim) == 2
        t = self.tilesize
        x = np.arange(0,dim[0]-t,t,dtype=np.int32)
        y = np.arange(0,dim[1]-t,t,dtype=np.int32)
        xx,xy = np.meshgrid(x,y)
        xx = xx.reshape(-1)
        xy = xy.reshape(-1)
        lattice_pts = [list(pos) for pos in zip(xx,xy)]
        n_val = int(self.test_frac*len(lattice_pts))
        test_idx = np.random.choice(np.arange(len(lattice_pts)),
                                   size=n_val,replace=False)
        test_pts = np.hstack([lattice_pts[idx] for idx in test_idx]).reshape((-1,2))
        mask = np.ones(dim,dtype=np.int32)
        for pt in test_pts:
            mask[pt[0]:pt[0]+t-1,pt[1]:pt[1]+t-1] = 0
            mask[pt[0]-t+1:pt[0],pt[1]-t+1:pt[1]] = 0
            mask[pt[0]-t+1:pt[0],pt[1]:pt[1]+t-1] = 0
            mask[pt[0]:pt[0]+t-1,pt[1]-t+1:pt[1]] = 0
    
        mask[-t:,:] = 0
        mask[:,-t:] = 0
        
        train_pts = np.where(mask)
        train_pts = np.hstack([list(pos) for pos in zip(train_pts[0],
                                                train_pts[1])]).reshape((-1,2))
        return train_pts, test_pts
    
    def sample_coordinates(self, mask, num_train_vols, num_val_vols, vol_dims=(96, 96, 96)):
        
        #This function is borrowed from:
        #https://github.com/juglab/cryoCARE_T2T/blob/master/example/generate_train_data.py
        """
        Sample random coordinates for train and validation volumes. The train and validation 
        volumes will not overlap. The volumes are only sampled from foreground regions in the mask.
        
        Parameters
        ----------
        mask : array(int)
            Binary image indicating foreground/background regions. Volumes will only be sampled from 
            foreground regions.
        num_train_vols : int
            Number of train-volume coordinates.
        num_val_vols : int
            Number of validation-volume coordinates.
        vol_dims : tuple(int, int, int)
            Dimensionality of the extracted volumes. Default: ``(96, 96, 96)``
            
        Returns
        -------
        list(tuple(slice, slice, slice))
            Training volume coordinates.
         list(tuple(slice, slice, slice))
            Validation volume coordinates.
        """

        dims = mask.shape
        cent = (np.array(vol_dims) / 2).astype(np.int32)
        mask[:cent[0]] = 0
        mask[-cent[0]:] = 0
        mask[:, :cent[1]] = 0
        mask[:, -cent[1]:] = 0
        mask[:, :, :cent[2]] = 0
        mask[:, :, -cent[2]:] = 0
        
        tv_span = np.round(np.array(vol_dims) / 2).astype(np.int32)
        span = np.round(np.array(mask.shape) * 0.1 / 2 ).astype(np.int32)
        val_sampling_mask = mask.copy()
        val_sampling_mask[:, :span[1]] = 0
        val_sampling_mask[:, -span[1]:] = 0
        val_sampling_mask[:, :, :span[2]] = 0
        val_sampling_mask[:, :, -span[2]:] = 0

        foreground_pos = np.where(val_sampling_mask == 1)
        sample_inds = np.random.choice(len(foreground_pos[0]), 2, replace=False)
    
        val_sampling_mask = np.zeros(mask.shape, dtype=np.int8)
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        for z, y, x in zip(*val_sampling_inds):
            val_sampling_mask[z - span[0]:z + span[0],
            y - span[1]:y + span[1],
            x - span[2]:x + span[2]] = mask[z - span[0]:z + span[0],
                                            y - span[1]:y + span[1],
                                            x - span[2]:x + span[2]].copy()
    
            mask[max(0, z - span[0] - tv_span[0]):min(mask.shape[0], z + span[0] + tv_span[0]),
            max(0, y - span[1] - tv_span[1]):min(mask.shape[1], y + span[1] + tv_span[1]),
            max(0, x - span[2] - tv_span[2]):min(mask.shape[2], x + span[2] + tv_span[2])] = 0
    
        foreground_pos = np.where(val_sampling_mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_val_vols, replace=num_val_vols<len(foreground_pos[0]))
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        val_coords = []
        for z, y, x in zip(*val_sampling_inds):
            val_coords.append(tuple([slice(z-tv_span[0], z+tv_span[0]),
                                     slice(y-tv_span[1], y+tv_span[1]),
                                     slice(x-tv_span[2], x+tv_span[2])]))
    
        foreground_pos = np.where(mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_train_vols, replace=num_train_vols < len(foreground_pos[0]))
        train_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        train_coords = []
        for z, y, x in zip(*train_sampling_inds):
            train_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                     slice(y - tv_span[1], y + tv_span[1]),
                                     slice(x - tv_span[2], x + tv_span[2])]))
        
        return train_coords, val_coords

    def calc_mean_std(self,tomo):
        mu = tomo.mean()
        std = tomo.std()
        return mu, std

    def __len__(self):
        if self.mode == 'train':
            return self.N_train * len(self.even)
        else:
            return self.N_test * len(self.even)
            
    def __getitem__(self,idx):
        
        if self.mode == 'train':
            Idx = int(idx / self.N_train)
            idx = self.train_idxs[idx]
        else:
            Idx = int(idx / self.N_test)
            idx = self.test_idxs[idx]

        even = self.even[Idx]
        odd = self.odd[Idx]
       
        mean = self.means[Idx]
        std = self.stds[Idx]
        
        even_ = even[idx]
        odd_  = odd[idx]
        
        even_ = (even_ - mean[0]) / std[0]
        odd_  = (odd_ - mean[1]) / std[1]
        even_, odd_ = self.augment(even_,odd_)

        even_ = np.expand_dims(even_, axis=0)
        odd_ = np.expand_dims(odd_, axis=0)
        
        source = torch.from_numpy(even_).float()
        target = torch.from_numpy(odd_).float()
        
        return source , target

    def set_mode(self,mode):
        modes = ['train','test']
        assert mode in modes
        self.mode = mode 

    def augment(self, x, y):
        # mirror axes
        for ax in range(3):
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=ax)
                y = np.flip(y, axis=ax)
        
        # rotate around each axis
        for ax in [(0,1), (0,2), (1,2)]:
            k = np.random.randint(4)
            x = np.rot90(x, k=k, axes=ax)
            y = np.rot90(y, k=k, axes=ax)

        return np.ascontiguousarray(x), np.ascontiguousarray(y)


def train_model(even_path, odd_path, save_prefix, save_interval, device
               , base_kernel_width=11
               , cost_func='L2'
               , weight_decay=0
               , learning_rate=0.001
               , optim='adagrad'
               , momentum=0.8
               , minibatch_size=10
               , num_epochs=500
               , N_train=1000
               , N_test=200
               , tilesize=96
               , num_workers=1
               ):
    output = sys.stdout
    log = sys.stderr

    if save_prefix is not None:
        save_dir = os.path.dirname(save_prefix)
        if len(save_dir) > 0 and not os.path.exists(save_dir):
            print('# creating save directory:', save_dir, file=log)
            os.makedirs(save_dir)

    start_time = time.time()
    now = datetime.datetime.now()
    print('# starting time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s'.format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)

    # initialize the model
    print('# initializing model...', file=log)
    model_base = UDenoiseNet3D(base_width=base_kernel_width)
    model,use_cuda,num_devices = set_device(model_base, device)
    
    if cost_func == 'L2':
        cost_func = nn.MSELoss()
    elif cost_func == 'L1':
        cost_func = nn.L1Loss()
    else:
        cost_func = nn.MSELoss()

    wd = weight_decay
    params = [{'params': model.parameters(), 'weight_decay': wd}]
    lr = learning_rate
    if optim == 'sgd':
        optim = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif optim == 'rmsprop':
        optim = torch.optim.RMSprop(params, lr=lr)
    elif optim == 'adam':
        optim = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(params, lr=lr)
    else:
        raise Exception('Unrecognized optim: ' + optim)
        
    # Load the data
    print('# loading data...', file=log)
    if not (os.path.isdir(even_path) or os.path.isfile(even_path)):
        print('ERROR: Cannot find file or directory:', even_path, file=log)
        sys.exit(3)
    if not (os.path.isdir(odd_path) or os.path.isfile(odd_path)):
        print('ERROR: Cannot find directory:', odd_path, file=log)
        sys.exit(3)
    
    if tilesize < 1:
        print('ERROR: tilesize must be >0', file=log)
        sys.exit(4)
    if tilesize < 10:
        print('WARNING: small tilesize is not recommended', file=log)
    data = TrainingDataset3D(even_path, odd_path, tilesize, N_train, N_test)
    
    N_train = len(data)
    data.set_mode('test')
    N_test = len(data)
    data.set_mode('train')
    num_workers = min(num_workers, mp.cpu_count())
    digits = int(np.ceil(np.log10(num_epochs)))

    iterator = torch.utils.data.DataLoader(data,batch_size=minibatch_size,num_workers=num_workers,shuffle=False)
    
    ## Begin model training
    print('# training model...', file=log)
    print('\t'.join(['Epoch', 'Split', 'Error']), file=output)

    for epoch in range(num_epochs):
        data.set_mode('train')
        epoch_loss_accum = train_epoch(iterator,
                                       model,
                                       cost_func,
                                       optim,
                                       epoch=epoch,
                                       num_epochs=num_epochs,
                                       N=N_train,
                                       use_cuda=use_cuda)

        line = '\t'.join([str(epoch+1), 'train', str(epoch_loss_accum)])
        print(line, file=output)
        
        # evaluate on the test set
        data.set_mode('test')
        epoch_loss_accum = eval_model(iterator,
                                   model,
                                   cost_func,
                                   epoch=epoch,
                                   num_epochs=num_epochs,
                                   N=N_test,
                                   use_cuda=use_cuda)
    
        line = '\t'.join([str(epoch+1), 'test', str(epoch_loss_accum)])
        print(line, file=output)

        ## save the models
        if save_prefix is not None and (epoch+1)%save_interval == 0:
            model.eval().cpu()
            save_model(model, epoch+1, save_prefix, digits=digits)
            if use_cuda:
                model.cuda()

    print('# training completed!', file=log)

    end_time = time.time()
    now = datetime.datetime.now()
    print("# ending time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s".format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)
    print("# total time:", time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)), file=log)

    return model_base, num_devices


def save_model(model, epoch, save_prefix, digits=3):
    if type(model) is nn.DataParallel:
        model = model.module

    path = save_prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
    #path = save_prefix + '_epoch{}.sav'.format(epoch)
    torch.save(model, path)


def load_model(path, base_kernel_width=11):
    from collections import OrderedDict
    log = sys.stderr

    # load the model
    pretrained = False
    if path == 'unet-3d': # load the pretrained unet model
        name = 'unet-3d-10a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    elif path == 'unet-3d-10a':
        name = 'unet-3d-10a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    elif path == 'unet-3d-20a':
        name = 'unet-3d-20a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    
    if pretrained:
        print('# loading pretrained model:', name, file=log)

        import pkg_resources
        pkg = __name__
        path = '../pretrained/denoise/' + name
        f = pkg_resources.resource_stream(pkg, path)
        state_dict = torch.load(f) # load the parameters

        model.load_state_dict(state_dict)

    else:
        model = torch.load(path)
        if type(model) is OrderedDict:
            state = model
            model = UDenoiseNet3D(base_width=base_kernel_width)
            model.load_state_dict(state)
    model.eval()

    return model


def set_device(model, device, log=sys.stderr):
    # set the device or devices
    d = device
    use_cuda = (d != -1) and torch.cuda.is_available()
    num_devices = 1
    if use_cuda:
        device_count = torch.cuda.device_count()
        try:
            if d >= 0:
                assert d < device_count
                torch.cuda.set_device(d)
                print('# using CUDA device:', d, file=log)
            elif d == -2:
                print('# using all available CUDA devices:', device_count, file=log)
                num_devices = device_count
                model = nn.DataParallel(model)
            else:
                raise ValueError
        except (AssertionError, ValueError):
            print('ERROR: Invalid device id or format', file=log)
            sys.exit(1)
        except Exception:
            print('ERROR: Something went wrong with setting the compute device', file=log)
            sys.exit(2)

    if use_cuda:
        model.cuda()

    return model, use_cuda, num_devices


class PatchDataset:
    def __init__(self, tomo, patch_size=96, padding=48):
        self.tomo = tomo
        self.patch_size = patch_size
        self.padding = padding

        nz,ny,nx = tomo.shape

        pz = int(np.ceil(nz/patch_size))
        py = int(np.ceil(ny/patch_size))
        px = int(np.ceil(nx/patch_size))
        self.shape = (pz,py,px)
        self.num_patches = pz*py*px


    def __len__(self):
        return self.num_patches

    def __getitem__(self, patch):
        # patch index
        i,j,k = np.unravel_index(patch, self.shape)

        patch_size = self.patch_size
        padding = self.padding
        tomo = self.tomo

        # pixel index
        i = patch_size*i
        j = patch_size*j
        k = patch_size*k

        # make padded patch
        d = patch_size + 2*padding
        x = np.zeros((d, d, d), dtype=np.float32)

        # index in tomogram
        si = max(0, i-padding)
        ei = min(tomo.shape[0], i+patch_size+padding)
        sj = max(0, j-padding)
        ej = min(tomo.shape[1], j+patch_size+padding)
        sk = max(0, k-padding)
        ek = min(tomo.shape[2], k+patch_size+padding)

        # index in crop
        sic = padding - i + si
        eic = sic + (ei - si)
        sjc = padding - j + sj
        ejc = sjc + (ej - sj)
        skc = padding - k + sk
        ekc = skc + (ek - sk)

        x[sic:eic,sjc:ejc,skc:ekc] = tomo[si:ei,sj:ej,sk:ek]
        return np.array((i,j,k), dtype=int),x


def denoise(model, path, outdir, suffix, patch_size=128, padding=128, batch_size=1
           , volume_num=1, total_volumes=1):
    with open(path, 'rb') as f:
        content = f.read()
    tomo,header,extended_header = mrc.parse(content)
    tomo = tomo.astype(np.float32)
    name = os.path.basename(path)

    mu = tomo.mean()
    std = tomo.std()
    # denoise in patches
    d = next(iter(model.parameters())).device
    denoised = np.zeros_like(tomo)

    with torch.no_grad():
        if patch_size < 1:
            x = (tomo - mu)/std
            x = torch.from_numpy(x).to(d)
            x = model(x.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy()
            x = std*x + mu
            denoised[:] = x
        else:
            patch_data = PatchDataset(tomo, patch_size, padding)
            total = len(patch_data)
            count = 0

            batch_iterator = torch.utils.data.DataLoader(patch_data, batch_size=batch_size)
            for index,x in batch_iterator:
                x = x.to(d)
                x = (x - mu)/std
                x = x.unsqueeze(1) # batch x channel

                # denoise
                x = model(x)
                x = x.squeeze(1).cpu().numpy()

                # restore original statistics
                x = std*x + mu

                # stitch into denoised volume
                for b in range(len(x)):
                    i,j,k = index[b]
                    xb = x[b]

                    patch = denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size]
                    pz,py,px = patch.shape

                    xb = xb[padding:padding+pz,padding:padding+py,padding:padding+px]
                    denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size] = xb

                    count += 1
                    print('# [{}/{}] {:.2%}'.format(volume_num, total_volumes, count/total), name, file=sys.stderr, end='\r')

            print(' '*100, file=sys.stderr, end='\r')


    ## save the denoised tomogram
    if outdir is None:
        # write denoised tomogram to same location as input, but add the suffix
        if suffix is None: # use default
            suffix = '.denoised'
        no_ext,ext = os.path.splitext(path)
        outpath = no_ext + suffix + ext
    else:
        if suffix is None:
            suffix = ''
        no_ext,ext = os.path.splitext(name)
        outpath = outdir + os.sep + no_ext + suffix + ext

    # use the read header except for a few fields
    header = header._replace(mode=2) # 32-bit real
    header = header._replace(amin=denoised.min())
    header = header._replace(amax=denoised.max())
    header = header._replace(amean=denoised.mean())

    with open(outpath, 'wb') as f:
        mrc.write(f, denoised, header=header, extended_header=extended_header)


def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    # do denoising
    model = None
    do_train = (args.even_train_path is not None) or (args.odd_train_path is not None)
    if do_train:
        print('# training denoising model!', file=sys.stderr)
        model, num_devices = train_model(args.even_train_path, args.odd_train_path
                           , args.save_prefix, args.save_interval
                           , args.device
                           , base_kernel_width=args.base_kernel_width
                           , cost_func=args.criteria
                           , learning_rate=args.lr
                           , optim=args.optim
                           , momentum=args.momentum
                           , minibatch_size=args.batch_size
                           , num_epochs=args.num_epochs
                           , N_train=args.N_train
                           , N_test=args.N_test
                           , tilesize=args.crop
                           , num_workers=args.num_workers
                           )

    if len(args.volumes) > 0: # tomograms to denoise!
        if model is None: # need to load model
            model = load_model(args.model, base_kernel_width=args.base_kernel_width)

        gaussian_sigma = args.gaussian
        if gaussian_sigma > 0:
            print('# apply Gaussian filter postprocessing with sigma={}'.format(gaussian_sigma), file=sys.stderr)
            model = nn.Sequential(model, GaussianDenoise3d(gaussian_sigma))
        model.eval()
        
        model, use_cuda, num_devices = set_device(model, args.device)

        #batch_size = args.batch_size
        #batch_size *= num_devices
        batch_size = num_devices

        patch_size = args.patch_size
        padding = args.patch_padding
        print('# denoising with patch size={} and padding={}'.format(patch_size, padding), file=sys.stderr)
        # denoise the volumes
        total = len(args.volumes)
        count = 0
        for path in args.volumes:
            count += 1
            denoise(model, path, args.output, args.suffix
                   , patch_size=patch_size
                   , padding=padding
                   , batch_size=batch_size
                   , volume_num=count
                   , total_volumes=total
                   )



