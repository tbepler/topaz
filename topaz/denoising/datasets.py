import glob
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import h5py
import torch
from topaz import mrc
from topaz.utils.data.loader import load_image


class DenoiseDataset(ABC):
    ''' Dataset of paired images for noise2noise model training
    '''
    paired = True
    
    @abstractmethod
    def __init__(self, data) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, i:int) -> Tuple[np.ndarray]:
        pass


class PairedImages(DenoiseDataset):
    def __init__(self, x:List[str], y:List[str], crop:int=800, xform:bool=True, preload:bool=False, cutoff:float=0):
        self.x = x
        self.y = y
        self.crop = crop
        self.xform = xform
        self.cutoff = cutoff

        self.preload = preload
        if preload:
            self.x = [self.load_image(p) for p in x]
            self.y = [self.load_image(p) for p in y]

    def load_image(self, path:str):
        x = load_image(path, make_image=False, return_header=False)
        x = x.astype(np.float32) # make sure dtype is single precision
        mu = x.mean()
        std = x.std()
        x = (x - mu)/std
        if self.cutoff > 0:
            x[(x < -self.cutoff) | (x > self.cutoff)] = 0
        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.preload:
            x = self.x[i]
            y = self.y[i]
        else:
            x = self.load_image(self.x[i])
            y = self.load_image(self.y[i])

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]
            y = y[i:i+size, j:j+size]

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

        return x, y
    
    
class HDFPairedDataset(DenoiseDataset):
    def __init__(self, dataset:h5py.File, start:int=0, end:int=None, xform:bool=False, cutoff:float=0):
        if end is None:
            self.end = len(dataset)
        n = (self.end - self.start)//2
        self.x = [dataset[start + i*2] for i in range(n)]
        self.y = [dataset[i + 1] for i in range(n)]
        self.xform = xform
        self.cutoff = cutoff

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i): # retrieve the i'th image pair
        x = self.x[i]
        y = self.x[i]

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


class TrainingDataset3D(DenoiseDataset):
    ''' Class for splitting tomograms into training and testing volumes.
        Applies random rotations and flips to augment.
    '''
    def __init__(self, even_path:str, odd_path:str, tilesize:int, N_train:int, N_test:int):
                    
        if tilesize < 1:
            raise ValueError('ERROR: tilesize must be >0')
        if tilesize < 10:
            print('WARNING: small tilesize is not recommended', file=sys.stderr)
        
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
        else:
            print('# Error: Cannot find files or directories:', file=sys.stderr)

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

    def load_mrc(self, path:str):
        with open(path, 'rb') as f:
            content = f.read()
        tomo,_,_ = mrc.parse(content)
        tomo = tomo.astype(np.float32)
        return tomo
    
    def get_train_test_idxs(self, dim:Union[List[int],Tuple[int]]):
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
    
    def sample_coordinates(self, mask:np.ndarray, num_train_vols:int, num_val_vols:int, vol_dims:Union[Tuple[int],List[int]]=(96, 96, 96)):
        
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

    def calc_mean_std(self,tomo:np.ndarray):
        mu = tomo.mean()
        std = tomo.std()
        return mu, std

    def __len__(self):
        if self.mode == 'train':
            return self.N_train * len(self.even)
        else:
            return self.N_test * len(self.even)
            
    def __getitem__(self,idx:int):
        
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
        
        # source = torch.from_numpy(even_).float()
        # target = torch.from_numpy(odd_).float()
        # np arrays fine, over torch Tensors
        source = even_
        target = odd_
        
        return source , target

    def set_mode(self,mode:str):
        modes = ['train','test']
        assert mode in modes
        self.mode = mode 

    def augment(self, x:np.ndarray, y:np.ndarray):
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


class PairedTomograms(DenoiseDataset):
    def __init__(self, x:List[np.ndarray], y:List[np.ndarray]):
        self.x = x
        self.y = y        
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, i: int) -> Tuple[np.ndarray]:
        return self.x[i], self.y[i] 


class PatchDataset(DenoiseDataset):
    def __init__(self, tomo:Union[np.ndarray,torch.Tensor], patch_size:int=96, padding:int=48):
        self.tomo = tomo
        self.patch_size = patch_size
        self.padding = padding

        nzyx = np.array(tomo.shape)
        pzyx = np.ceil(nzyx / patch_size).astype(np.int32)
        self.shape = tuple(pzyx)
        self.num_patches = np.prod(pzyx)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, patch:int):
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
        
        # ensure output is same type and device (if Tensor) as input
        if type(tomo) == np.ndarray:
            x = np.zeros((d, d, d), dtype=np.float32)
        elif type(tomo) == torch.Tensor:
            x = torch.zeros((d, d, d), dtype=torch.float32, device=tomo.device)
        else:
            raise ValueError()

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
        indices = np.array((i,j,k), dtype=int)
        return indices, x



def make_paired_images_datasets(dir_a:str, dir_b:str, crop:int=800, random=np.random, holdout:float=0.1, preload:bool=False, cutoff:float=0):
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

    dataset_train = PairedImages(A_train, B_train, crop=crop, xform=True, preload=preload, cutoff=cutoff)
    dataset_val = PairedImages(A_val, B_val, crop=crop, preload=preload, cutoff=cutoff)

    return dataset_train, dataset_val


def make_hdf5_datasets(path:str, paired:bool=True, preload:bool=False, holdout:float=0.1, cutoff:float=0):

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

    print('# training with', len(dataset_train), 'image pairs', file=sys.stderr)
    print('# validating on', len(dataset_val), 'image pairs', file=sys.stderr)

    return dataset_train, dataset_val


def make_tomogram_datasets(even_path:str, odd_path:str, tilesize:int, N_train:int, N_test:int):
    ''' Split tomograms into training and testing data, augmented with rotations and flips.
    '''
    data = TrainingDataset3D(even_path, odd_path, tilesize, N_train, N_test)
    data.set_mode('train')
    train_data = list(data)
    train_x = [x for (x,_) in train_data]
    train_y = [y for (_,y) in train_data]
    train_dataset = PairedTomograms(train_x, train_y)
    data.set_mode('test')
    test_data = list(data)
    test_x = [x for (x,_) in test_data]
    test_y = [y for (_,y) in test_data]
    test_dataset = PairedTomograms(test_x, test_y)
    
    return train_dataset, test_dataset