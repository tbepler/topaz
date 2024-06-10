import numpy as np
import pandas as pd
import torch
import torchvision
from topaz.utils.data.loader import load_mrc
from topaz.mrc import parse_header, get_mode_from_header
from typing import List
from sklearn.neighbors import KDTree

class MemoryMappedImage():
    '''Class for memory mapping an MRC file and sampling random crops from it.'''
    def __init__(self, image_path:str, targets:pd.DataFrame, crop_size:int, balance:float=0.5, mode='pn'):
        self.image_path = image_path
        self.targets = targets
        self.size = crop_size
        self.balance = balance
        self.mode = mode
        self.rng = np.random.default_rng()
        
        # read image information from header
        with open(self.image_path, 'rb') as f:
            header_bytes = f.read(1024)
        self.header = parse_header(header_bytes)
        self.shape = (self.header.nz, self.header.ny, self.header.nx)
        self.dtype = get_mode_from_header(self.header)
        self.offset = 1024 + self.header.next # array beginning
        
        # build a KDTree for the targets
        if mode == 'pn' and len(self.shape)==3:
            self.positive_tree = KDTree(targets[['z_coord', 'y_coord', 'x_coord']].values) 
        elif mode == 'pn' and len(self.shape)==2:
            self.positive_tree = KDTree(targets[['y_coord', 'x_coord']].values) 
        else:
            self.positive_tree = None
        
    def __getitem__(self, i):
        '''Randomly sample a target and the associated crop of given size'''
        label = 0
        if self.rng.random() < self.balance:
            # sample a positive target
            target = self.targets.sample()
            z, y, x = target['z_coord'].item(), target['y_coord'].item(), target['x_coord'].item() #TODO: need to round, why float in the first place
            label = 1
        elif self.mode == 'pn':
            # sample a negative target
            z, y, x = self.get_random_negative_crop_indices()
        elif self.mode == 'pu':
            # sample any crop
            z, y, x = self.get_random_crop_indices()
            
        crop = self.get_crop((z, y, x))
        return crop, label
    
    def get_crop(self, center_indices):
        z,y,x = center_indices
        # set crop index ranges and any necessary 0-padding
        xmin, xmax, ymin, ymax = x-self.size//2, x+self.size//2+1, y-self.size//2, y+self.size//2+1
        xpad = abs(min(0,xmin)), abs(min(0,self.shape[-1]-xmax))
        ypad = abs(min(0,ymin)), abs(min(0,self.shape[-2]-ymax))
        
        if z is not None:
            zmin, zmax = z-self.size//2, z+self.size//2+1
            zpad = abs(min(0,zmin)), abs(min(0,self.shape[-3]-zmax))

        with open(self.image_path, 'rb') as f:
            array = np.memmap(f, shape=self.shape, dtype=self.dtype, mode='r', offset=self.offset)
            if len(self.shape) == 3:
                crop = array[max(0,zmin):zmax, max(0,ymin):ymax, max(0,xmin):xmax]
                crop = np.pad(crop, (zpad, ypad, xpad))
            elif len(self.shape) == 2:
                crop = array[max(0,ymin):ymax, max(0,xmin):xmax]
                crop = np.pad(crop, (ypad, xpad))

        crop = torch.from_numpy(crop)
        return crop
    
    def get_random_crop_indices(self):
        '''Return indices for any random pixel in image.'''
        x = self.rng.choice(self.shape[-1])
        y = self.rng.choice(self.shape[-2])
        z = self.rng.choice(self.shape[-3]) if len(self.shape) == 3 else None
        return z, y, x

    def get_random_negative_crop_indices(self):
        '''Sample random indices until we find one that's not in the positive set.'''
        while True:
            x = self.rng.choice(self.shape[-1])
            y = self.rng.choice(self.shape[-2])
            if len(self.shape) == 3:
                z = self.rng.choice(self.shape[-2]) 
                idx, dist = self.positive_tree.query([[z, y, x]])
            else:
                z = None
                idx, dist = self.positive_tree.query((y, x))
                
            if dist > 0: # not in one of the nodes
                return z, y, x


class MultipleImageSetDataset(torch.utils.data.Dataset):
    def __init__(self, paths:List[List[str]], targets:pd.DataFrame, number_samples:int, crop_size:int, image_set_balance:List[float]=None, positive_balance:float=0.5, mode:str='pn'):
        # convert float coords to ints, regardless of 2d/3d
        names = targets['image_name']
        targets = targets.drop(columns=['image_name']).round().astype(int)
        targets['image_name'] = names
        
        self.paths = paths
        self.images = []
        for group in paths:
            group_list = []
            for path in group:
                #get image name without file extension
                img_name = path.split('/')[-1].replace('.mrc','')
                img_targets = targets[targets['image_name']==img_name]
                group_list.append(MemoryMappedImage(path, img_targets, crop_size, positive_balance, mode))
            self.images.append(group_list)
            
        self.number_samples = number_samples # per epoch
        self.crop_size = crop_size
        self.image_set_balance = image_set_balance
        self.positive_balance = positive_balance
        self.mode = mode
        self.image_set_balance = image_set_balance # probabilties or uniform
        self.rng = np.random.default_rng()
        
    def __len__(self):
        return self.number_samples # how many crops we want in each epoch

    def __getitem__(self, i):
        # sample an image set
        img_set_idx = self.rng.choice(len(self.paths), p=self.image_set_balance)
        # sample an image from the set
        img_idx = self.rng.choice(len(self.paths[img_set_idx]))
        # sample a random crop and label from the image
        img = self.images[img_set_idx][img_idx]
        crop, label = img[i]
        
        return crop,label