import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision
from topaz.mrc import parse_header, get_mode_from_header
from typing import List, Literal
from sklearn.neighbors import KDTree
from topaz.stats import calculate_pi
from topaz.utils.printing import report
from topaz.utils.data.loader import load_image


class CroppableImage:
    """Base class for images that support random crop sampling.
    
    Subclasses must call _load(image_path) to set self.array and self.shape
    before calling super().__init__().
    """
    def __init__(self, image_path: str, targets: pd.DataFrame, crop_size: int,
                 split: str = 'pn', dims: int = 2, mask_size: int = 123):
        self.image_path = image_path
        self.targets = targets
        self.size = crop_size
        self.split = split
        self.dims = dims
        self.rng = np.random.default_rng()
        self.num_pixels = len(targets)
        self.mask_size = mask_size
        # self.array and self.shape must be set by subclass _load() before this runs
        self.check_particle_image_bounds()

        if split == 'pn' and len(targets) > 0:
            if dims == 3:
                self.positive_tree = KDTree(targets[['z_coord', 'y_coord', 'x_coord']].values)
            else:
                self.positive_tree = KDTree(targets[['y_coord', 'x_coord']].values)
        else:
            self.positive_tree = None

    def _load(self, image_path: str):
        """Load image data and set self.array and self.shape. Must be implemented by subclasses."""
        raise NotImplementedError

    def _ensure_array_loaded(self):
        """Hook for lazy loading. No-op by default (preloaded subclass relies on this)."""
        pass

    def get_crop(self, center_indices):
        """Return a crop centered at the given indices. Pads with zeros if crop exceeds image bounds."""
        self._ensure_array_loaded()
        z, y, x = center_indices
        xmin, xmax = x - self.size // 2, x + self.size // 2 + 1
        ymin, ymax = y - self.size // 2, y + self.size // 2 + 1
        xpad = abs(min(0, xmin)), abs(min(0, self.shape[-1] - xmax))
        ypad = abs(min(0, ymin)), abs(min(0, self.shape[-2] - ymax))
        if z is not None:
            zmin, zmax = z - self.size // 2, z + self.size // 2 + 1
            zpad = abs(min(0, zmin)), abs(min(0, self.shape[-3] - zmax))

        if self.dims == 3:
            crop = self.array[max(0, zmin):zmax, max(0, ymin):ymax, max(0, xmin):xmax]
            crop = np.pad(crop, (zpad, ypad, xpad))
        else:
            crop = self.array[max(0, ymin):ymax, max(0, xmin):xmax]
            crop = np.pad(crop, (ypad, xpad))

        crop = torch.from_numpy(crop)
        return crop

    def get_random_crop_indices(self):
        """Return indices for any random pixel in image."""
        x = self.rng.choice(self.shape[-1])
        y = self.rng.choice(self.shape[-2])
        z = self.rng.choice(self.shape[-3]) if self.dims == 3 else None
        return z, y, x

    def get_random_negative_crop_indices(self):
        """Sample random indices until we find one that's not in the positive set."""
        while True:
            x = self.rng.choice(self.shape[-1])
            y = self.rng.choice(self.shape[-2])
            if self.dims == 3:
                z = self.rng.choice(self.shape[-3])
                idx, dist = self.positive_tree.query([[z, y, x]])
            else:
                z = None
                idx, dist = self.positive_tree.query([[y, x]])
            if dist > 0:
                return z, y, x

    def get_UN_crop(self):
        """Sample a random unlabeled/negative crop from the image."""
        if self.split == 'pu' or len(self.targets) == 0:
            z, y, x = self.get_random_crop_indices()
        else:  # 'pn'
            z, y, x = self.get_random_negative_crop_indices()
        return self.get_crop((z, y, x))

    def check_particle_image_bounds(self):
        """Check that particles are within the image bounds and warn if coordinates seem unscaled."""
        if self.dims == 3:
            out_of_bounds = (
                (self.targets['x_coord'] < 0) | (self.targets['x_coord'] >= self.shape[-1]) |
                (self.targets['y_coord'] < 0) | (self.targets['y_coord'] >= self.shape[-2]) |
                (self.targets['z_coord'] < 0) | (self.targets['z_coord'] >= self.shape[-3])
            )
        else:
            out_of_bounds = (
                (self.targets['x_coord'] < 0) | (self.targets['x_coord'] >= self.shape[-1]) |
                (self.targets['y_coord'] < 0) | (self.targets['y_coord'] >= self.shape[-2])
            )
        if out_of_bounds.any():
            report(f'WARNING: ~{int(out_of_bounds.sum() // self.mask_size)} particles are out of bounds '
                   f'for image {self.image_path}. Did you scale the micrographs and particle coordinates correctly?')
            self.targets = self.targets[~out_of_bounds]
            self.num_pixels -= out_of_bounds.sum()

        if len(self.targets) == 0:
            return

        x_max, y_max = self.targets.x_coord.max(), self.targets.y_coord.max()
        z_max = self.targets.z_coord.max() if self.dims == 3 else None
        xy_below_cutoff = (x_max < 0.7 * self.shape[-1]) and (y_max < 0.7 * self.shape[-2])
        z_below_cutoff = (z_max < 0.7 * self.shape[-3]) if self.dims == 3 else False
        if xy_below_cutoff and self.dims == 2:
            z_output = f'or z_coord > {z_max}' if self.dims == 3 else ''
            report(f'WARNING: no coordinates are observed with x_coord > {x_max} or y_coord > {y_max} {z_output}. '
                   f'Did you scale the micrographs and particle coordinates correctly?')


class MemoryMappedImage(CroppableImage):
    """Lazily memory-maps an MRC file and samples random crops from it."""
    def __init__(self, image_path: str, targets: pd.DataFrame, crop_size: int,
                 split: str = 'pn', dims: int = 2, mask_size: int = 123):
        self.dims = dims # needed by _load() to set shape correctly
        self._load(image_path)  # sets self.array, self.shape, self.dtype, self.offset
        super().__init__(image_path, targets, crop_size, split, dims, mask_size)

    def _load(self, image_path: str):
        with open(image_path, 'rb') as f:
            header_bytes = f.read(1024)
        self.header = parse_header(header_bytes)
        self.shape = (self.header.nz, self.header.ny, self.header.nx) if self.dims == 3 \
            else (self.header.ny, self.header.nx)
        self.dtype = get_mode_from_header(self.header)
        self.offset = 1024 + self.header.next
        self.array = None  # deferred

    def _ensure_array_loaded(self):
        if self.array is None:
            with open(self.image_path, 'rb') as f:
                self.array = np.memmap(f, shape=self.shape, dtype=self.dtype, mode='r', offset=self.offset)


class PreloadedImage(CroppableImage):
    """Eagerly loads an image into memory and samples random crops from it.
    
    Preferred for 2D micrograph or small datasets where the full dataset fits in RAM, avoiding repeated file I/O during training.
    """
    def __init__(self, image_path: str, targets: pd.DataFrame, crop_size: int,
                 split: str = 'pn', dims: int = 2, mask_size: int = 123):
        self.dims = dims # not needed, but keeps consistent with mem-mapped version
        self._load(image_path)  # sets self.array and self.shape
        super().__init__(image_path, targets, crop_size, split, dims, mask_size)

    def _load(self, image_path: str):
        # load_image returns a numpy array; make_image=False skips PIL conversion
        array = load_image(image_path, make_image=False, return_header=False)
        self.array = array
        self.shape = self.array.shape


class MultipleImageSetDataset(torch.utils.data.Dataset):
    def __init__(self, paths:List[List[str]], targets:pd.DataFrame, number_samples:int, crop_size:int, image_set_balance:List[float]=None, 
                 positive_balance:float=.5, split:str='pn', rotate:bool=False, flip:bool=False, dims:int=2, mode:str='training', radius:int=3,
                 mask_size=123, preload:bool=True):
        '''Dataset for sampling random crops from multiple memory-mapped images. Expects targets to include each positive pixel
        individually, not just particle centers.'''
        self.paths = paths
        # convert float coords to ints
        targets[['y_coord', 'x_coord']] = targets[['y_coord', 'x_coord']].round().astype(int)
        if dims == 3:
            targets[['z_coord']] = targets[['z_coord']].round().astype(int)
        self.targets = targets
        self.number_samples = number_samples # per epoch
        # increase crop_size to avoid clipping corners
        self.crop_size = crop_size
        crop_size = int(np.ceil(crop_size*np.sqrt(2))) if rotate else crop_size
        # store other parameters
        self.image_set_balance = image_set_balance # probabilities or uniform
        self.positive_balance = positive_balance
        self.split = split
        self.rotate = rotate
        self.flip = flip
        self.dims = dims
        self.mode = mode
        self.rng = np.random.default_rng()
        
        self.num_pixels = len(targets) # all given pixels, remove any unmatched/out-of-bounds later
        self.images = []
        self.num_images = 0
        self.name_dict = {}
        
        unseen_targets = targets.copy()
        for group in paths:
            group_list = []
            for path in group:
                # get image name without file extension
                img_name = os.path.splitext(path.split('/')[-1])[0]
                # create image object with matching targets
                image_name_matches = unseen_targets['image_name'] == img_name
                img_targets = unseen_targets[image_name_matches]
                # choose subclass based on preload flag (preload for 2D micrographs or small datasets, memory-map for large 3D tomograms)
                image = PreloadedImage(path, img_targets, crop_size, split, dims=dims, mask_size=mask_size) if preload \
                    else MemoryMappedImage(path, img_targets, crop_size, split, dims=dims, mask_size=mask_size) 
                # find image's out-of-bounds particles from its targets
                valid_img_targets = image.targets
                invalid_img_targets = img_targets[~img_targets.index.isin(valid_img_targets.index)]
                # remove invalid_img_targets from self.targets
                self.targets = self.targets[~self.targets.index.isin(invalid_img_targets.index)]
                self.num_pixels -= len(invalid_img_targets)
                # store image and map name to image object
                self.num_images += 1
                self.name_dict[img_name] = image
                group_list.append(image)
                # remove targets just processed
                unseen_targets = unseen_targets[~image_name_matches]
            self.images.append(group_list)
        
        # remove any targets that don't match any images
        self.num_pixels -= len(unseen_targets)
        self.targets = self.targets[~self.targets.index.isin(unseen_targets.index)]
        if len(unseen_targets) > 0:
            missing = unseen_targets.image_name.unique().tolist()
            report(f'WARNING: {len(missing)} micrographs listed in the coordinates file are missing from the {mode} images. Image names are listed below.')
            report(f'WARNING: missing micrographs are: {missing}')
            
    def __len__(self):
        return self.number_samples # how many crops we want in each epoch

    def __getitem__(self, i):
        # sample an image set
        img_set_idx = self.rng.choice(len(self.paths), p=self.image_set_balance)
        # sample an image from the set
        if self.rng.random() < self.positive_balance:
            # sample a positive coordinate
            target = self.targets.sample()
            name = target['image_name'].item()
            # get the image with a matching name
            img = self.name_dict[name]
            # extract the crop and positive label
            y, x = target['y_coord'].item(), target['x_coord'].item()
            z = target['z_coord'].item() if self.dims==3 else None
            crop, label = img.get_crop((z, y, x)), 1.
        else:
            # sample a random image
            img_idx = self.rng.choice(len(self.paths[img_set_idx]))
            # sample U/N crop from the image
            img = self.images[img_set_idx][img_idx]
            crop,label = img.get_UN_crop(), 0.
        # apply random transformations
        crop = crop.unsqueeze(0) # add C dim (rotate/flip expects this)
        if self.rotate:
            angle = self.rng.uniform(0, 360)
            crop = torchvision.transforms.functional.rotate(crop, angle)
            # remove the extra padding we added before
            size_diff = crop.shape[-1] - self.crop_size
            xmin, xmax = size_diff//2, size_diff//2 + self.crop_size
            ymin, ymax = size_diff//2, size_diff//2 + self.crop_size
            if self.dims == 3:
                zmin, zmax = size_diff//2, size_diff//2 + self.crop_size
                crop = crop[..., zmin:zmax, ymin:ymax, xmin:xmax]
            else:
                crop = crop[..., ymin:ymax, xmin:xmax]
        if self.flip:
            if self.rng.random() < 0.5:
                crop = torchvision.transforms.functional.hflip(crop)
            if self.rng.random() < 0.5:
                crop = torchvision.transforms.functional.vflip(crop)
        crop = crop.squeeze(0) # remove channel dim

        return crop,label