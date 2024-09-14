import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from topaz.mrc import parse_header, get_mode_from_header
from typing import List, Literal, Optional
from sklearn.neighbors import KDTree
from topaz.stats import calculate_pi
from topaz.utils.printing import report
from functools import lru_cache
import functools


class MemoryMappedImage():
    '''Class for memory mapping an MRC file and sampling random crops from it.'''
    def __init__(self, image_path: str, targets: pd.DataFrame, crop_size: int, split: str = 'pn', dims: int = 2, use_cuda: bool = False, mask_size=123):
        self.image_path = image_path
        self.targets = targets
        self.size = crop_size
        self.half_size = self.size // 2
        self.split = split
        self.dims = dims
        self.use_cuda = use_cuda
        self.rng = np.random.default_rng()  # Keep this for consistency with original
        self.num_pixels = len(targets)
        self.mask_size = mask_size

        # read image information from header
        with open(self.image_path, 'rb') as f:
            header_bytes = f.read(1024)
        self.header = parse_header(header_bytes)
        self.shape = (self.header.nz, self.header.ny, self.header.nx) if self.dims == 3 else (self.header.ny, self.header.nx)
        self.shape_x = self.shape[-1]
        self.shape_y = self.shape[-2]
        self.shape_z = self.shape[-3] if self.dims == 3 else None
        self.dtype = get_mode_from_header(self.header)
        self.offset = 1024 + self.header.next  # array beginning

        self.check_particle_image_bounds()

        # build a KDTree for the targets
        if split == 'pn' and len(targets) > 0:
            if dims == 3:
                self.positive_tree = KDTree(targets[['z_coord', 'y_coord', 'x_coord']].values)
            elif dims == 2:
                self.positive_tree = KDTree(targets[['y_coord', 'x_coord']].values)
        else:
            self.positive_tree = None

        self.mmap = np.memmap(self.image_path, shape=self.shape, dtype=self.dtype, mode='r', offset=self.offset)

    def extract_crop(self, mmap, xmin: int, xmax: int, ymin: int, ymax: int, zmin: Optional[int] = None, zmax: Optional[int] = None):
        if zmin is not None and zmax is not None:
            return mmap[zmin:zmax, ymin:ymax, xmin:xmax]
        else:
            return mmap[ymin:ymax, xmin:xmax]

    @lru_cache(maxsize=1000)
    def get_crop(self, center_indices):
        z, y, x = center_indices
        # set crop index ranges and any necessary 0-padding
        x_min = max(0, x - self.half_size)
        x_max = min(self.shape_x, x + self.half_size + 1)
        y_min = max(0, y - self.half_size)
        y_max = min(self.shape_y, y + self.half_size + 1)

        xpad = max(0, self.half_size - x), max(0, (x + self.half_size + 1) - self.shape_x)
        ypad = max(0, self.half_size - y), max(0, (y + self.half_size + 1) - self.shape_y)

        zmin, zmax, zpad = None, None, (0, 0)
        if z is not None:
            zmin = max(0, z - self.half_size)
            zmax = min(self.shape_z, z + self.half_size + 1)
            zpad = max(0, self.half_size - z), max(0, (z + self.half_size + 1) - self.shape_z)

        # Create a copy of the data only when necessary
        if self.dims == 3:
            crop = self.mmap[zmin:zmax, y_min:y_max, x_min:x_max]
            if any(pad > 0 for pad in (*zpad, *ypad, *xpad)):
                crop = np.pad(crop, (zpad, ypad, xpad), mode='constant')
        else:
            crop = self.mmap[y_min:y_max, x_min:x_max]
            if any(pad > 0 for pad in (ypad, xpad)):
                crop = np.pad(crop, (ypad, xpad), mode='constant')

        # Convert to float32 and then to a tensor
        crop = torch.from_numpy(crop.astype(np.float32))

        # Move to CUDA if necessary
        if self.use_cuda:
            crop = crop.cuda(non_blocking=True)

        return crop

    def get_random_crop_indices(self):
        '''Return indices for any random pixel in image.'''
        return (self.rng.integers(self.shape_z) if self.dims == 3 else None,
                self.rng.integers(self.shape_y),
                self.rng.integers(self.shape_x))

    def get_random_negative_crop_indices(self):
        '''Sample random indices until we find one that's not in the positive set.'''
        while True:
            if self.dims == 3:
                z, y, x = self.rng.integers(self.shape_z, size=3)
                idx, dist = self.positive_tree.query([[z, y, x]])
            else:
                y, x = self.rng.integers((self.shape_y, self.shape_x), size=2)
                z = None
                idx, dist = self.positive_tree.query([[y, x]])

            if dist > 0:  # not in one of the nodes (assumes all particles are in the tree)
                return z, y, x

    def get_UN_crop(self):
        '''Sample a random crop from the image.'''
        if self.split == 'pu' or len(self.targets) == 0:
            z, y, x = self.get_random_crop_indices()
        elif self.split == 'pn':
            z, y, x = self.get_random_negative_crop_indices()
        return self.get_crop((z, y, x))

    def check_particle_image_bounds(self):
        '''Check that particles are within the image bounds.'''
        coords = ['x_coord', 'y_coord', 'z_coord'] if self.dims == 3 else ['x_coord', 'y_coord']
        limits = [self.shape_x, self.shape_y, self.shape_z] if self.dims == 3 else [self.shape_x, self.shape_y]

        out_of_bounds = np.any((self.targets[coords] < 0) | (self.targets[coords] >= limits), axis=1)

        if out_of_bounds.any():
            report(f'WARNING: ~{int(out_of_bounds.sum() // self.mask_size)} particles are out of bounds for image {self.image_path}. Did you scale the micrographs and particle coordinates correctly?')
            self.targets = self.targets[~out_of_bounds]
            self.num_pixels -= out_of_bounds.sum()

        # also check that the coordinates fill most of the micrograph, cutoffs are arbitrary
        x_max, y_max = self.targets.x_coord.max(), self.targets.y_coord.max()
        z_max = self.targets.z_coord.max() if self.dims == 3 else None

        xy_below_cutoff = (x_max < 0.7 * self.shape_x) and (y_max < 0.7 * self.shape_y)
        z_below_cutoff = (z_max < 0.7 * self.shape_z) if self.dims == 3 else False
        if xy_below_cutoff and self.dims == 2:  # don't warn if 3D
            z_output = f'or z_coord > {z_max}' if (self.dims == 3) else ''
            output = f'WARNING: no coordinates are observed with x_coord > {x_max} or y_coord > {y_max} {z_output}. \
                    Did you scale the micrographs and particle coordinates correctly?'
            report(output)


@functools.lru_cache(maxsize=360)
def cached_rotate(angle):
    return F.rotate(torch.eye(3).unsqueeze(0), angle).squeeze(0)


class MultipleImageSetDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[List[str]], targets: pd.DataFrame, number_samples: int, crop_size: int, image_set_balance: List[float] = None,
                 positive_balance: float = .5, split: str = 'pn', rotate: bool = False, flip: bool = False, dims: int = 2, mode: str = 'training', radius: int = 3,
                 use_cuda: bool = False, mask_size=123):
        '''Dataset for sampling random crops from multiple memory-mapped images. Expects targets to include each positive pixel
        individually, not just particle centers.'''
        self.paths = paths
        # convert float coords to ints
        targets[['y_coord', 'x_coord']] = targets[['y_coord', 'x_coord']].round().astype(int)
        if dims == 3:
            targets[['z_coord']] = targets[['z_coord']].round().astype(int)
        self.targets = targets.to_numpy()
        self.target_columns = targets.columns
        self.number_samples = number_samples  # per epoch
        # increase crop_size to avoid clipping corners
        self.crop_size = crop_size
        self.rotated_crop_size = int(np.ceil(crop_size * np.sqrt(2))) if rotate else crop_size
        # store other parameters
        self.image_set_balance = image_set_balance  # probabilities or uniform
        self.positive_balance = positive_balance
        self.split = split
        self.rotate = rotate
        self.flip = flip
        self.dims = dims
        self.mode = mode
        self.rng = np.random.default_rng()  # Keep this for consistency with original

        self.num_pixels = len(targets)  # all given pixels, remove any unmatched/out-of-bounds later
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
                image = MemoryMappedImage(path, img_targets, self.rotated_crop_size, split, dims=dims, use_cuda=use_cuda, mask_size=mask_size)
                # find image's out-of-bounds particles from its targets
                valid_img_targets = image.targets
                invalid_img_targets = img_targets[~img_targets.index.isin(valid_img_targets.index)]
                # remove invalid_img_targets from self.targets
                self.targets = self.targets[~np.isin(self.targets[:, self.target_columns.get_loc('image_name')], invalid_img_targets['image_name'])]
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
        self.targets = self.targets[~np.isin(self.targets[:, self.target_columns.get_loc('image_name')], unseen_targets['image_name'])]
        if len(unseen_targets) > 0:
            missing = unseen_targets['image_name'].unique().tolist()
            report(f'WARNING: {len(missing)} micrographs listed in the coordinates file are missing from the {mode} images. Image names are listed below.')
            report(f'WARNING: missing micrographs are: {missing}')

    def __len__(self):
        return self.number_samples  # how many crops we want in each epoch

    def __getitem__(self, i):
        # sample an image set
        img_set_idx = self.rng.choice(len(self.paths), p=self.image_set_balance)
        # sample an image from the set
        if self.rng.random() < self.positive_balance:
            # sample a positive coordinate
            target_idx = self.rng.choice(len(self.targets))
            target = self.targets[target_idx]
            name = target[self.target_columns.get_loc('image_name')]
            # get the image with a matching name
            img = self.name_dict[name]
            # extract the crop and positive label
            y, x = target[self.target_columns.get_loc('y_coord')], target[self.target_columns.get_loc('x_coord')]
            z = target[self.target_columns.get_loc('z_coord')] if self.dims == 3 else None
            crop, label = img.get_crop((z, y, x)), 1.
        else:
            # sample a random image
            img_idx = self.rng.choice(len(self.paths[img_set_idx]))
            # sample U/N crop from the image
            img = self.images[img_set_idx][img_idx]
            crop, label = img.get_UN_crop(), 0.

        # apply random transformations (2D only)
        crop = crop.unsqueeze(0)  # add C dim (rotate/flip expects this)
        if self.rotate:
            angle = self.rng.integers(0, 360)
            rotation_matrix = cached_rotate(angle)
            crop = F.affine(crop, angle=0, translate=[0, 0], scale=1, shear=0, matrix=rotation_matrix[:2])
            # remove extra crop/padding
            size_diff = crop.shape[-1] - self.crop_size
            xmin, xmax = size_diff // 2, size_diff // 2 + self.crop_size
            ymin, ymax = size_diff // 2, size_diff // 2 + self.crop_size
            crop = crop[..., ymin:ymax, xmin:xmax]
        if self.flip:
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, [2])  # horizontal flip
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, [1])  # vertical flip
        crop = crop.squeeze(0)  # remove channel dim

        return crop, label
