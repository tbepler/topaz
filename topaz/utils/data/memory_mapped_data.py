import os
import sys
import numpy as np
import torch
import torchvision
from topaz.mrc import parse_header, get_mode_from_header
from typing import List, Literal
from sklearn.neighbors import KDTree
from topaz.stats import calculate_pi
from topaz.utils.printing import report
from torch.cuda.amp import autocast, GradScaler
import multiprocessing as mp
from torch.utils.data import DataLoader

class MemoryMappedImage:
    """
    A class to handle memory-mapped images for efficient data loading.
    """

    def __init__(self, image_path: str, targets: np.ndarray, crop_size: int, split: str = 'pn', dims: int = 2, use_cuda: bool = False):
        """
        Initialize the MemoryMappedImage.

        Args:
            image_path (str): Path to the image file.
            targets (np.ndarray): Array of target coordinates.
            crop_size (int): Size of the crop.
            split (str): Type of split ('pn' or 'pu').
            dims (int): Number of dimensions (2 or 3).
            use_cuda (bool): Whether to use CUDA.
        """
        self.image_path = image_path
        self.targets = targets
        self.size = crop_size
        self.split = split
        self.dims = dims
        self.use_cuda = use_cuda
        self.rng = np.random.default_rng()
        self.num_particles = len(targets)

        # Read the image header
        with open(self.image_path, 'rb') as f:
            header_bytes = f.read(1024)
        self.header = parse_header(header_bytes)
        self.shape = (self.header.nz, self.header.ny, self.header.nx)
        self.dtype = get_mode_from_header(self.header)
        self.offset = 1024 + self.header.next

        # Create memory-mapped array
        self.array = np.memmap(self.image_path, dtype=self.dtype, mode='r', offset=self.offset, shape=self.shape)

        self.check_particle_image_bounds()

        # Create KDTree for efficient nearest neighbor search
        if split == 'pn' and len(targets) > 0:
            coords = self.targets[:, 1:3] if dims == 2 else self.targets[:, 1:4]
            self.positive_tree = KDTree(coords)
        else:
            self.positive_tree = None

    def get_crop(self, center_indices):
        """
        Get a crop from the image centered at the given indices.

        Args:
            center_indices (tuple): Center coordinates of the crop.

        Returns:
            torch.Tensor: The cropped image.
        """
        z, y, x = center_indices
        xmin, xmax = max(0, int(x - self.size // 2)), min(self.shape[-1], int(x + self.size // 2 + 1))
        ymin, ymax = max(0, int(y - self.size // 2)), min(self.shape[-2], int(y + self.size // 2 + 1))

        if self.dims == 3:
            zmin, zmax = max(0, int(z - self.size // 2)), min(self.shape[-3], int(z + self.size // 2 + 1))
            crop = self.array[zmin:zmax, ymin:ymax, xmin:xmax]
        else:
            crop = self.array[ymin:ymax, xmin:xmax]

        # Ensure the crop is the correct size
        if self.dims == 3:
            pad_width = [(0, max(0, self.size - (zmax - zmin))),
                         (0, max(0, self.size - (ymax - ymin))),
                         (0, max(0, self.size - (xmax - xmin)))]
        else:
            pad_width = [(0, max(0, self.size - (ymax - ymin))),
                         (0, max(0, self.size - (xmax - xmin)))]

        crop = np.pad(crop, pad_width, mode='constant')

        # Ensure the crop is exactly the right size (in case of rounding errors)
        if self.dims == 3:
            crop = crop[:self.size, :self.size, :self.size]
        else:
            crop = crop[:self.size, :self.size]

        crop = torch.from_numpy(crop.copy())

        if self.use_cuda:
            crop = crop.cuda()

        return crop

    def get_random_crop_indices(self):
        """
        Get random crop indices.

        Returns:
            tuple: Random crop indices.
        """
        x = self.rng.choice(self.shape[-1])
        y = self.rng.choice(self.shape[-2])
        z = self.rng.choice(self.shape[-3]) if self.dims == 3 else None
        return z, y, x

    def get_random_negative_crop_indices(self):
        """
        Get random negative crop indices.

        Returns:
            tuple: Random negative crop indices.
        """
        while True:
            x = self.rng.choice(self.shape[-1])
            y = self.rng.choice(self.shape[-2])
            if self.dims == 3:
                z = self.rng.choice(self.shape[-3])
                idx, dist = self.positive_tree.query([[z, y, x]])
            else:
                z = None
                idx, dist = self.positive_tree.query([[y, x]])

            if dist[0][0] > 0:
                return z, y, x

    def get_UN_crop(self):
        """
        Get an unlabeled negative crop.

        Returns:
            torch.Tensor: The unlabeled negative crop.
        """
        if self.split == 'pu' or len(self.targets) == 0:
            z, y, x = self.get_random_crop_indices()
        elif self.split == 'pn':
            z, y, x = self.get_random_negative_crop_indices()
        return self.get_crop((z, y, x))

    def check_particle_image_bounds(self):
        """
        Check if particles are within image bounds and warn if not.
        """
        if self.dims == 3:
            out_of_bounds = (self.targets[:, 1] < 0) | (self.targets[:, 2] < 0) | (self.targets[:, 3] < 0) | \
                             (self.targets[:, 1] >= self.shape[-1]) | (self.targets[:, 2] >= self.shape[-2]) | (self.targets[:, 3] >= self.shape[-3])
        else:
            out_of_bounds = (self.targets[:, 1] < 0) | (self.targets[:, 2] < 0) | \
                             (self.targets[:, 1] >= self.shape[-1]) | (self.targets[:, 2] >= self.shape[-2])
        if out_of_bounds.any():
            report(f'WARNING: {out_of_bounds.sum()} particles are out of bounds for image {self.image_path}. Did you scale the micrographs and particle coordinates correctly?')
            self.targets = self.targets[~out_of_bounds]
            self.num_particles -= out_of_bounds.sum()

        x_max, y_max = self.targets[:, 1].max(), self.targets[:, 2].max()
        z_max = self.targets[:, 3].max() if self.dims == 3 else None

        xy_below_cutoff = (x_max < 0.7 * self.shape[-1]) and (y_max < 0.7 * self.shape[-2])
        z_below_cutoff = (z_max < 0.7 * self.shape[-3]) if self.dims == 3 else False
        if xy_below_cutoff and self.dims == 2:
            z_output = f'or z_coord > {z_max}' if (self.dims == 3) else ''
            output = f'WARNING: no coordinates are observed with x_coord > {x_max} or y_coord > {y_max} {z_output}. \
                    Did you scale the micrographs and particle coordinates correctly?'
            report(output)

class MultipleImageSetDataset(torch.utils.data.Dataset):
    """
    A dataset class for handling multiple sets of images and their targets.
    """

    def __init__(self, paths: List[List[str]], targets: np.ndarray, number_samples: int, crop_size: int, image_set_balance: List[float] = None,
                 positive_balance: float = .5, split: str = 'pn', rotate: bool = False, flip: bool = False, dims: int = 2, mode: str = 'training', radius: int = 3, use_cuda: bool = False):
        """
        Initialize the MultipleImageSetDataset.

        Args:
            paths (List[List[str]]): List of lists of image paths.
            targets (np.ndarray): Array of target coordinates.
            number_samples (int): Number of samples in the dataset.
            crop_size (int): Size of the crop.
            image_set_balance (List[float], optional): Balance between different image sets.
            positive_balance (float): Balance between positive and negative samples.
            split (str): Type of split ('pn' or 'pu').
            rotate (bool): Whether to apply rotation augmentation.
            flip (bool): Whether to apply flip augmentation.
            dims (int): Number of dimensions (2 or 3).
            mode (str): Mode of the dataset ('training' or 'testing').
            radius (int): Radius for particle detection.
            use_cuda (bool): Whether to use CUDA.
        """
        self.paths = paths
        self.targets = targets
        self.number_samples = number_samples
        self.crop_size = crop_size
        crop_size = int(np.ceil(crop_size * np.sqrt(2))) if rotate else crop_size
        self.image_set_balance = image_set_balance
        self.positive_balance = positive_balance
        self.split = split
        self.rotate = rotate
        self.flip = flip
        self.dims = dims
        self.mode = mode
        self.rng = np.random.default_rng()
        self.images = []
        self.num_images = 0
        self.num_particles = 0

        print('Initializing MultipleImageSetDataset...')

        targets_dict = {name: group for name, group in targets.groupby('image_name')}

        processed_images = set()
        all_particles = set()

        for group in paths:
            group_list = []
            for path in group:
                img_name = os.path.basename(path)
                if img_name in processed_images:
                    continue
                processed_images.add(img_name)

                print(f"Processing image: {img_name}")

                img_targets = targets_dict.get(img_name)
                if img_targets is None:
                    print(f'No targets matched for image: {img_name}')
                    continue

                print(f'Matched targets: {len(img_targets)}')

                try:
                    mmi = MemoryMappedImage(path, img_targets.to_numpy(), crop_size, split, dims=dims, use_cuda=use_cuda)
                    group_list.append(mmi)
                    self.num_images += 1

                    for _, row in img_targets.iterrows():
                        particle = (img_name, row['x_coord'], row['y_coord'])
                        if dims == 3:
                            particle += (row['z_coord'],)
                        all_particles.add(particle)

                except Exception as e:
                    print(f"Error processing image {img_name}: {str(e)}")

            if group_list:
                self.images.append(group_list)

        self.num_particles = len(all_particles)

        missing = set(targets_dict.keys()) - processed_images
        if missing:
            print(f'WARNING: {len(missing)} micrographs listed in the coordinates file are missing from the {mode} images. Image names are listed below.')
            print(f'WARNING: missing micrographs are: {list(missing)}')

        print(f'MultipleImageSetDataset initialized with {self.num_images} images and {self.num_particles} unique particles.')

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.number_samples

    def __getitem__(self, i):
        """
        Get an item from the dataset.

        Args:
            i (int): Index of the item.

        Returns:
            tuple: A tuple containing the crop and its label.
        """
        if not self.images:
            raise ValueError("No images loaded in the dataset.")

        img_set_idx = self.rng.choice(len(self.images), p=self.image_set_balance)
        if self.rng.random() < self.positive_balance:
            img = self.rng.choice(self.images[img_set_idx])
            if img.num_particles == 0:
                return self.__getitem__(i)
            target = img.targets[self.rng.choice(img.num_particles)]
            y, x = int(target[2]), int(target[1])
            z = int(target[3]) if self.dims == 3 else None
            crop, label = img.get_crop((z, y, x)), 1.
        else:
            img = self.rng.choice(self.images[img_set_idx])
            crop, label = img.get_UN_crop(), 0.

        if self.rotate:
            angle = self.rng.uniform(0, 360)
            crop = torchvision.transforms.functional.rotate(crop, angle)
            size_diff = crop.shape[-1] - self.crop_size
            xmin, xmax = size_diff // 2, size_diff // 2 + self.crop_size
            ymin, ymax = size_diff // 2, size_diff // 2 + self.crop_size
            crop = crop[:, ymin:ymax, xmin:xmax]
        if self.flip:
            if self.rng.random() < 0.5:
                crop = torchvision.transforms.functional.hflip(crop)
            if self.rng.random() < 0.5:
                crop = torchvision.transforms.functional.vflip(crop)

        return crop, label

    def get_batch(self, batch_size):
        """
        Get a batch of samples from the dataset.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            tuple: A tuple containing a batch of crops and their labels.
        """
        crops = []
        labels = []
        for _ in range(batch_size):
            crop, label = self.__getitem__(0)  # 0 is a dummy index
            crops.append(crop)
            labels.append(label)
        return torch.stack(crops), torch.tensor(labels)

def worker_init_fn(worker_id):
    """
    Initialize the random number generator for each worker.

    Args:
        worker_id (int): ID of the worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.rng = np.random.default_rng(np.random.MT19937(np.random.SeedSequence(worker_id)))

def train_model(model, train_dataset, test_dataset, optimizer, criterion, num_epochs, batch_size, use_cuda):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        test_dataset (torch.utils.data.Dataset): The testing dataset.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        use_cuda (bool): Whether to use CUDA.

    Returns:
        torch.nn.Module: The trained model.
    """
    scaler = GradScaler()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count(), worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count()) if test_dataset else None

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if test_loader:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    if use_cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(targets.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            print(f'Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model
