from __future__ import division, print_function

import glob
import multiprocessing as mp
import os
import sys
from typing import List, Literal, Tuple, Union
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import topaz.methods as methods
import topaz.model.classifier as C
import topaz.utils.data.partition
import topaz.utils.files as file_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from topaz.metrics import average_precision
from topaz.model.classifier import classify_patches
from topaz.model.factory import get_feature_extractor, load_model
from topaz.model.generative import ConvGenerator
from topaz.mrc import parse_header
from topaz.stats import pixels_given_radius, calculate_pi
from topaz.utils.data.coordinates import match_coordinates_to_images
from topaz.utils.data.loader import (LabeledImageCropDataset,
                                     SegmentedImageDataset,
                                     load_images_from_list, load_image)
from topaz.utils.data.sampler import (RandomImageTransforms,
                                      StratifiedCoordinateSampler)
from topaz.utils.data.memory_mapped_data import MultipleImageSetDataset
from topaz.utils.printing import report
from topaz.utils.picks import as_mask

def match_images_targets(images: dict, targets: pd.DataFrame, radius: float, dims: int = 2, use_cuda: bool = False) \
        -> Tuple[List[List[Union[Image.Image, np.ndarray]]], List[List[np.ndarray]]]:
    """
    Match images with their corresponding targets.

    Args:
        images (dict): Dictionary of images.
        targets (pd.DataFrame): DataFrame containing target coordinates.
        radius (float): Radius for matching.
        dims (int): Number of dimensions (2 or 3).
        use_cuda (bool): Whether to use CUDA.

    Returns:
        Tuple containing lists of matched images and targets.
    """
    matched = match_coordinates_to_images(targets, images, radius=radius, dims=dims, use_cuda=use_cuda)
    images_list = []
    targets_list = []
    for key in matched:
        these_images, these_targets = zip(*list(matched[key].values()))
        images_list.append(list(these_images))
        targets_list.append(list(these_targets))
    return images_list, targets_list

def filter_targets_missing_images(images: pd.DataFrame, targets: pd.DataFrame, mode: str = 'training'):
    """
    Filter out targets that don't have corresponding images.

    Args:
        images (pd.DataFrame): DataFrame of images.
        targets (pd.DataFrame): DataFrame of targets.
        mode (str): 'training' or 'test'.

    Returns:
        Filtered targets DataFrame.
    """
    names = set()
    for k, d in images.items():
        for name in d.keys():
            names.add(name)
    check = targets.image_name.apply(lambda x: x in names)
    missing = targets.image_name.loc[~check].unique().tolist()
    if len(missing) > 0:
        print(f'WARNING: {len(missing)} micrographs listed in the coordinates file are missing from the {mode} images. Image names are listed below.', file=sys.stderr)
        print(f'WARNING: missing micrographs are: {missing}', file=sys.stderr)
    targets = targets.loc[check]
    return targets

def convert_path_to_grouped_list(images_path: str, targets: pd.DataFrame) -> List[List[str]]:
    """
    Convert image paths to a grouped list based on the targets.

    Args:
        images_path (str): Path to images or directory containing images.
        targets (pd.DataFrame): DataFrame of targets.

    Returns:
        List of lists containing grouped image paths.
    """
    if os.path.isdir(images_path):
        glob_base = images_path + os.sep + '*'
        image_paths = glob.glob(glob_base + '.mrc')
        image_name = [os.path.basename(x) for x in image_paths]
        image_paths = pd.DataFrame({'image_path': image_paths, 'image_name': image_name})
    else:
        image_paths = pd.read_csv(images_path, sep='\s+')

    # Logging to check the image paths and names
    print("Image paths loaded:")
    print(image_paths)

    if 'source' not in targets.columns:
        targets['source'] = 0

    # Merging directly on the original image names
    merged = pd.merge(image_paths, targets, on='image_name', how='inner')

    # Logging to check the merged dataframe
    print("Merged image paths with targets:")
    print(merged)

    if 'source' not in merged.columns:
        merged['source'] = 0

    grouped = merged.groupby('source')['image_path'].apply(list).tolist()
    return grouped

def load_image_set(images_path, targets_path, image_ext, radius, format_, as_images=True, mode='training',
                   dims=2, use_cuda=False) -> Tuple[List[List[Union[Image.Image, np.ndarray]]], List[List[np.ndarray]]]:
    """
    Load a set of images and their corresponding targets.

    Args:
        images_path (str): Path to images or directory containing images.
        targets_path (str): Path to targets file.
        image_ext (str): Image file extension.
        radius (float): Radius for matching targets to images.
        format_ (str): Format of the targets file.
        as_images (bool): Whether to load as PIL Images.
        mode (str): 'training' or 'test'.
        dims (int): Number of dimensions (2 or 3).
        use_cuda (bool): Whether to use CUDA.

    Returns:
        Tuple containing lists of images and targets.
    """
    if os.path.isdir(images_path):
        paths = glob.glob(images_path + os.sep + '*' + image_ext)
        valid_paths, image_names = [], []
        for path in paths:
            name = os.path.basename(path)
            if name.endswith('.mrc') or name.endswith('.tiff') or name.endswith('.png'):
                image_names.append(name)
                valid_paths.append(path)
        images = pd.DataFrame({'image_name': image_names, 'path': valid_paths})
    else:
        images = pd.read_csv(images_path, sep='\t')
    targets = file_utils.read_coordinates(targets_path, format=format_)

    if 'source' not in images and 'source' not in targets:
        images['source'] = 0
        targets['source'] = 0

    images = load_images_from_list(images.image_name, images.path, sources=images.source, as_images=as_images)

    targets = filter_targets_missing_images(images, targets, mode=mode)

    check_particle_image_bounds(images, targets, dims=dims)

    num_micrographs = sum(len(images[k]) for k in images.keys())
    num_particles = len(targets)
    report(f'Loaded {num_micrographs} {mode} micrographs with {num_particles} labeled particles')
    if num_particles == 0 and mode == 'training':
        print('ERROR: no training particles specified. Check that micrograph names in the particles file match those in the micrographs file/directory.', file=sys.stderr)
        raise Exception('No training particles.')

    images, targets = match_images_targets(images, targets, radius, dims=dims, use_cuda=use_cuda)
    report(f'Created target binary masks for {mode} micrographs.')
    return images, targets

def check_particle_image_bounds(images: pd.DataFrame, targets: pd.DataFrame, dims=2):
    """
    Check if particle coordinates are within image bounds.

    Args:
        images (pd.DataFrame): DataFrame of images.
        targets (pd.DataFrame): DataFrame of targets.
        dims (int): Number of dimensions (2 or 3).
    """
    width, height, depth = 0, 0, 0
    for k, d in images.items():
        for image in d.values():
            if dims == 2:
                w, h = image.size if (type(image) == Image.Image) else (image.shape[1], image.shape[0])
            elif dims == 3:
                d, h, w = image.shape
            width, height = max(w, width), max(h, height)
            depth = max(d, depth) if dims == 3 else 0
    out_of_bounds = (targets.x_coord > width) | (targets.y_coord > height) | (dims == 3 and targets.z_coord > depth)
    count = out_of_bounds.sum()

    if count > int(0.1 * len(targets)):
        print(f'WARNING: {count} particle coordinates are out of the micrograph dimensions. Did you scale the micrographs and particle coordinates correctly?', file=sys.stderr)
    x_max, y_max = targets.x_coord.max(), targets.y_coord.max()
    z_max = targets.z_coord.max() if dims == 3 else None
    xy_below_cutoff = (x_max < 0.7 * width) and (y_max < 0.7 * height)
    if xy_below_cutoff:
        z_output = f'or z_coord > {z_max}' if (dims == 3) and (z_max < 0.7 * depth) else ''
        output = f'WARNING: no coordinates are observed with x_coord > {x_max} or y_coord > {y_max} {z_output}. \
                Did you scale the micrographs and particle coordinates correctly?'
        print(output, file=sys.stderr)

def make_traindataset(X: List[List[Union[Image.Image, np.ndarray]]], Y: List[List[np.ndarray]], crop: int,
                      dims: int = 2) -> Union[LabeledImageCropDataset, RandomImageTransforms]:
    """
    Create a dataset for training with data augmentation.

    Args:
        X (List[List[Union[Image.Image, np.ndarray]]]): List of lists of images.
        Y (List[List[np.ndarray]]): List of lists of targets.
        crop (int): Crop size.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        Union[LabeledImageCropDataset, RandomImageTransforms]: Dataset for training.
    """
    size = int(np.ceil(crop * np.sqrt(2)))  # multiply square side by hypotenuse to ensure rotations dont remove corners
    size += 1 if size % 2 == 0 else 0
    dataset = LabeledImageCropDataset(X, Y, size, dims=dims)
    if dims == 3:  # don't augment 3D volumes
        transformed = RandomImageTransforms(dataset, crop=crop, dims=dims, flip=False, rotate=False)
    else:
        transformed = RandomImageTransforms(dataset, crop=crop, dims=dims, flip=True, rotate=True)
    return transformed

def calculate_positive_fraction(targets):
    """
    Calculate the fraction of positive samples in the targets.

    Args:
        targets (List[List[np.ndarray]]): List of lists of target arrays.

    Returns:
        float: Mean fraction of positive samples.
    """
    per_source = []
    for source_targets in targets:
        positives = sum(target.sum() for target in source_targets)
        total = sum(target.size for target in source_targets)
        per_source.append(positives / total)
    return np.mean(per_source)

def cross_validation_split(k: int, fold: int, images: List[Union[Image.Image, np.ndarray]], targets: List[np.ndarray], random=np.random):
    """
    Perform k-fold cross-validation split.

    Args:
        k (int): Number of folds.
        fold (int): Current fold.
        images (List[Union[Image.Image, np.ndarray]]): List of images.
        targets (List[np.ndarray]): List of targets.
        random (np.random.RandomState): Random state for reproducibility.

    Returns:
        Tuple containing train and test splits of images and targets.
    """
    ## calculate number of positives per image for stratified split
    source = []
    index = []
    count = []
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            source.append(i)
            index.append(j)
            count.append(targets[i][j].sum())
    counts_table = pd.DataFrame({'source': source, 'image_name': index, 'count': count})
    partitions = list(topaz.utils.data.partition.kfold(k, counts_table))

    ## make the split from the partition indices
    train_table, validate_table = partitions[fold]

    test_images = [[]*len(images)]
    test_targets = [[]*len(targets)]
    for _, row in validate_table.iterrows():
        i = row['source']
        j = row['image_name']
        test_images[i].append(images[i][j])
        test_targets[i].append(targets[i][j])

    train_images = [[]*len(images)]
    train_targets = [[]*len(targets)]
    for _, row in train_table.iterrows():
        i = row['source']
        j = row['image_name']
        train_images[i].append(images[i][j])
        train_targets[i].append(targets[i][j])

    return train_images, train_targets, test_images, test_targets

def load_data(train_images_path: str, train_targets_path: str, test_images_path: str, test_targets_path: str, radius: float, k_fold: int = 0, fold: int = 0,
              cross_validation_seed: int = 42, format_: str = 'auto', image_ext: str = '', as_images: bool = True, dims: int = 2, use_cuda: bool = False):
    """
    Load training and testing data, optionally performing cross-validation split.

    Args:
        train_images_path (str): Path to training images.
        train_targets_path (str): Path to training targets.
        test_images_path (str): Path to test images (optional).
        test_targets_path (str): Path to test targets (optional).
        radius (float): Radius for matching targets to images.
        k_fold (int): Number of folds for cross-validation (if > 1).
        fold (int): Current fold for cross-validation.
        cross_validation_seed (int): Random seed for cross-validation.
        format_ (str): Format of the targets file.
        image_ext (str): Image file extension.
        as_images (bool): Whether to load as PIL Images.
        dims (int): Number of dimensions (2 or 3).
        use_cuda (bool): Whether to use CUDA.

    Returns:
        Tuple containing training and testing images and targets.
    """
    # load training images and target particles
    train_images, train_targets = load_image_set(train_images_path, train_targets_path, image_ext=image_ext, radius=radius,
                                                 format_=format_, as_images=as_images, mode='training', dims=dims, use_cuda=use_cuda)
    # load test images and target particles or split training
    if test_images_path is not None:
        test_images, test_targets = load_image_set(test_images_path, test_targets_path, image_ext=image_ext, radius=radius,
                                                   format_=format_, as_images=as_images, mode='test', dims=dims, use_cuda=use_cuda)
    elif k_fold > 1:
        ## seed for partitioning the data
        random = np.random.RandomState(cross_validation_seed)
        ## make the split
        train_images, train_targets, test_images, test_targets = cross_validation_split(k_fold, fold, train_images, train_targets, random=random)

        n_train = sum(len(images) for images in train_images)
        n_test = sum(len(images) for images in test_images)
        report('Split into {} train and {} test micrographs'.format(n_train, n_test))
    else:
        test_images, test_targets = None, None

    return train_images, train_targets, test_images, test_targets

def report_data_stats_old(train_images, train_targets, test_images, test_targets):
    """
    Report statistics about the dataset (old version).

    Args:
        train_images (List[List[Union[Image.Image, np.ndarray]]]): Training images.
        train_targets (List[List[np.ndarray]]): Training targets.
        test_images (List[List[Union[Image.Image, np.ndarray]]]): Test images.
        test_targets (List[List[np.ndarray]]): Test targets.

    Returns:
        Tuple[int, int]: Number of positive regions and total regions.
    """
    report('source\tsplit\tp_observed\tnum_positive_regions\ttotal_regions')
    num_positive_regions = 0
    total_regions = 0
    for i in range(len(train_images)):
        p = sum(train_targets[i][j].sum() for j in range(len(train_targets[i])))
        p = int(p)
        total = sum(train_targets[i][j].numel() for j in range(len(train_targets[i])))
        num_positive_regions += p
        total_regions += total
        p_observed = p / total
        p_observed = '{:.3g}'.format(p_observed)
        report(str(i) + '\t' + 'train' + '\t' + p_observed + '\t' + str(p) + '\t' + str(total))
    if test_targets is not None:
        p = sum(test_targets[i][j].sum() for j in range(len(test_targets[i])))
        p = int(p)
        total = sum(test_targets[i][j].numel() for j in range(len(test_targets[i])))
        p_observed = p / total
        p_observed = '{:.3g}'.format(p_observed)
        report(str(i) + '\t' + 'test' + '\t' + p_observed + '\t' + str(p) + '\t' + str(total))
    return num_positive_regions, total_regions

def extract_image_stats(image_paths: List[List[str]], targets: pd.DataFrame, mode: str = 'train', radius: int = 3, dims: int = 2) -> Tuple[int, int]:
    """
    Extract statistics from images and targets.

    Args:
        image_paths (List[List[str]]): List of lists of image paths.
        targets (pd.DataFrame): DataFrame of targets.
        mode (str): 'train' or 'test'.
        radius (int): Radius for particle detection.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        Tuple[int, int]: Number of positive regions and total regions.
    """
    num_positive_regions = 0
    total_regions = 0
    pixels_per_particle = pixels_given_radius(radius, dims)
    for source, source_paths in enumerate(image_paths):
        source_positive_regions = 0
        source_total_regions = 0
        for path in source_paths:
            with open(path, 'rb') as f:
                header_bytes = f.read(1024)
                header = parse_header(header_bytes)
            source_total_regions += header.nz * header.ny * header.nx
            image_name = os.path.basename(path)
            target = targets[targets['image_name'] == image_name]
            source_positive_regions += (len(target) * pixels_per_particle)
        p_observed = source_positive_regions / source_total_regions
        report(f'{source}\t{mode}\t{p_observed:.2f}\t{source_positive_regions}\t{source_total_regions}')
        num_positive_regions += source_positive_regions
        total_regions += source_total_regions
    return num_positive_regions, total_regions

def report_data_stats(train_images_path: str, train_targets_path: str, test_images_path: str = None, test_targets_path: str = None,
                      radius: int = 3, dims: int = 2) -> Tuple[int, int, int]:
    """
    Report statistics about the dataset.

    Args:
        train_images_path (str): Path to training images.
        train_targets_path (str): Path to training targets.
        test_images_path (str): Path to test images (optional).
        test_targets_path (str): Path to test targets (optional).
        radius (int): Radius for particle detection.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        Tuple[int, int, int]: Number of positive regions, total regions, and number of training images.
    """
    report('source\tsplit\tp_observed\tnum_positive_regions\ttotal_regions')
    # Read targets into dataframe
    train_targets = file_utils.read_coordinates(train_targets_path)
    # Convert paths to grouped lists of paths
    train_grouped = convert_path_to_grouped_list(train_images_path, train_targets)
    num_train_images = sum(len(group) for group in train_grouped)
    # Calculate the number of positive and total regions
    num_positive_regions, total_regions = extract_image_stats(train_grouped, train_targets, mode='train', radius=radius, dims=dims)
    # Repeat on testing set if given
    if (test_images_path is not None) and (test_targets_path is not None):
        test_targets = file_utils.read_coordinates(test_targets_path)
        test_grouped = convert_path_to_grouped_list(test_images_path, test_targets)
        test_positive, test_total = extract_image_stats(test_grouped, test_targets, mode='test', radius=radius, dims=dims)
    return num_positive_regions, total_regions, num_train_images

def make_model(args):
    """
    Create or load a model based on the provided arguments.

    Args:
        args: Argument object containing model parameters.

    Returns:
        nn.Module: The created or loaded model.
    """
    report('Loading model:', args.model)
    if args.model.endswith('.sav'):  # loading pretrained model
        model = torch.load(args.model)
        model.train()
        return model

    report('Model parameters: units={}, dropout={}, bn={}'.format(args.units, args.dropout, args.bn))

    # initialize the model
    units = args.units
    dropout = args.dropout
    bn = args.bn == 'on'
    pooling = args.pooling
    unit_scaling = args.unit_scaling

    arch = args.model
    flag = None
    if args.pretrained:
        # check if model parameters match an available pretrained model
        if arch == 'resnet8' and units == 32:
            flag = 'resnet8_u32'
        elif arch == 'resnet8' and units == 64:
            flag = 'resnet8_u64'
        elif arch == 'resnet16' and units == 32:
            flag = 'resnet16_u32'
        elif arch == 'resnet16' and units == 64:
            flag = 'resnet16_u64'

    if flag is not None:
        report('Loading pretrained model:', flag)
        classifier = load_model(flag)
        classifier.train()
    else:
        feature_extractor = get_feature_extractor(args.model, units, dropout=dropout, bn=bn
                                                  , unit_scaling=unit_scaling, pooling=pooling)
        classifier = C.LinearClassifier(feature_extractor, dims=2, patch_size=args.patch_size, padding=args.patch_padding, batch_size=args.minibatch_size)

    ## if the method is generative, create the generative model as well
    generative = None
    if args.autoencoder > 0:
        ngf = args.ngf
        depth = int(np.log2(classifier.width + 1) - 3)
        generative = ConvGenerator(classifier.latent_dim, units=ngf, depth=depth)
        ## attach the generative model
        classifier.generative = generative
        report('Generator: units={}, size={}'.format(ngf, generative.width))

    report('Receptive field:', classifier.width)

    return classifier

def make_training_step_method(classifier, num_positive_regions, positive_fraction
                              , lr=1e-3, l2=0, method='GE-binomial', pi=0, slack=-1
                              , autoencoder=0):
    """
    Create a training step method based on the specified parameters.

    Args:
        classifier (nn.Module): The classifier model.
        num_positive_regions (int): Number of positive regions.
        positive_fraction (float): Fraction of positive samples.
        lr (float): Learning rate.
        l2 (float): L2 regularization strength.
        method (str): Training method ('GE-binomial', 'GE-KL', 'PN', or 'PU').
        pi (float): Expected fraction of positives.
        slack (float): Slack parameter for GE methods.
        autoencoder (int): Autoencoder strength.

    Returns:
        Tuple containing the trainer, criteria, and split type.
    """
    criteria = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam

    # pi sets the expected fraction of positives
    # but during training, we iterate over unlabeled data with labeled positives removed
    # therefore, we expected the fraction of positives in the unlabeled data
    # to be pi - fraction of labeled positives
    # if we are using the 'GE-KL' or 'GE-binomial' loss functions
    p_observed = positive_fraction
    if pi <= p_observed and method in ['GE-KL', 'GE-binomial']:
        # if pi <= p_observed, then we think the unlabeled data is all negatives
        # report this to the user and switch method to 'PN' if it isn't already
        print(f'WARNING: pi={pi} but the observed fraction of positives is {p_observed} and method is set to {method}.', file=sys.stderr)
        print(f'WARNING: setting method to PN with pi={p_observed} instead.', file=sys.stderr)
        print(f'WARNING: if you meant to use {method}, please set pi > {p_observed}.', file=sys.stderr)
        pi = p_observed
        method = 'PN'
    elif method in ['GE-KL', 'GE-binomial']:
        pi = pi - p_observed

    split = 'pn'
    if method == 'PN':
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PN(classifier, optim, criteria, pi=pi, l2=l2, autoencoder=autoencoder)

    elif method == 'GE-KL':
        if slack < 0:
            slack = 10
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_KL(classifier, optim, criteria, pi, l2=l2, slack=slack)

    elif method == 'GE-binomial':
        if slack < 0:
            slack = 1
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.GE_binomial(classifier, optim, criteria, pi, l2=l2, slack=slack, autoencoder=autoencoder)

    elif method == 'PU':
        split = 'pu'
        optim = optim(classifier.parameters(), lr=lr)
        trainer = methods.PU(classifier, optim, criteria, pi, l2=l2, autoencoder=autoencoder)

    else:
        raise Exception('Invalid method: ' + method)

    return trainer, criteria, split

def make_data_iterators_old(train_images: List[List[Union[Image.Image, np.ndarray]]], train_targets: List[List[np.ndarray]],
                            test_images: List[List[Union[Image.Image, np.ndarray]]], test_targets: List[List[np.ndarray]],
                            crop: int, split: Literal['pn', 'pu'], args, dims: int = 2, to_tensor: bool = True):
    """
    Create data iterators for training and testing (old version).

    Args:
        train_images (List[List[Union[Image.Image, np.ndarray]]]): Training images.
        train_targets (List[List[np.ndarray]]): Training targets.
        test_images (List[List[Union[Image.Image, np.ndarray]]]): Test images.
        test_targets (List[List[np.ndarray]]): Test targets.
        crop (int): Crop size.
        split (Literal['pn', 'pu']): Split type ('pn' or 'pu').
        args: Argument object containing training parameters.
        dims (int): Number of dimensions (2 or 3).
        to_tensor (bool): Whether to convert to PyTorch tensors.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    ## training parameters
    minibatch_size = args.minibatch_size
    epoch_size = args.epoch_size
    num_epochs = args.num_epochs
    num_workers = mp.cpu_count() if args.num_workers < 0 else args.num_workers  # set num workers to use all CPUs
    testing_batch_size = args.test_batch_size
    balance = None if args.natural else args.minibatch_balance  # ratio of positive to negative in minibatch
    report(f'minibatch_size={minibatch_size}, epoch_size={epoch_size}, num_epochs={num_epochs}')

    ## create augmented training dataset
    train_dataset = make_traindataset(train_images, train_targets, crop, dims=dims)
    test_dataset = SegmentedImageDataset(test_images, test_targets, to_tensor=to_tensor) if test_targets is not None else None

    ## create minibatch iterators
    labels = train_dataset.data.labels
    sampler = StratifiedCoordinateSampler(labels, size=epoch_size * minibatch_size, balance=balance, split=split)

    train_iterator = DataLoader(train_dataset, batch_size=minibatch_size, sampler=sampler, num_workers=num_workers)
    test_iterator = DataLoader(test_dataset, batch_size=testing_batch_size, num_workers=0) if test_dataset is not None else None
    return train_iterator, test_iterator

class TestingImageDataset():
    """
    Dataset for testing images.

    Args:
        images_path (str): Path to images or directory containing images.
        targets (pd.DataFrame): DataFrame of targets.
        radius (int): Radius for particle detection.
        dims (int): Number of dimensions (2 or 3).
        use_cuda (bool): Whether to use CUDA.
    """
    def __init__(self, images_path: str, targets: pd.DataFrame, radius: int = 3, dims: int = 2, use_cuda: bool = False):
        if os.path.isdir(images_path):
            glob_base = images_path + os.sep + '*'
            image_paths = glob.glob(glob_base + '.mrc')
        else:
            image_paths = pd.read_csv(images_path, sep='\s+')['image_name'].tolist()
        self.image_paths = image_paths
        self.targets = targets
        self.radius = radius
        self.dims = dims
        self.use_cuda = use_cuda

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        img = load_image(path, make_image=False, return_header=False)
        img = torch.from_numpy(img.copy())
        img_name = os.path.basename(path)
        image_name_matches = self.targets['image_name'] == img_name
        img_targets = self.targets[image_name_matches]
        x = img_targets['x_coord'].values
        y = img_targets['y_coord'].values
        z = img_targets['z_coord'].values if self.dims == 3 else None
        mask = as_mask(img.shape, self.radius, x, y, z, use_cuda=self.use_cuda)

        if self.use_cuda:
            img = img.cuda()
            mask = mask.cuda()

        return img, mask

def expand_target_points(targets: pd.DataFrame, radius: int, dims: int = 2) -> pd.DataFrame:
    """
    Expand target point coordinates into coordinates of a sphere with the given radius.

    Args:
        targets (pd.DataFrame): DataFrame of targets.
        radius (int): Radius for expansion.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        Tuple[pd.DataFrame, int]: Expanded targets and mask size.
    """
    x_coord, y_coord = targets['x_coord'].values, targets['y_coord'].values
    # make the spherically mask array of offsets to apply to the coordinates
    sphere_width = int(np.floor(radius)) * 2 + 1
    center = sphere_width // 2
    filter_range = torch.arange(sphere_width)
    grid = torch.meshgrid([filter_range] * dims, indexing='xy')
    xgrid, ygrid = grid[0], grid[1]
    d2 = (xgrid - center) ** 2 + (ygrid - center) ** 2
    if dims == 3:
        z_coord = targets['z_coord'].values
        zgrid = grid[2]
        d2 += (zgrid - center) ** 2
    mask = (d2 <= radius ** 2).float()

    mask_size = mask.sum()
    sphere_offsets = mask.nonzero() - center
    sphere_offsets = pd.DataFrame(sphere_offsets.numpy(), columns=['z_offset', 'y_offset', 'x_offset'])
    # create all combinations of targets and offsets
    expanded = targets.merge(sphere_offsets, how='cross')
    expanded['x_coord'] = expanded['x_coord'] + expanded['x_offset']
    expanded['y_coord'] = expanded['y_coord'] + expanded['y_offset']
    if dims == 3:
        expanded['z_coord'] = expanded['z_coord'] + expanded['z_offset']
        return expanded[['image_name', 'x_coord', 'y_coord', 'z_coord']], mask_size
    else:
        return expanded[['image_name', 'x_coord', 'y_coord']], mask_size

def make_data_iterators(train_image_path: str, train_targets_path: str, crop: int, split: Literal['pn', 'pu'], minibatch_size: int, epoch_size: int,
                        test_image_path: str = None, test_targets_path: str = None, testing_batch_size: int = 1, num_workers: int = 0, balance: float = 0.5,
                        dims: int = 2, use_cuda: bool = False, radius: int = 3) -> Tuple[DataLoader, DataLoader]:
    """
    Create data iterators for training and testing.

    Args:
        train_image_path (str): Path to training images.
        train_targets_path (str): Path to training targets.
        crop (int): Crop size.
        split (Literal['pn', 'pu']): Split type ('pn' or 'pu').
        minibatch_size (int): Minibatch size.
        epoch_size (int): Epoch size.
        test_image_path (str): Path to test images (optional).
        test_targets_path (str): Path to test targets (optional).
        testing_batch_size (int): Batch size for testing.
        num_workers (int): Number of workers for data loading.
        balance (float): Balance ratio for positive/negative samples.
        dims (int): Number of dimensions (2 or 3).
        use_cuda (bool): Whether to use CUDA.
        radius (int): Radius for particle detection.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    report('Reading train targets...')
    train_targets = file_utils.read_coordinates(train_targets_path)
    report(f'Loaded {len(train_targets)} train targets')

    if len(train_targets) == 0:
        report('ERROR: no training particles specified. Check that micrograph names in the particles file match those in the micrographs file/directory.', file=sys.stderr)
        raise Exception('No training particles.')

    report('Converting path to grouped list...')
    train_image_paths = convert_path_to_grouped_list(train_image_path, train_targets)
    report(f'Converted {len(train_image_paths)} train image paths')

    report('Expanding target points...')
    expanded_train_targets, mask_size = expand_target_points(train_targets, radius, dims)
    report(f'Expanded train targets with mask size {mask_size}')

    report('Creating train dataset...')
    train_dataset = MultipleImageSetDataset(train_image_paths, expanded_train_targets, epoch_size * minibatch_size, crop, positive_balance=balance, split=split,
                                            rotate=(dims == 2), flip=(dims == 2), mode='training', dims=dims, radius=radius, use_cuda=use_cuda)
    train_dataloader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    report(f'Loaded {train_dataset.num_images} training micrographs with ~{int(train_dataset.num_particles // mask_size)} labeled particles')

    if test_targets_path is not None:
        report('Reading test targets...')
        test_targets = file_utils.read_coordinates(test_targets_path)
        report(f'Loaded {len(test_targets)} test targets')

        report('Creating test dataset...')
        test_dataset = TestingImageDataset(test_image_path, test_targets, radius=radius, dims=dims, use_cuda=use_cuda)
        test_dataloader = DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False, num_workers=num_workers)
        report(f'Loaded {len(test_dataset)} testing micrographs with {len(test_targets)} labeled particles')
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, None

def evaluate_model(classifier, criteria, data_iterator, use_cuda=False):
    """
    Evaluate the model on a given dataset.

    Args:
        classifier (nn.Module): The classifier model.
        criteria (nn.Module): Loss function.
        data_iterator (DataLoader): Data iterator for the dataset.
        use_cuda (bool): Whether to use CUDA.

    Returns:
        Tuple containing loss, precision, true positive rate, false positive rate, and AUPRC.
    """
    classifier.eval()
    classifier.fill()

    n = 0
    loss = 0
    scores = []
    Y_true = []

    with torch.no_grad():
        for X, Y in data_iterator:
            Y = Y.view(-1)
            Y_true.append(Y.cpu().numpy())
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()

            if classifier.dims == 2:
                score = classifier(X).view(-1)
            elif classifier.dims == 3:
                score = classify_patches(classifier, X, batch_size=data_iterator.batch_size,
                                         patch_size=classifier.patch_size, padding=classifier.padding).view(-1)

            scores.append(score.data.cpu().numpy())
            this_loss = criteria(score, Y).item()

            n += Y.size(0)
            delta = Y.size(0) * (this_loss - loss)
            loss += delta / n

    scores = np.concatenate(scores, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    y_hat = 1.0 / (1.0 + np.exp(-scores))
    precision = y_hat[Y_true == 1].sum() / y_hat.sum()
    tpr = y_hat[Y_true == 1].mean()
    fpr = y_hat[Y_true == 0].mean()

    auprc = average_precision(Y_true, scores)

    classifier.unfill()

    return loss, precision, tpr, fpr, auprc

def fit_epoch(step_method, data_iterator, epoch=1, it=1, use_cuda=False, output=sys.stdout):
    """
    Fit the model for one epoch.

    Args:
        step_method: Training step method.
        data_iterator (DataLoader): Data iterator for training.
        epoch (int): Current epoch number.
        it (int): Current iteration number.
        use_cuda (bool): Whether to use CUDA.
        output: Output stream for logging.

    Returns:
        Tuple[int, float]: Updated iteration number and AUPRC for the epoch.
    """
    pbar = tqdm(data_iterator, desc=f"Epoch {epoch}", leave=False, position=2)
    all_scores = []
    all_labels = []
    for X, Y in pbar:
        Y = Y.view(-1)
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        metrics = step_method.step(X, Y)

        # Use the model attribute of step_method instead of classifier
        scores = step_method.model(X).view(-1)

        all_scores.extend(scores.detach().cpu().numpy())
        all_labels.extend(Y.detach().cpu().numpy())
        line = '\t'.join([str(epoch), str(it), 'train'] + [str(metric) for metric in metrics] + ['-'])
        print(line, file=output)
        it += 1
        pbar.set_postfix({'loss': metrics[0]})  # Assuming the first metric is loss

    # Calculate AUPRC for the entire epoch
    auprc = average_precision(np.array(all_labels), np.array(all_scores))
    return it, auprc

def fit_epochs(classifier, criteria, step_method, train_iterator, test_iterator, num_epochs,
              save_prefix=None, use_cuda=False, output=sys.stdout, num_sets=1, current_set=1):
    """
    Fit the model for multiple epochs.

    Args:
        classifier (nn.Module): The classifier model.
        criteria (nn.Module): Loss function.
        step_method: Training step method.
        train_iterator (DataLoader): Data iterator for training.
        test_iterator (DataLoader): Data iterator for testing.
        num_epochs (int): Number of epochs to train.
        save_prefix (str): Prefix for saving model checkpoints.
        use_cuda (bool): Whether to use CUDA.
        output: Output stream for logging.
        num_sets (int): Total number of sets.
        current_set (int): Current set number.

    Returns:
        nn.Module: The trained classifier.
    """
    header = step_method.header
    line = '\t'.join(['epoch', 'iter', 'split'] + header + ['auprc'])
    print(line, file=output)

    it = 1
    pbar_sets = tqdm(total=num_sets, desc="Overall Progress", position=0)
    pbar_sets.update(current_set - 1)  # Update to the current set
    pbar_epochs = tqdm(range(1, num_epochs+1), desc=f"Set {current_set} Progress", position=1, leave=False)

    for epoch in pbar_epochs:
        classifier.train()
        it, train_auprc = fit_epoch(step_method, train_iterator, epoch=epoch, it=it, use_cuda=use_cuda, output=output)

        # Add train AUPRC to the output
        line = '\t'.join([str(epoch), str(it), 'train'] + ['-']*(len(header)) + [str(train_auprc)])
        print(line, file=output)

        if test_iterator is not None:
            loss, precision, tpr, fpr, auprc = evaluate_model(classifier, criteria, test_iterator, use_cuda=use_cuda)
            line = '\t'.join([str(epoch), str(it), 'test', str(loss)] + ['-']*(len(header)-4) + [str(precision), str(tpr), str(fpr), str(auprc)])
            print(line, file=output)
            output.flush()
            pbar_epochs.set_postfix({'test_loss': loss, 'test_auprc': auprc, 'train_auprc': train_auprc})

        if save_prefix is not None:
            prefix = save_prefix
            digits = int(np.ceil(np.log10(num_epochs)))
            path = prefix + (f'_set{current_set}_epoch{{:0{digits}}}.sav').format(epoch)
            classifier.cpu()
            torch.save(classifier, path)
            if use_cuda:
                classifier.cuda()

    pbar_sets.update(1)
    pbar_epochs.close()

    if current_set == num_sets:
        pbar_sets.close()

def train_model_old(classifier, train_images, train_targets, test_images, test_targets, use_cuda, save_prefix, output, args, dims: int = 2, to_tensor: bool = True):
    """
    Train the model (old version).

    Args:
        classifier (nn.Module): The classifier model.
        train_images (List[List[Union[Image.Image, np.ndarray]]]): Training images.
        train_targets (List[List[np.ndarray]]): Training targets.
        test_images (List[List[Union[Image.Image, np.ndarray]]]): Test images.
        test_targets (List[List[np.ndarray]]): Test targets.
        use_cuda (bool): Whether to use CUDA.
        save_prefix (str): Prefix for saving model checkpoints.
        output: Output stream for logging.
        args: Argument object containing training parameters.
        dims (int): Number of dimensions (2 or 3).
        to_tensor (bool): Whether to convert to PyTorch tensors.

    Returns:
        nn.Module: The trained classifier.
    """
    num_positive_regions, total_regions = report_data_stats_old(train_images, train_targets, test_images, test_targets)

    ## make the training step method
    if args.num_particles > 0:
        num_micrographs = sum(len(images) for images in train_images)
        # expected particles in training set rather than per micrograph
        expected_num_particles = args.num_particles * num_micrographs

        pi = calculate_pi(expected_num_particles, args.radius, total_regions, dims)

        report('Specified expected number of particle per micrograph = {}'.format(args.num_particles))
        report('With radius = {}'.format(args.radius))
        report('Setting pi = {}'.format(pi))
    else:
        pi = args.pi
        report('pi = {}'.format(pi))

    trainer, criteria, split = make_training_step_method(classifier, num_positive_regions,
                                                         num_positive_regions / total_regions,
                                                         lr=args.learning_rate, l2=args.l2,
                                                         method=args.method, pi=pi, slack=args.slack,
                                                         autoencoder=args.autoencoder)

    ## training parameters
    report(f'minibatch_size={args.minibatch_size}, epoch_size={args.epoch_size}, num_epochs={args.num_epochs}')
    num_workers = mp.cpu_count() if args.num_workers < 0 else args.num_workers  # set num workers to use all CPUs
    balance = None if args.natural else args.minibatch_balance  # ratio of positive to negative in minibatch

    train_iterator, test_iterator = make_data_iterators_old(train_images, train_targets, test_images, test_targets,
                                                            classifier.width, split, args, dims=dims, to_tensor=to_tensor)

    fit_epochs(classifier, criteria, trainer, train_iterator, test_iterator, args.num_epochs,
               save_prefix=save_prefix, use_cuda=use_cuda, output=output)

    return classifier

def train_model(classifier, train_images_path: str, train_targets_path: str, test_images_path: str, test_targets_path: str, use_cuda: bool,
                save_prefix: str, output, args, dims: int = 2, num_sets: int = 1):
    """
    Train the model.

    Args:
        classifier (nn.Module): The classifier model.
        train_images_path (str): Path to training images.
        train_targets_path (str): Path to training targets.
        test_images_path (str): Path to test images.
        test_targets_path (str): Path to test targets.
        use_cuda (bool): Whether to use CUDA.
        save_prefix (str): Prefix for saving model checkpoints.
        output: Output stream for logging.
        args: Argument object containing training parameters.
        dims (int): Number of dimensions (2 or 3).
        num_sets (int): Number of training sets.

    Returns:
        nn.Module: The trained classifier.
    """
    overall_pbar = tqdm(total=num_sets, desc="Overall Progress", position=0)

    for current_set in range(1, num_sets + 1):
        print(f"\nStarting training set {current_set} of {num_sets}")

        report('Starting to report data stats...')
        num_positive_regions, total_regions, num_images = report_data_stats(train_images_path, train_targets_path, test_images_path, test_targets_path,
                                                                             radius=args.radius, dims=dims)

        report('Completed reporting data stats...')
        if args.num_particles > 0:
            expected_num_particles = args.num_particles * num_images
            pi = calculate_pi(expected_num_particles, args.radius, total_regions, dims)
            report('Specified expected number of particles per micrograph = {}'.format(args.num_particles))
            report('With radius = {}'.format(args.radius))
            report('Setting pi = {}'.format(pi))
        else:
            pi = args.pi
            report('pi = {}'.format(pi))

        report('Making training step method...')
        trainer, criteria, split = make_training_step_method(classifier, num_positive_regions,
                                                             num_positive_regions / total_regions,
                                                             lr=args.learning_rate, l2=args.l2,
                                                             method=args.method, pi=pi, slack=args.slack,
                                                             autoencoder=args.autoencoder)

        report(f'minibatch_size={args.minibatch_size}, epoch_size={args.epoch_size}, num_epochs={args.num_epochs}')
        num_workers = mp.cpu_count() if args.num_workers < 0 else args.num_workers
        balance = None if args.natural else args.minibatch_balance

        report('Creating data iterators...')
        train_iterator, test_iterator = make_data_iterators(train_images_path, train_targets_path, classifier.width, split, args.minibatch_size, args.epoch_size,
                                                            test_image_path=test_images_path, test_targets_path=test_targets_path, testing_batch_size=args.test_batch_size,
                                                            num_workers=0, balance=balance, dims=dims, use_cuda=use_cuda, radius=args.radius)

        report('Starting training epochs...')
        fit_epochs(classifier, criteria, trainer, train_iterator, test_iterator, args.num_epochs,
                   save_prefix=save_prefix, use_cuda=use_cuda, output=output, num_sets=num_sets, current_set=current_set)

        overall_pbar.update(1)

    overall_pbar.close()
    return classifier
