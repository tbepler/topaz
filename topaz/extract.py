#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import multiprocessing
import os
import sys
import time
from typing import List, Union, Iterator, TextIO, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import topaz.cuda
import topaz.predict
import topaz.utils.files as file_utils
import torch
import math
from topaz.algorithms import match_coordinates, non_maximum_suppression, non_maximum_suppression_3d
from topaz.metrics import average_precision
from topaz.utils.data.loader import load_image
from topaz.model.factory import load_model
from tqdm import tqdm

class NonMaximumSuppression:
    """
    Class for performing non-maximum suppression on 2D or 3D data.
    """
    def __init__(self, radius:int, threshold:float, dims:int=2):
        """
        Initialize the NonMaximumSuppression class.

        Args:
            radius (int): Radius for suppression.
            threshold (float): Threshold for suppression.
            dims (int): Number of dimensions (2 or 3).
        """
        self.radius = radius
        self.threshold = threshold
        self.dims = dims

    def __call__(self, args) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Perform non-maximum suppression on the input data.

        Args:
            args (tuple): Tuple containing name and score.

        Returns:
            tuple: Tuple containing name, suppressed score, and coordinates.
        """
        name, score = args
        if self.dims == 2:
            score, coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        elif self.dims == 3:
            score, coords = non_maximum_suppression_3d(score, self.radius*2, threshold=self.threshold)
        return name, score, coords

def nms_iterator(scores:Iterable[np.ndarray], radius:int, threshold:float, pool:multiprocessing.Pool=None, dims:int=3, chunk_size:int=128) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """
    Iterator for performing non-maximum suppression on a batch of scores.

    Args:
        scores (Iterable[np.ndarray]): Iterable of score arrays.
        radius (int): Radius for suppression.
        threshold (float): Threshold for suppression.
        pool (multiprocessing.Pool, optional): Multiprocessing pool for parallel processing.
        dims (int): Number of dimensions (2 or 3).
        chunk_size (int): Size of chunks for processing large 3D volumes.

    Yields:
        tuple: Tuple containing name, suppressed score, and coordinates.
    """
    process = NonMaximumSuppression(radius, threshold, dims=dims)
    if pool is not None:
        for name, score in scores:
            print(f"Debug: Initial score shape: {score.shape}")
            original_shape = score.shape
            if score.ndim != 3:
                print(f"Warning: Expected 3D score array, but got shape {score.shape}. Attempting to reshape...")
                if score.ndim == 1:
                    side = int(np.cbrt(score.size))
                    score = score.reshape(side, side, side)
                elif score.ndim == 2:
                    score = score.reshape(1, *score.shape)
                print(f"Reshaped score array to {score.shape}")

            coords_list = []
            scores_list = []
            for z in range(0, score.shape[0], chunk_size):
                for y in range(0, score.shape[1], chunk_size):
                    for x in range(0, score.shape[2], chunk_size):
                        z_end = min(z + chunk_size, score.shape[0])
                        y_end = min(y + chunk_size, score.shape[1])
                        x_end = min(x + chunk_size, score.shape[2])
                        
                        chunk = score[z:z_end, y:y_end, x:x_end]
                        _, chunk_score, chunk_coords = process((name, chunk))
                        
                        # Adjust coordinates to reflect position in full tomogram
                        chunk_coords[:, 0] += x  # x coordinate
                        chunk_coords[:, 1] += y  # y coordinate
                        chunk_coords[:, 2] += z  # z coordinate
                        
                        coords_list.append(chunk_coords)
                        scores_list.append(chunk_score)
            
            coords = np.concatenate(coords_list, axis=0) if coords_list else np.array([])
            score = np.concatenate(scores_list, axis=0) if scores_list else np.array([])
            
            print(f"Debug: Final score shape: {score.shape}, coords shape: {coords.shape}")
            
            if coords.size > 0:
                # Clip coordinates to ensure they're within the tomogram boundaries
                coords = np.clip(coords, 0, np.array([original_shape[2]-1, original_shape[1]-1, original_shape[0]-1]))
            
            yield name, score, coords
    else:
        for name, score in scores:
            print(f"Debug: Initial score shape: {score.shape}")
            original_shape = score.shape
            if score.ndim != 3:
                print(f"Warning: Expected 3D score array, but got shape {score.shape}. Attempting to reshape...")
                if score.ndim == 1:
                    side = int(np.cbrt(score.size))
                    score = score.reshape(side, side, side)
                elif score.ndim == 2:
                    score = score.reshape(1, *score.shape)
                print(f"Reshaped score array to {score.shape}")

            score, coords = non_maximum_suppression_3d(score, radius*2, threshold=threshold)
            
            print(f"Debug: Final score shape: {score.shape}, coords shape: {coords.shape}")
            
            if coords.size > 0:
                coords = np.clip(coords, 0, np.array([original_shape[2]-1, original_shape[1]-1, original_shape[0]-1]))
            
            yield name, score, coords

def iterate_score_target_pairs(scores, targets:pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Iterator for score-target pairs.

    Args:
        scores: Dictionary of scores.
        targets (pd.DataFrame): DataFrame of target coordinates.

    Yields:
        tuple: Tuple containing score and target coordinates.
    """
    for image_name, score in scores.items():
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values
        yield score, target

class ExtractMatches:
    """
    Class for extracting matches between predicted and target coordinates.
    """
    def __init__(self, radius:float, threshold:float, match_radius:Union[float,None], dims:int=2):
        """
        Initialize the ExtractMatches class.

        Args:
            radius (float): Radius for non-maximum suppression.
            threshold (float): Threshold for non-maximum suppression.
            match_radius (float or None): Radius for matching coordinates.
            dims (int): Number of dimensions (2 or 3).
        """
        self.radius = radius
        self.threshold = threshold
        self.match_radius = match_radius
        self.dims = dims

    def __call__(self, args:tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Extract matches between predicted and target coordinates.

        Args:
            args (tuple): Tuple containing score and target.

        Returns:
            tuple: Tuple containing assignment, score, mean squared error, and number of targets.
        """
        score, target = args
        if self.dims == 2: 
            score, coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        elif self.dims == 3:
            score, coords = non_maximum_suppression_3d(score, self.radius*2, threshold=self.threshold)
        
        radius = self.radius if (self.match_radius is None) else self.match_radius
        assignment, dist = match_coordinates(target, coords, radius)

        mse = np.sum(dist[assignment==1]**2)
        
        return assignment, score, mse, len(target)

def extract_auprc(targets:np.ndarray, scores:np.ndarray, radius:float, threshold:float, match_radius:float=None,
                  pool:multiprocessing.Pool=None, dims:int=2) -> tuple[float, float, int, int]:
    """
    Extract Area Under Precision-Recall Curve (AUPRC) and related metrics.

    Args:
        targets (np.ndarray): Array of target coordinates.
        scores (np.ndarray): Array of scores.
        radius (float): Radius for non-maximum suppression.
        threshold (float): Threshold for non-maximum suppression.
        match_radius (float, optional): Radius for matching coordinates.
        pool (multiprocessing.Pool, optional): Multiprocessing pool for parallel processing.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        tuple: Tuple containing AUPRC, RMSE, number of hits, and total number of targets.
    """
    N = 0
    mse = 0
    hits = []
    preds = []

    if pool is not None:
        process = ExtractMatches(radius, threshold, match_radius, dims=dims)
        iterator = iterate_score_target_pairs(scores, targets)
        for assignment, score, this_mse, n in pool.imap_unordered(process, iterator):
            mse += this_mse
            hits.append(assignment)
            preds.append(score)
            N += n
    else:
        for score, target in iterate_score_target_pairs(scores, targets):
            if dims == 2:
                score, coords = non_maximum_suppression(score, radius, threshold=threshold)
            elif dims == 3:
                score, coords = non_maximum_suppression_3d(score, radius*2, threshold=threshold)
                       
            radius_to_use = radius if (match_radius is None) else match_radius
            assignment, dist = match_coordinates(target, coords, radius_to_use)
            
            mse += np.sum(dist[assignment==1]**2)
            hits.append(assignment)
            preds.append(score)
            N += len(target)

    hits = np.concatenate(hits, 0)
    preds = np.concatenate(preds, 0)
    auprc = average_precision(hits, preds, N=N)

    rmse = np.sqrt(mse/hits.sum())

    return auprc, rmse, int(hits.sum()), N

class Process:
    """
    Class for processing targets and scores to extract metrics.
    """
    def __init__(self, targets:np.ndarray, target_scores:np.ndarray, threshold:float, match_radius:float, dims:int=2):
        """
        Initialize the Process class.

        Args:
            targets (np.ndarray): Array of target coordinates.
            target_scores (np.ndarray): Array of target scores.
            threshold (float): Threshold for non-maximum suppression.
            match_radius (float): Radius for matching coordinates.
            dims (int): Number of dimensions (2 or 3).
        """
        self.targets = targets
        self.target_scores = target_scores
        self.threshold = threshold
        self.match_radius = match_radius
        self.dims = 2

    def __call__(self, r):
        """
        Process targets and scores for a given radius.

        Args:
            r (int): Radius for non-maximum suppression.

        Returns:
            tuple: Tuple containing radius, AUPRC, RMSE, recall, and number of targets.
        """
        auprc, rmse, recall, n = extract_auprc(self.targets, self.target_scores, r, self.threshold, match_radius=self.match_radius, dims=self.dims)
        return r, auprc, rmse, recall, n

def find_opt_radius(targets:np.ndarray, target_scores:np.ndarray, threshold:float, lo:int=0, hi:int=200, step:int=10,
                    match_radius:int=None, pool:multiprocessing.Pool=None, dims:int=2) -> tuple[int, float]:
    """
    Find the optimal radius for non-maximum suppression.

    Args:
        targets (np.ndarray): Array of target coordinates.
        target_scores (np.ndarray): Array of target scores.
        threshold (float): Threshold for non-maximum suppression.
        lo (int): Lower bound for radius search.
        hi (int): Upper bound for radius search.
        step (int): Step size for radius search.
        match_radius (int, optional): Radius for matching coordinates.
        pool (multiprocessing.Pool, optional): Multiprocessing pool for parallel processing.
        dims (int): Number of dimensions (2 or 3).

    Returns:
        tuple: Tuple containing optimal radius and corresponding AUPRC.
    """
    auprc = np.zeros(hi+1) - 1
    process = Process(targets, target_scores, threshold, match_radius, dims=dims)

    if pool is not None:
        for r, au, rmse, recall, n in pool.imap_unordered(process, range(lo, hi+1, step)):
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))
    else:
        for r in range(lo, hi+1, step):
            _, au, rmse, recall, n = process(r)
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))

    r = np.argmax(auprc)
    return r, auprc[r]

def stream_images(paths:List[str]) -> Iterator[np.ndarray]:
    """
    Stream images from given paths.

    Args:
        paths (List[str]): List of image paths.

    Yields:
        np.ndarray: Loaded image.
    """
    for path in paths:
        image = load_image(path, make_image=False, return_header=False)
        yield image

def get_available_gpu_memory():
    """
    Get available GPU memory.

    Returns:
        int: Available GPU memory in bytes.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    return 0

def calculate_chunk_size(image_shape, available_memory):
    """
    Calculate chunk size based on available memory.

    Args:
        image_shape (tuple): Shape of the image.
        available_memory (int): Available memory in bytes.

    Returns:
        int: Calculated chunk size.
    """
    mem_per_slice = image_shape[1] * image_shape[2] * 4 * 2
    return max(1, int(available_memory / mem_per_slice))

def score_images(model:str, paths:List[str], device:int=-1, batch_size:int=1) -> Iterator[np.ndarray]:
    """
    Score images using the given model.

    Args:
        model (str): Path to the model or 'none' for no model.
        paths (List[str]): List of image paths.
        device (int): GPU device index (-1 for CPU).
        batch_size (int): Batch size for processing.

    Yields:
        tuple: Tuple containing path, scores, and original shape.
    """
    if model is not None and model != 'none':
        use_cuda = topaz.cuda.set_device(device)
        model = load_model(model)
        model.eval()
        model.fill()
        if use_cuda:
            model.cuda()
        
        def process_patch(patch):
            with torch.no_grad():
                patch = torch.from_numpy(patch).float()
                if use_cuda:
                    patch = patch.cuda()
                result = model(patch).squeeze(1).cpu().numpy()
                if use_cuda:
                    torch.cuda.empty_cache()
                return result

        for path in tqdm(paths, desc="Processing tomograms", unit="tomogram"):
            print(f"\nProcessing tomogram: {path}")
            start_time = time.time()
            image = load_image(path, make_image=False, return_header=False)
            original_shape = image.shape
            print(f"Tomogram shape: {original_shape}")
            
            # Define patch size and overlap
            patch_size = (32, 128, 128)  # (z, y, x)
            overlap = (8, 32, 32)  # 25% overlap
            
            scores = np.zeros(original_shape, dtype=np.float32)
            weights = np.zeros(original_shape, dtype=np.float32)
            
            total_patches = ((original_shape[0] - overlap[0]) // (patch_size[0] - overlap[0]) + 1) * \
                            ((original_shape[1] - overlap[1]) // (patch_size[1] - overlap[1]) + 1) * \
                            ((original_shape[2] - overlap[2]) // (patch_size[2] - overlap[2]) + 1)
            
            with tqdm(total=total_patches, desc="Processing patches", unit="patch") as pbar:
                for z in range(0, original_shape[0] - overlap[0], patch_size[0] - overlap[0]):
                    for y in range(0, original_shape[1] - overlap[1], patch_size[1] - overlap[1]):
                        for x in range(0, original_shape[2] - overlap[2], patch_size[2] - overlap[2]):
                            z_end = min(z + patch_size[0], original_shape[0])
                            y_end = min(y + patch_size[1], original_shape[1])
                            x_end = min(x + patch_size[2], original_shape[2])
                            
                            patch = image[z:z_end, y:y_end, x:x_end]
                            patch = patch[None, None, ...]  # Add batch and channel dimensions
                            
                            patch_scores = process_patch(patch)
                            
                            # Apply Gaussian smoothing to the patch
                            patch_scores = gaussian_filter(patch_scores[0], sigma=1)
                            
                            # Create a weight array for blending
                            weight = np.ones_like(patch_scores)
                            weight = gaussian_filter(weight, sigma=2)
                            
                            scores[z:z_end, y:y_end, x:x_end] += patch_scores * weight
                            weights[z:z_end, y:y_end, x:x_end] += weight
                            
                            del patch
                            if use_cuda:
                                torch.cuda.empty_cache()
                            pbar.update(1)
            
            # Normalize the scores
            scores /= np.maximum(weights, 1e-10)
            
            end_time = time.time()
            print(f"Tomogram processing completed in {end_time - start_time:.2f} seconds")
            yield path, scores, original_shape
            
            del scores
            del weights
            if use_cuda:
                torch.cuda.empty_cache()
    else:
        for path in tqdm(paths, desc="Loading tomograms", unit="tomogram"):
            image = load_image(path, make_image=False, return_header=False)
            yield path, image, image.shape
            
def stream_inputs(f:TextIO) -> Iterator[str]:
    """
    Stream inputs from a file or stdin.

    Args:
        f (TextIO): File-like object to read from.

    Yields:
        str: Each non-empty line from the input.
    """
    for line in f:
        line = line.strip()
        if len(line) > 0:
            yield line

def extract_particles(paths:List[str], model:Union[torch.nn.Module, str], device:int, batch_size:int, threshold:float, radius:int, num_workers:int, targets:str, min_radius:int, max_radius:int, step:int, match_radius:int,
                      only_validate:bool, output:str, per_micrograph:bool, suffix:str, out_format:str, up_scale:float, down_scale:float, dims=3):
    """
    Extract particles from tomograms.

    Args:
        paths (List[str]): List of tomogram paths.
        model (Union[torch.nn.Module, str]): Model for scoring or path to model.
        device (int): GPU device index.
        batch_size (int): Batch size for processing.
        threshold (float): Threshold for particle detection.
        radius (int): Radius for non-maximum suppression.
        num_workers (int): Number of worker processes.
        targets (str): Path to target coordinates file.
        min_radius (int): Minimum radius for optimization.
        max_radius (int): Maximum radius for optimization.
        step (int): Step size for radius optimization.
        match_radius (int): Radius for matching coordinates.
        only_validate (bool): Only perform validation without extraction.
        output (str): Output file path.
        per_micrograph (bool): Whether to output per micrograph.
        suffix (str): Suffix for output files.
        out_format (str): Output format.
        up_scale (float): Upscaling factor.
        down_scale (float): Downscaling factor.
        dims (int): Number of dimensions (2 or 3).
    """
    paths = list(stream_inputs(sys.stdin) if len(paths) == 0 else paths)
    print(f"Total number of tomograms to process: {len(paths)}")
    stream = score_images(model, paths, device=device, batch_size=batch_size)
    radius = radius if radius is not None else -1
    num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers) if num_workers > 0 else None

    if not per_micrograph and output:
        if os.path.isdir(output):
            output = os.path.join(output, "extracted_particles.txt")
        with open(output, 'w') as f:
            print('image_name\tx_coord\ty_coord\tz_coord\tscore', file=f)

    # Create COORDS directory
    coords_dir = os.path.join(os.path.dirname(output), "COORDS")
    os.makedirs(coords_dir, exist_ok=True)

    for path, score, original_shape in stream:
        print(f"\nExtracting particles from: {path}")
        start_time = time.time()
        name, score, coords = next(nms_iterator([(path, score)], radius, threshold, pool=pool, dims=dims))

        if not only_validate:
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            
            scale = up_scale/down_scale
            if scale != 1:
                coords = np.round(coords*scale).astype(int)

            print(f"Number of particles extracted: {len(coords)}")

            # Ensure coordinates are within tomogram boundaries
            coords = np.clip(coords, 0, np.array(original_shape)[[2, 1, 0]] - 1)

            # Save IMOD .coords file
            coords_file = os.path.join(coords_dir, f"{name}.coords")
            with open(coords_file, 'w') as f:
                for coord in coords:
                    # IMOD .coords format: x y z (one coordinate per line)
                    print(f"{coord[0]} {coord[1]} {coord[2]}", file=f)
            print(f"IMOD .coords file saved to: {coords_file}")

            if per_micrograph:
                table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 'z_coord': coords[:,2], 'score': score})
                out_path, ext = os.path.splitext(path)
                out_path = out_path + suffix + '.' + out_format
                with open(out_path, 'w') as f:
                    file_utils.write_table(f, table, format=out_format, image_ext=ext)
                print(f"Results written to: {out_path}")
            else:
                out_file = output if output else sys.stdout
                with open(out_file, 'a') if isinstance(out_file, str) else out_file as f:
                    for i in range(len(score)):
                        print(f"{name}\t{coords[i,0]}\t{coords[i,1]}\t{coords[i,2]}\t{score[i]}", file=f)
                if output:
                    print(f"Results appended to: {output}")
                else:
                    print("Results printed to stdout")

        end_time = time.time()
        print(f"Particle extraction completed in {end_time - start_time:.2f} seconds")

        del score
        del coords
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if pool is not None:
        pool.close()
        pool.join()

    print("\nParticle extraction completed for all tomograms.")
    print(f"IMOD .coords files saved in: {coords_dir}")
