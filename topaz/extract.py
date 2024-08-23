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

import topaz.cuda
import topaz.predict
import topaz.utils.files as file_utils
import torch
from topaz.algorithms import match_coordinates, non_maximum_suppression, non_maximum_suppression_3d
from topaz.metrics import average_precision
from topaz.utils.data.loader import load_image
from topaz.model.factory import load_model
from topaz.model.utils import predict_in_patches, get_patches
from tqdm import tqdm

class NonMaximumSuppression: 
    def __init__(self, radius:int, threshold:float, dims:int=2):
        self.radius = radius
        self.threshold = threshold
        self.dims = dims

    def __call__(self, args) -> tuple[str, np.ndarray, np.ndarray]:
        name,score = args
        if self.dims == 2:
            score,coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        elif self.dims == 3:
            score,coords = non_maximum_suppression_3d(score, self.radius*2, threshold=self.threshold)
        return name, score, coords


def nms_iterator(scores:Iterable[np.ndarray], radius:int, threshold:float, pool:multiprocessing.Pool=None, dims:int=2, 
                 patch_size:int=0) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    process = NonMaximumSuppression(radius, threshold, dims=dims)
    
    def crop_translate_coords_scores(scores, coords, patch_size, patch_overlap, x, y, z=None):
        within_patch = np.logical_and(coords >= patch_overlap, coords <= patch_size+patch_overlap)
        within_patch = np.all(within_patch, axis=-1)
        coords = coords[within_patch]
        scores = scores[within_patch]
        # Adjust coordinates to reflect position in full tomogram
        coords[:, -1] += x
        coords[:, -2] += y
        if z is not None:
            coords[:, -3] += z
        return scores, coords
    
    if pool is not None:
        # for name,score,coords in pool.imap_unordered(process, scores): # TODO: multiprocessing was removed!
        for name, score in scores:
            original_shape = score.shape
            y, x = original_shape[-2:]
            z = original_shape[-3] if dims==3 else None
            assert score.ndim == dims, f"Expected {dims}D score array, but got shape {score.shape}"

            coords_list = []
            scores_list = []
            
            if patch_size:
                patch_overlap = patch_size // 2                
                patches = get_patches(score, patch_size)
                step_size = patch_size - patch_overlap * 2 # good crop size
                # process each patch
                patch_idx = 0
                for i in range(0, y, step_size):
                    for j in range(0, x, step_size):
                        if dims==3:
                            for k in range(0, z, step_size):
                                patch = patches[patch_idx]
                                _, patch_score, patch_coords = process((name, patch))
                                # find coordinates within patch boundaries
                                patch_score, patch_coords = crop_translate_coords_scores(patch_score, patch_coords, 
                                                                                         patch_size, patch_overlap, j, i, k)
                                scores_list.append(patch_score)
                                coords_list.append(patch_coords)
                                patch_idx += 1
                        else:
                            patch = patches[patch_idx]
                            _, patch_score, patch_coords = process((name, patch))
                            patch_score, patch_coords = crop_translate_coords_scores(patch_score, patch_coords, 
                                                                                     patch_size, patch_overlap, j, i)
                            scores_list.append(patch_score)
                            coords_list.append(patch_coords)
                            patch_idx += 1
                # combine patch results
                coords = np.concatenate(coords_list, axis=0) if coords_list else np.array([])
                scores = np.concatenate(scores_list, axis=0) if scores_list else np.array([])           
            else:
                _, scores, coords = process((name, score))
            
            yield name, scores, coords
    else:
        nms_nD = non_maximum_suppression if dims == 2 else non_maximum_suppression_3d
        for name, score in scores:
            score, coords = nms_nD(score, radius, threshold=threshold)
            yield name, score, coords


def iterate_score_target_pairs(scores, targets:pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for image_name,score in scores.items():
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values
        yield score,target


class ExtractMatches:
    def __init__(self, radius:float, threshold:float, match_radius:Union[float,None], dims:int=2):
        self.radius = radius
        self.threshold = threshold
        self.match_radius = match_radius
        self.dims = dims

    def __call__(self, args:tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, int]:
        score,target = args
        if self.dims == 2: 
            score,coords = non_maximum_suppression(score, self.radius, threshold=self.threshold)
        elif self.dims == 3:
            score,coords = non_maximum_suppression_3d(score, self.radius*2, threshold=self.threshold)
        
        radius = self.radius if (self.match_radius is None) else self.match_radius
        assignment, dist = match_coordinates(target, coords, radius)

        mse = np.sum(dist[assignment==1]**2)
        
        return assignment, score, mse, len(target)


def extract_auprc(targets:np.ndarray, scores:np.ndarray, radius:float, threshold:float, match_radius:float=None, 
                  pool:multiprocessing.Pool=None, dims:int=2) -> tuple[float, float, int, int]:
    N = 0
    mse = 0
    hits = []
    preds = []

    if pool is not None:
        process = ExtractMatches(radius, threshold, match_radius, dims=dims)
        iterator = iterate_score_target_pairs(scores, targets)
        for assignment,score,this_mse,n in pool.imap_unordered(process, iterator):
            mse += this_mse
            hits.append(assignment)
            preds.append(score)
            N += n
    else:
        for score,target in iterate_score_target_pairs(scores, targets):
            if dims == 2:
                score,coords = non_maximum_suppression(score, radius, threshold=threshold)
            elif dims == 3:
                score,coords = non_maximum_suppression_3d(score, radius*2, threshold=threshold)
                       
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
    def __init__(self, targets:np.ndarray, target_scores:np.ndarray, threshold:float, match_radius:float, dims:int=2):
        self.targets = targets
        self.target_scores = target_scores
        self.threshold = threshold
        self.match_radius = match_radius
        self.dims = 2

    def __call__(self, r):
        auprc, rmse, recall, n = extract_auprc(self.targets, self.target_scores, r, self.threshold, match_radius=self.match_radius, dims=self.dims)
        return r, auprc, rmse, recall, n


def find_opt_radius(targets:np.ndarray, target_scores:np.ndarray, threshold:float, lo:int=0, hi:int=200, step:int=10, 
                    match_radius:int=None, pool:multiprocessing.Pool=None, dims:int=2) -> tuple[int, float]:

    auprc = np.zeros(hi+1) - 1
    process = Process(targets, target_scores, threshold, match_radius, dims=dims)

    if pool is not None:
        for r,au,rmse,recall,n in pool.imap_unordered(process, range(lo, hi+1, step)):
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))
    else:
        for r in range(lo, hi+1, step):
            _,au,rmse,recall,n = process(r)
            auprc[r] = au
            print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(r, au, rmse, recall, n))

    r = np.argmax(auprc)
    return r, auprc[r]


def stream_images(paths:List[str]) -> Iterator[np.ndarray]:
    for path in paths:
        image = load_image(path, make_image=False, return_header=False)
        yield image


def get_available_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    return 0


def calculate_chunk_size(image_shape, available_memory):
    mem_per_slice = image_shape[1] * image_shape[2] * 4 * 2
    return max(1, int(available_memory / mem_per_slice))


def score_images(model:Union[torch.nn.Module, str], paths:List[str], device:int=-1, patch_size:int=0, batch_size:int=1) -> Iterator[np.ndarray]:
    if model is not None and model != 'none': # score each image with the model
        ## set the device
        use_cuda = topaz.cuda.set_device(device)
        model = load_model(model)
        model.eval()
        model.fill()
        if use_cuda:
            model.cuda()
        
        # for path in tqdm(paths, desc="Scoring tomograms", unit="tomogram"):
        for path in paths:
            # print(f"\Scoring tomogram: {path}")
            start_time = time.time()
            image = load_image(path, make_image=False, return_header=False)
            image = torch.from_numpy(image.copy()).float()
            image = image[...,:150,:150,:150]
            image = image.unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
            # print(f"Image shape: {image.shape} FIX CROPPING LATER!!!!")
            original_shape = image.shape
            is_3d = len(original_shape) == 3
            
            if patch_size:
                patch_overlap = patch_size // 2
                scores = predict_in_patches(model, image, patch_size*2, patch_overlap=patch_overlap, is_3d=is_3d, use_cuda=use_cuda)
                scores = scores[0,0] # remove added dimensions
            else:
                if use_cuda:
                    image = image.cuda()
                scores = model(image).data[0,0].cpu().numpy()
            
            # print(f"Tomogram {path} scoring completed in {time.time() - start_time:.2f} seconds")
            yield path, scores            
    else:
        # TODO: scoring without model? check if 2d and use pretrained
        for path in tqdm(paths, desc="Loading tomograms", unit="tomogram"):
            image = load_image(path, make_image=False, return_header=False)
            yield path, image

            
def stream_inputs(f:TextIO) -> Iterator[str]:
    for line in f:
        line = line.strip()
        if len(line) > 0:
            yield line


def extract_particles(paths:List[str], model:Union[torch.nn.Module, str], device:int, batch_size:int, threshold:float, radius:int, num_workers:int, targets:str, min_radius:int, max_radius:int, step:int, match_radius:int,
                      patch_size, only_validate:bool, output:str, per_micrograph:bool, suffix:str, out_format:str, up_scale:float, down_scale:float, dims=2):
    paths = list(stream_inputs(sys.stdin) if len(paths) == 0 else paths)
    # print(f"Total number of tomograms to process: {len(paths)}")
    stream = score_images(model, paths, device=device, patch_size=patch_size, batch_size=batch_size)

    radius = radius if radius is not None else -1

    num_workers = multiprocessing.cpu_count() if num_workers < 0 else num_workers
    pool = multiprocessing.Pool(num_workers) if num_workers > 0 else None

    if not per_micrograph and output:
        if os.path.isdir(output):
            output = os.path.join(output, "extracted_particles.txt")
        with open(output, 'w') as f:
            print('image_name\tx_coord\ty_coord\tz_coord\tscore', file=f)

    # Create COORDS directory
    coords_dir = os.path.join(os.path.dirname(output), "COORDS")
    os.makedirs(coords_dir, exist_ok=True)

    for path, scores in stream:
        # print(f"\nExtracting particles from: {path}")
        start_time = time.time()
        name, score, coords = next(nms_iterator([(path, scores)], radius, threshold, pool=pool, dims=dims))

        if not only_validate:
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            
            scale = up_scale/down_scale
            if scale != 1:
                coords = np.round(coords*scale).astype(int)

            print(f"Number of particles extracted: {len(coords)}")

            # Save IMOD .coords file
            coords_file = os.path.join(coords_dir, f"{name}.coords")
            with open(coords_file, 'w') as f:
                for coord in coords:
                    # IMOD .coords format: x y z (one coordinate per line)
                    print(f"{coord[0]} {coord[1]} {coord[2]}", file=f)
            # print(f"IMOD .coords file saved to: {coords_file}")

            if per_micrograph:
                table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 'z_coord': coords[:,2], 'score': score})
                out_path, ext = os.path.splitext(path)
                out_path = out_path + suffix + '.' + out_format
                with open(out_path, 'w') as f:
                    file_utils.write_table(f, table, format=out_format, image_ext=ext)
                # print(f"Results written to: {out_path}")
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
        # print(f"Particle extraction completed in {end_time - start_time:.2f} seconds")

        del score
        del coords
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if pool is not None:
        pool.close()
        pool.join()

    # print("\nParticle extraction completed for all tomograms.")
    # print(f"IMOD .coords files saved in: {coords_dir}")