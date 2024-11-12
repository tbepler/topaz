#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import multiprocessing
import os
import sys
import time
from typing import List, Union, Iterator, TextIO, Iterable, Tuple

import numpy as np
import pandas as pd

import topaz.cuda
import topaz.predict
import topaz.utils.files as file_utils
import torch
from topaz.algorithms import match_coordinates, non_maximum_suppression, non_maximum_suppression_3d
from topaz.metrics import average_precision
from topaz.utils.data.loader import load_image
from topaz.utils.printing import report
from topaz.model.factory import load_model
from topaz.model.utils import predict_in_patches, get_patches
from tqdm import tqdm

class NonMaximumSuppression: 
    def __init__(self, radius:int, threshold:float, dims:int=2, patch_size=64, patch_overlap=32, verbose:bool=False):
        self.radius = radius
        self.threshold = threshold
        self.dims = dims
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.verbose = verbose

    def __call__(self, args) -> Tuple[str, np.ndarray, np.ndarray]:
        nms = non_maximum_suppression if self.dims == 2 else non_maximum_suppression_3d
        name,score = args
        if self.verbose:
            report(f'Scoring {name}')
        original_shape = score.shape
        y, x = original_shape[-2:]
        z = original_shape[-3] if self.dims==3 else None   
        if self.patch_size:
            scores_list = []
            coords_list = []
            patches = get_patches(score, self.patch_size, self.patch_overlap, is_3d=(self.dims==3))
            step_size = self.patch_size - self.patch_overlap * 2 # good crop size
            # process each patch
            patch_idx = 0
            # TODO: this would be a good place for a progress bar
            for i in range(0, y, step_size):
                for j in range(0, x, step_size):
                    if self.dims==3:
                        for k in range(0, z, step_size):
                            patch = patches[patch_idx]
                            _, patch_score, patch_coords = nms(patch, r=self.radius, threshold=self.threshold)
                            # find coordinates within patch boundaries, and adjust to full tomogram coordinates
                            patch_score, patch_coords = crop_translate_coords_scores(patch_score, patch_coords, self.patch_size, 
                                                                                     self.patch_overlap, j, i, k)
                            scores_list.append(patch_score)
                            coords_list.append(patch_coords)
                            patch_idx += 1
                            # print(f'Processed patch {patch_idx} of {len(patches)}', end='\r', flush=True)
                    else:
                        patch = patches[patch_idx]
                        _, patch_score, patch_coords = nms(patch, r=self.radius, threshold=self.threshold)
                        patch_score, patch_coords = crop_translate_coords_scores(patch_score, patch_coords, self.patch_size, 
                                                                                 self.patch_overlap, j, i)
                        scores_list.append(patch_score)
                        coords_list.append(patch_coords)
                        patch_idx += 1
                        # print(f'Processed patch {patch_idx} of {len(patches)}', end='\r')
            score = np.concatenate(scores_list, axis=0) if scores_list else np.array([])
            coords = np.concatenate(coords_list, axis=0) if coords_list else np.array([])  
        else:
            score,coords = nms(score, self.radius, threshold=self.threshold)
        return name, score, coords


def crop_translate_coords_scores(scores, coords, patch_size, patch_overlap, x, y, z=None):
    within_patch = np.logical_and(patch_overlap <= coords, coords < patch_size+patch_overlap)
    within_patch = np.all(within_patch, axis=-1)
    coords = coords[within_patch]
    scores = scores[within_patch]
    # Adjust coordinates to reflect position in full tomogram
    coords[:, -1] += x
    coords[:, -2] += y
    if z is not None:
        coords[:, -3] += z
    return scores, coords


def nms_iterator(paths_scores:Iterable[np.ndarray], radius:int, threshold:float, pool:multiprocessing.Pool=None, dims:int=2, 
                 patch_size:int=0, patch_overlap:int=0, verbose:bool=False) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    # create the process, can be patched or not, 2d or 3d
    process = NonMaximumSuppression(radius, threshold, dims=dims, patch_size=patch_size, patch_overlap=patch_overlap, verbose=verbose)
    # parallelize on CPU at the image level
    if pool is not None:
        for name,score,coords in pool.imap_unordered(process, paths_scores):
            yield name, score, coords
    else:
        for name, score in paths_scores:
            name, score, coords = process((name, score))
            yield name, score, coords


def iterate_score_target_pairs(scores, targets:pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for image_name,score in scores.items():
        target = targets.loc[targets.image_name == image_name][['x_coord', 'y_coord']].values
        yield score,target


class ExtractMatches:
    def __init__(self, radius:float, threshold:float, match_radius:Union[float,None], dims:int=2):
        self.radius = radius
        self.threshold = threshold
        self.match_radius = match_radius
        self.dims = dims

    def __call__(self, args:Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, int]:
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
                  pool:multiprocessing.Pool=None, dims:int=2) -> Tuple[float, float, int, int]:
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
                    match_radius:int=None, pool:multiprocessing.Pool=None, dims:int=2) -> Tuple[int, float]:

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


def score_images(model:Union[torch.nn.Module, str], paths:Union[List[str], Iterable[str]], device:int=-1, patch_size:int=0, batch_size:int=1) -> Iterator[np.ndarray]:
    if model is not None and model != 'none': # score each image with the model
        ## set the device
        use_cuda = topaz.cuda.set_device(device)
        model = load_model(model)
        model.eval()
        model.fill()
        if use_cuda:
            model.cuda()
        
        for path in paths:
            image = load_image(path, make_image=False, return_header=False)
            original_shape = image.shape
            is_3d = len(original_shape) == 3
            image = torch.from_numpy(image.copy()).float()
            image = image.unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
            
            if patch_size:
                patch_overlap = model.width // 2 # patch_overlap == receptive field // 2
                # TODO: does this need further refactoring?
                scores = predict_in_patches(model, image, patch_size+2*patch_overlap, is_3d=is_3d, use_cuda=use_cuda)
                scores = scores[0,0] # remove added dimensions
            else:
                if use_cuda:
                    image = image.cuda()
                scores = model(image).data[0,0].cpu().numpy()
            
            yield path, scores            
    else:
        # TODO: scoring without model? check if 2d and use pretrained
        for path in paths:
            image = load_image(path, make_image=False, return_header=False)
            yield path, image

            
def stream_inputs(f:TextIO) -> Iterator[str]:
    for line in f:
        line = line.strip()
        if len(line) > 0:
            yield line


def extract_particles(paths:List[str], model:Union[torch.nn.Module, str], device:int, batch_size:int, threshold:float, radius:int, num_workers:int, targets:str, 
                      min_radius:int, max_radius:int, step:int, match_radius:int,patch_size, only_validate:bool, output:str, per_micrograph:bool, suffix:str, 
                      out_format:str, up_scale:float, down_scale:float, dims=2, verbose:bool=False):
    report('Beginning extraction')
    # score the images lazily with a generator
    paths = stream_inputs(sys.stdin) if len(paths) == 0 else paths # no paths, read from stdin
    
    # generator of images and their scores
    stream = score_images(model, paths, device=device, patch_size=patch_size, batch_size=batch_size)

    # set the number of workers
    num_workers = multiprocessing.cpu_count() if num_workers < 0 else num_workers
    pool = multiprocessing.Pool(num_workers) if num_workers > 0 else None

    # extract coordinates from scored images
    radius = radius if radius is not None else -1

    # if no radius is set, we choose the radius based on targets provided
    if radius < 0 and targets is not None: # set the radius to optimize AUPRC of the targets
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        ## find radius maximizing AUPRC
        report('Finding optimal radius for extraction')
        radius, auprc = find_opt_radius(targets, target_scores, threshold, lo=min_radius, hi=max_radius,
                                        step=step, match_radius=match_radius, pool=pool, dims=dims)
        report(f'Optimal radius found: {radius} with AUPRC: {auprc}')

    elif targets is not None:
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        # calculate AUPRC for radius
        au, rmse, recall, n = extract_auprc(targets, target_scores, radius, threshold, match_radius=match_radius, 
                                            pool=pool, dims=dims)
        print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(radius, au, rmse, recall, n))
    
    elif radius < 0:
        # must have targets if radius < 0
        raise Exception('Must specify targets for choosing the extraction radius if extraction radius is not provided')

    # extract all particles from scored images
    if not only_validate:
        scale = up_scale/down_scale
        
        # prepare output file or directory
        if not per_micrograph:
            # if output is a directory, create a file within
            output = sys.path.join(output, 'extracted_particles.txt') \
                if (output is not None and os.path.isdir(output)) else output
            # open file (or stdout) for writing
            f = sys.stdout if (output is None) else open(output, 'w')
            z_string = '\tz_coord' if dims == 3 else ''
            print(f'image_name\tx_coord\ty_coord{z_string}\tscore', file=f)
        elif not os.path.isdir(output):
            # output isn't a directory, so create one
            os.makedirs(os.path.dirname(output), exist_ok=True)
            output_dir = os.path.join(os.path.dirname(output), 'COORDS')
        else:
            # already a directory
            output_dir = output
                  
        # extract coordinates using radius 
        for path,score,coords in nms_iterator(stream, radius, threshold, pool=pool, dims=dims, verbose=verbose):
            # get the name of the image w/o extension
            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)
            if verbose:
                report(f'Extracted {len(score)} particles from {name}')
            # scale the coordinates
            coords = np.round(coords*scale).astype(int) if scale != 1 else coords
            if per_micrograph:
                out_path = os.path.join(output_dir, name + suffix + '.' + out_format)
                if dims == 2:
                    table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 
                                          'score': score})
                else:
                    table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 
                                          'z_coord': coords[:,2], 'score': score})
                with open(out_path, 'w') as f:
                    file_utils.write_table(f, table, format=out_format, image_ext=ext)
            else:
                for i in range(len(score)):
                    z_coord = f'\t{coords[i,2]}' if dims == 3 else ''
                    print(f'{name}\t{coords[i,0]}\t{coords[i,1]}{z_coord}\t{score[i]}', file=f)   
        
        # close a file if it was opened
        try:
            f.close()
        except:
            pass
    
    # Close multiprocessing pool
    if num_workers > 0:
        pool.close()
        pool.join()
        
    report('Extraction complete')