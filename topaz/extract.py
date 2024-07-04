#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import multiprocessing
import os
import sys
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


def nms_iterator(scores:Iterable[np.ndarray], radius:int, threshold:float, pool:multiprocessing.Pool=None, dims:int=2) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    process = NonMaximumSuppression(radius, threshold, dims=dims)
    if pool is not None:
        for name,score,coords in pool.imap_unordered(process, scores):
            yield name,score,coords
    else:
        for name,score in scores:
            if dims == 2:
                score,coords = non_maximum_suppression(score, radius, threshold=threshold)
            elif dims == 3:
                score,coords = non_maximum_suppression_3d(score, radius*2, threshold=threshold)
            yield name,score,coords


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


def score_images(model:str, paths:List[str], device:int=-1, batch_size:int=1) -> Iterator[np.ndarray]:
    if model is not None and model != 'none': # score each image with the model
        ## set the device
        use_cuda = topaz.cuda.set_device(device)
        ## load the model
        from topaz.model.factory import load_model
        model = load_model(model)
        model.eval()
        model.fill()
        if use_cuda:
            model.cuda()
        scores = topaz.predict.score_stream(model, stream_images(paths), use_cuda=use_cuda, batch_size=batch_size)
    else: # load scores directly
        scores = stream_images(paths)
    for path,score in zip(paths, scores):
        yield path, score


def stream_inputs(f:TextIO) -> Iterator[str]:
    for line in f:
        line = line.strip()
        if len(line) > 0:
            yield line


def extract_particles(paths:List[str], model:Union[torch.nn.Module, str], device:int, batch_size:int, threshold:float, radius:int, num_workers:int, targets:str, min_radius:int, max_radius:int, step:int, match_radius:int,
                      only_validate:bool, output:str, per_micrograph:bool, suffix:str, out_format:str, up_scale:float, down_scale:float, dims=2):
    # score the images lazily with a generator
    paths = stream_inputs(sys.stdin) if len(paths) == 0 else paths # no paths, read from stdin

    # generator of images and their scores
    stream = score_images(model, paths, device=device, batch_size=batch_size)

    # extract coordinates from scored images
    radius = radius if radius is not None else -1

    num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers) if num_workers > 0 else None

    # if no radius is set, we choose the radius based on targets provided
    if radius < 0 and targets is not None: # set the radius to optimize AUPRC of the targets
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        ## find radius maximizing AUPRC
        radius, auprc = find_opt_radius(targets, target_scores, threshold, lo=min_radius, hi=max_radius, step=step, match_radius=match_radius, pool=pool, dims=dims)

    elif targets is not None:
        scores = {k:v for k,v in stream} # process all images for this part
        stream = scores.items()

        targets = pd.read_csv(targets, sep='\t')
        target_scores = {name: scores[name] for name in targets.image_name.unique() if name in scores}
        # calculate AUPRC for radius
        au, rmse, recall, n = extract_auprc(targets, target_scores, radius, threshold, match_radius=match_radius, pool=pool, dims=dims)
        print('# radius={}, auprc={}, rmse={}, recall={}, targets={}'.format(radius, au, rmse, recall, n))
    
    elif radius < 0:
        # must have targets if radius < 0
        raise Exception('Must specify targets for choosing the extraction radius if extraction radius is not provided')


    # now, extract all particles from scored images
    if not only_validate:
        f = sys.stdout if output is None or not per_micrograph else open(output, 'w')
        
        scale = up_scale/down_scale

        # combining all files together, print header first
        if not per_micrograph:
            print('image_name\tx_coord\ty_coord\tscore', file=f)
        
        ## extract coordinates using radius 
        for path,score,coords in nms_iterator(stream, radius, threshold, pool=pool, dims=dims):
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            ## scale the coordinates
            if scale != 1:
                coords = np.round(coords*scale).astype(int)
            if per_micrograph:
                table = pd.DataFrame({'image_name': name, 'x_coord': coords[:,0], 'y_coord': coords[:,1], 'score': score})
                out_path,ext = os.path.splitext(path)
                out_path = out_path + suffix + '.' + out_format
                with open(out_path, 'w') as f:
                    file_utils.write_table(f, table, format=out_format, image_ext=ext)
            else:
                for i in range(len(score)):
                    print(name + '\t' + str(coords[i,0]) + '\t' + str(coords[i,1]) + '\t' + str(score[i]), file=f)   
        f.close()