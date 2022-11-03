from __future__ import print_function,division
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from topaz.utils.picks import as_mask

def coordinates_table_to_dict(coords:pd.DataFrame, dims:int=2) -> Union[Dict[str,np.ndarray], Dict[Any,Dict[str,np.ndarray]]]:
    '''Converts a pandas DataFrame to a dictionary mapping image names to their contained particle coordinates.
    If source columns are included, sources are first mapped to image names.'''
    root = {}
    columns = ['x_coord','y_coord', 'z_coord'] if dims == 3 else ['x_coord','y_coord']
    if 'source' in coords:
        for (source,name),df in coords.groupby(['source', 'image_name']):
            xy_z = df[columns].values.astype(np.int32)
            root.setdefault(source,{})[name] = xy_z
    else:
        for name,df in coords.groupby('image_name'):
            xy_z = df[columns].values.astype(np.int32)
            root[name] = xy_z
    return root


def make_coordinate_mask(image:Union[Image.Image, np.ndarray], coords:np.ndarray, radius:float, use_cuda:bool=False):
    if radius < 0:
        return coords
    # radii = np.full(len(coords), radius).astype(np.int32)
    shape = (image.height, image.width) if type(image) == Image.Image else image.shape
    if len(shape) == 2:
        coords = as_mask(shape, radius, coords[:,0], coords[:,1], z_coord=None, use_cuda=use_cuda)
    elif len(shape) == 3:
        coords = as_mask(shape, radius, coords[:,0], coords[:,1], z_coord=coords[:,2], use_cuda=use_cuda)
    return coords


def match_coordinates_to_images(coords:pd.DataFrame, images:dict, radius:float=-1, dims:int=2, use_cuda:bool=False) -> \
    Union[Dict[str,Tuple[Union[Image.Image, np.ndarray],np.ndarray]], \
        Dict[Any,Dict[str,Tuple[Union[Image.Image, np.ndarray],np.ndarray]]]]:
    """If radius >= 0, convert point coordinates to mask of circles/spheres."""
    nested = ('source' in coords)
    coords = coordinates_table_to_dict(coords, dims=dims)
    null_coords = np.zeros((0,dims), dtype=np.int32)

    matched = {}
    if nested:
        for source in images.keys():
            this_matched = matched.setdefault(source,{})
            this_images = images[source]
            this_coords = coords.get(source, {})
            for name in this_images.keys():
                im = this_images[name]
                xy = this_coords.get(name, null_coords)
                xy = make_coordinate_mask(im, xy, radius, use_cuda=use_cuda) # make coord points into mask
                this_matched[name] = (im,xy)
    else:
        for name in images.keys():
            im = images[name]
            xy = coords.get(name, null_coords)
            xy = make_coordinate_mask(im, xy, radius, use_cuda=use_cuda)
            matched[name] = (im,xy)

    return matched 
    