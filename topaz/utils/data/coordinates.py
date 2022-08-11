from __future__ import print_function,division
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

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

def match_coordinates_to_images(coords, images, radius=-1, dims=2):
    """If radius >= 0, convert the coordinates to an image mask"""
    nested = 'source' in coords
    coords = coordinates_table_to_dict(coords)
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
                if radius >= 0:
                    radii = np.array([radius]*len(xy), dtype=np.int32)
                    shape = (im.height, im.width)
                    xy = as_mask(shape, xy[:,0], xy[:,1], radii)
                this_matched[name] = (im,xy)
    else:
        for name in images.keys():
            im = images[name]
            xy = coords.get(name, null_coords)
            if radius >= 0:
                radii = np.array([radius]*len(xy), dtype=np.int32)
                shape = (im.height, im.width)
                xy = as_mask(shape, xy[:,0], xy[:,1], radii)
            matched[name] = (im,xy)

    return matched 
    





