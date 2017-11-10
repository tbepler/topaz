from __future__ import print_function,division

import numpy as np

from topaz.utils.picks import as_mask

def coordinates_table_to_dict(coords):
    root = {}
    if 'source' in coords:
        for (source,name),df in coords.groupby(['source', 'image_name']):
            xy = df[['x_coord','y_coord']].values.astype(np.int32)
            root.setdefault(source,{})[name] = xy
    else:
        for name,df in coords.groupby('image_name'):
            xy = df[['x_coord','y_coord']].values.astype(np.int32)
            root[name] = xy
    return root

def match_coordinates_to_images(coords, images, radius=-1):
    """
    If radius >= 0, then convert the coordinates to an image mask
    """
    
    nested = 'source' in coords
    coords = coordinates_table_to_dict(coords)
    null_coords = np.zeros((0,2), dtype=np.int32)

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
    





