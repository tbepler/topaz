from __future__ import division, print_function

import sys

import numpy as np
import pandas as pd


def as_mask(shape, x_coord, y_coord, radii):

    ygrid = np.arange(shape[0])
    xgrid = np.arange(shape[1])
    xgrid,ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')

    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(len(x_coord)):
        x = x_coord[i]
        y = y_coord[i]
        radius = radii[i]
        threshold = radius**2
        
        d2 = (xgrid - x)**2 + (ygrid - y)**2
        mask += (d2 <= threshold)

    mask = np.clip(mask, 0, 1)
    return mask


def scale_coordinates(input_file:str, scale:float, output_file:str=None):
    '''Scale pick coordinates for resized images
    '''
    ## load picks
    df = pd.read_csv(input_file, sep='\t')

    if 'diameter' in df:
        df['diameter'] = np.ceil(df.diameter*scale).astype(np.int32)
    df['x_coord'] = np.round(df.x_coord*scale).astype(np.int32)
    df['y_coord'] = np.round(df.y_coord*scale).astype(np.int32)
    
    ## write the scaled df
    out = sys.stdout if output_file is None else open(output_file, 'w')
    df.to_csv(out, sep='\t', header=True, index=False)
    if output_file is not None:
        out.close()