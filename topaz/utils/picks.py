from __future__ import print_function, division

import numpy as np

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







