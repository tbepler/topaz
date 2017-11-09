from __future__ import print_function,division

import numpy as np
import pandas as pd


def boxes_to_coordinates(boxes, shape, image_name=None):
    ## first 2 columns are x and y coordinates of lower left box corners
    ## next 2 columns are width and height
    ## requires knowing image size to invert y-axis (shape parameter)
    ## to conform with origin in upper-left rather than lower-left
    x_lo = boxex[0]
    y_lo = boxes[1]
    width = boxes[2]
    height = boxes[3]
    x_coord = x_lo + width//2
    y_coord = (shape[0]-1-y_lo) - height//2

    coords = np.stack([x_coord, y_coord], axis=1)
    if image_name is not None: # in this case, return as table with image_name column
        coords = pd.DataFrame(coords, columns=['x_coord', 'y_coord'])
        coords.insert('image_name', 0, [image_name]*len(coords))

    return coords




