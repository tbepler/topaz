from __future__ import print_function,division

import numpy as np
import pandas as pd


def boxes_to_coordinates(boxes, shape=None, invert_y=False, image_name=None):
    ## first 2 columns are x and y coordinates of lower left box corners
    ## next 2 columns are width and height

    ## requires knowing image size to invert y-axis (shape parameter)
    ## to conform with origin in upper-left rather than lower-left
    ## apparently, EMAN2 only inverts the y-axis for .tiff images
    ## so box files only need to be inverted when working with .tiff
    x_lo = boxes[:,0]
    y_lo = boxes[:,1]
    width = boxes[:,2]
    height = boxes[:,3]
    x_coord = x_lo + width//2
    y_coord = y_lo + height//2

    if invert_y:
        y_coord = (shape[0]-1-y_lo) - height//2

    coords = np.stack([x_coord, y_coord], axis=1)
    if image_name is not None: # in this case, return as table with image_name column
        coords = pd.DataFrame(coords, columns=['x_coord', 'y_coord'])
        coords.insert(0, 'image_name', [image_name]*len(coords))

    return coords


def coordinates_to_boxes(coords, shape, box_width, box_height):
    x_coord = coords[:,0]
    y_coord = shape[0]-1-coords[:,1]
    box_width = np.array([box_width]*len(x_coord), dtype=np.int32)
    box_height = np.array([box_height]*len(x_coord), dtype=np.int32)

    boxes = np.stack([x_coord, y_coord, box_width, box_height], 1)

    return boxes


def coordinates_to_eman2_json(coords, shape=None, invert_y=False, tag='manual'):
    entries = []
    x_coords = coords[:,0]
    y_coords = coords[:,1]
    if invert_y:
        y_coords = shape[0]-1-coords[:,1]
    for i in range(len(x_coords)):
        entries.append([int(x_coords[i]), int(y_coords[i]), tag])
    return entries


