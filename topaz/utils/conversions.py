from __future__ import print_function,division

import topaz.utils.star as star

import numpy as np
import pandas as pd

def mirror_y_axis(coords, n):
    coords = coords.clone()
    coords['y_coord'] = n-1-coords['y_coord']
    return coords

def boxes_to_coordinates(boxes, shape=None, invert_y=False, image_name=None):
    if len(boxes) < 1: # boxes are empty, return empty coords table
        columns = ['x_coord', 'y_coord']
        if image_name is not None:
            columns.append('image_name')
        coords = pd.DataFrame(columns=columns)
        return coords

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

def coordinates_to_boxes(coords, box_width, box_height, shape=None, invert_y=False, tag='manual'):
    entries = []
    x_coords = coords[:,0]
    y_coords = coords[:,1]
    if invert_y:
        y_coords = shape[0]-1-coords[:,1]
    box_width = np.array([box_width]*len(x_coords), dtype=np.int32)
    box_height = np.array([box_height]*len(x_coords), dtype=np.int32)

    # x and y are centers, make lower left corner
    x_coords = x_coords - box_width//2
    y_coords = y_coords - box_height//2

    boxes = np.stack([x_coords, y_coords, box_width, box_height], 1)
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


def coordinates_to_star(table, image_ext=''):
    # fix column names to be star format
    d = {'score': star.SCORE_COLUMN_NAME,
            'image_name': 'MicrographName',
            'x_coord': star.X_COLUMN_NAME,
            'y_coord': star.Y_COLUMN_NAME,
            'voltage': star.VOLTAGE,
            'detector_pixel_size': star.DETECTOR_PIXEL_SIZE,
            'magnification': star.MAGNIFICATION,
            'amplitude_contrast': star.AMPLITUDE_CONTRAST,
            }
    table = table.copy()
    for k,v in d.items():
        if k in table.columns:
            table[v] = table[k]
            table = table.drop(k, axis=1)
    # append image extension
    table['MicrographName'] = table['MicrographName'].apply(lambda x: x + image_ext)

    return table

