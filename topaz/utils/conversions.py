from __future__ import division, print_function

import glob
import json
import os
import sys
from locale import strcoll
from typing import List

import numpy as np
import pandas as pd
import topaz.utils.star as star
from topaz.utils.data.loader import load_image


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


def file_boxes_to_coordinates(input_paths:List[str], image_dir:str, image_ext:str, invert_y:bool, output_path:str=None):
    tables = []

    for path in input_paths:
        if os.path.getsize(path) == 0:
            continue

        shape = None
        image_name = os.path.splitext(os.path.basename(path))[0]
        if invert_y:
            impath = os.path.join(image_dir, image_name) + '.' + image_ext
            # use glob incase image_ext is '*'
            impath = glob.glob(impath)[0]
            im = load_image(impath)
            shape = (im.height,im.width)

        box = pd.read_csv(path, sep='\t', header=None).values

        coords = boxes_to_coordinates(box, shape=shape, invert_y=invert_y, image_name=image_name)

        tables.append(coords)

    table = pd.concat(tables, axis=0)

    output = sys.stdout if output_path is None else output_path
    table.to_csv(output, sep='\t', index=False)


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


def file_coordinates_to_boxes(input_paths:List[str], destdir:str, boxsize:int, invert_y:bool, image_dir:str, image_ext:str):
    dfs = []
    for path in input_paths:
        coords = pd.read_csv(path, sep='\t')
        dfs.append(coords)
    coords = pd.concat(dfs, axis=0)

    coords = coords.drop_duplicates()

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    for image_name,group in coords.groupby('image_name'):
        path = destdir + '/' + image_name + '.box'

        shape = None
        if invert_y:
            impath = os.path.join(image_dir, image_name) + '.' + image_ext
            # use glob incase image_ext is '*'
            impath = glob.glob(impath)[0]
            im = load_image(impath)
            shape = (im.height,im.width)
        
        xy = group[['x_coord', 'y_coord']].values.astype(np.int32)

        boxes = coordinates_to_boxes(xy, boxsize, boxsize, shape=shape, invert_y=invert_y)
        boxes = pd.DataFrame(boxes)

        boxes.to_csv(path, sep='\t', header=False, index=False)


def coordinates_to_eman2_json(coords, shape=None, invert_y=False, tag='manual'):
    entries = []
    x_coords = coords[:,0]
    y_coords = coords[:,1]
    if invert_y:
        y_coords = shape[0]-1-coords[:,1]
    for i in range(len(x_coords)):
        entries.append([int(x_coords[i]), int(y_coords[i]), tag])
    return entries


def file_coordinates_to_eman2_json(input_paths:List[str], destdir:str, invert_y:bool, image_dir:str, image_ext:str):
    dfs = []
    for path in input_paths:
        coords = pd.read_csv(path, sep='\t')
        dfs.append(coords)
    coords = pd.concat(dfs, axis=0)

    coords = coords.drop_duplicates()
    print(len(coords))

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    for image_name,group in coords.groupby('image_name'):
        path = destdir + '/' + image_name + '_info.json'

        shape = None
        if invert_y:
            impath = os.path.join(image_dir, image_name) + '.' + image_ext
            # use glob incase image_ext is '*'
            impath = glob.glob(impath)[0]
            im = load_image(impath)
            shape = (im.height,im.width)
        
        xy = group[['x_coord','y_coord']].values.astype(int)
        boxes = coordinates_to_eman2_json(xy, shape=shape, invert_y=invert_y)

        with open(path, 'w') as f:
            json.dump({'boxes': boxes}, f, indent=0)


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


def star_to_coordinates(input_file, output_file=None):
    def strip_ext(name):
        clean_name,ext = os.path.splitext(name)
        return clean_name

    with open(input_file, 'r') as f:
        table = star.parse(f)

    if 'ParticleScore' in table.columns:
        ## columns of interest are 'MicrographName', 'CoordinateX', 'CoordinateY', and 'ParticleScore'
        table = table[['MicrographName', 'CoordinateX', 'CoordinateY', 'ParticleScore']]
        table.columns = ['image_name', 'x_coord', 'y_coord', 'score']
    else:
        ## columns of interest are 'MicrographName', 'CoordinateX', and 'CoordinateY'
        table = table[['MicrographName', 'CoordinateX', 'CoordinateY']]
        table.columns = ['image_name', 'x_coord', 'y_coord']
    ## convert the coordinates to integers
    table['x_coord'] = table['x_coord'].astype(float).astype(int)
    table['y_coord'] = table['y_coord'].astype(float).astype(int)
    ## strip file extensions off the image names if present
    table['image_name'] = table['image_name'].apply(strip_ext) 

    out = output_file if output_file is not None else sys.stdout
    table.to_csv(out, sep='\t', index=False)
