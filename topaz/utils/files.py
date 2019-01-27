from __future__ import print_function,division

import json
import pandas as pd
import numpy as np
import os
import sys

import topaz.utils.star as star
from topaz.utils.conversions import boxes_to_coordinates, coordinates_to_boxes, coordinates_to_eman2_json

particle_format_map = {
    '.star': 'star',
    '.box': 'box',
    '.json': 'json',
    '.csv': 'csv',
    '.txt': 'coord',
    '.tab': 'coord',
}

class UnknownFormatError(Exception):
    def __init__(self, ext):
        self.ext = ext

def detect_format(path):
    _,ext = os.path.splitext(path)
    if ext not in particle_format_map:
        raise UnknownFormatError(ext)
    return particle_format_map[ext]

def strip_ext(name):
    clean_name,ext = os.path.splitext(name)
    return clean_name

def read_coordinates(path, format='auto'):
    if format == 'auto':
        format = detect_format(path)

    if format == 'star':
        with open(path, 'r') as f:
            table = star.parse(f)

        # standardize the image name, x, y, and score column names
        d = {star.SCORE_COLUMN_NAME: 'score',
             'MicrographName': 'image_name',
             star.X_COLUMN_NAME: 'x_coord',
             star.Y_COLUMN_NAME: 'y_coord',
             }

        for k,v in d.items():
            if k in table.columns:
                table[v] = table[k]
                table = table.drop(k, axis=1)
        # strip off image extension, but save this for later
        table['image_name'] = table['image_name'].apply(strip_ext) 
        particles = table

    elif format == 'box':
        box = pd.read_csv(path, sep='\t', header=None).values
        image_name = os.path.basename(os.path.splitext(path)[0])
        particles = boxes_to_coordinates(box, image_name=image_name)
    elif format == 'csv':
        particles = pd.read_csv(path)
    else: # default to coordiantes table format
        particles = pd.read_csv(path, sep='\t')

    return particles

def write_coordinates(path, table, format='auto', boxsize=0, image_ext='.mrc'):
    if format == 'box' or format == 'json':
        # writing separate file per image
        for image_name,group in table.groupby('image_name'):
            if format == 'box':
                this_path = path + '/' + image_name + '.box'
                xy = group[['x_coord', 'y_coord']].values.astype(np.int32)
                boxes = coordinates_to_boxes(xy, boxsize, boxsize)
                boxes = pd.DataFrame(boxes)
                boxes.to_csv(this_path, sep='\t', header=False, index=False)
            else: # json format
                this_path = path + '/' + image_name + '_info.json'
                xy = group[['x_coord','y_coord']].values.astype(int)
                boxes = coordinates_to_eman2_json(xy)
                with open(this_path, 'w') as f:
                    json.dump({'boxes': boxes}, f, indent=0)

    elif format == 'star':
        # fix column names to be star format
        d = {'score': star.SCORE_COLUMN_NAME,
             'image_name': 'MicrographName',
             'x_coord': star.X_COLUMN_NAME,
             'y_coord': star.Y_COLUMN_NAME,
             }
        table = table.copy()
        for k,v in d.items():
            if k in table.columns:
                table[v] = table[k]
                table = table.drop(k, axis=1)
        # append image extension
        table['MicrographName'] = table['MicrographName'].apply(lambda x: x + image_ext)
        
        star.write(table, path)

    elif format == 'csv':
        # filter columns to only include image name, x, y, score (if score is present)
        columns = ['image_name', 'x_coord', 'y_coord']
        if 'score' in table.columns:
            columns.append('score')
        table = table[columns]
        table.to_csv(path, index=False)

    else: # write default coordinates format
        # filter columns to only include image name, x, y, score (if score is present)
        columns = ['image_name', 'x_coord', 'y_coord']
        if 'score' in table.columns:
            columns.append('score')
        table = table[columns]
        table.to_csv(path, sep='\t', index=False)
    





