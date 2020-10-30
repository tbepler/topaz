from __future__ import print_function,division

import json
import pandas as pd
import numpy as np
import csv
import os
import sys

import topaz.utils.star as star
from topaz.utils.conversions import boxes_to_coordinates, coordinates_to_boxes, coordinates_to_eman2_json, coordinates_to_star

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

def read_via_csv(path):
    # this is the VIA format CSV
    table = pd.read_csv(path)
    # need to:
    # 1. remove image ext
    # 2. parse region shape attributes dictionary into particle coordinates
    table['image_name'] = table['filename'].apply(strip_ext) 
    table = table.drop('filename', axis=1)

    # don't include images with region_count==0
    table = table.loc[table['region_count'] > 0]

    regions = table['region_shape_attributes']
    x_coord = np.zeros(len(table), dtype=int)
    y_coord = np.zeros(len(table), dtype=int)
    for i in range(len(regions)):
        region = json.loads(regions.iloc[i])
        x_coord[i] = region['cx']
        y_coord[i] = region['cy']

    # parse region_attributes for scores
    scores = None
    attributes = table['region_attributes']
    if len(table) > 0:
        # check that we have score attributes
        att = json.loads(attributes.iloc[0])
        if 'score' in att:
            scores = np.zeros(len(table), dtype=np.float32) - np.inf
            for i in range(len(attributes)):
                att = json.loads(attributes.iloc[i])
                if 'score' in att:
                    scores[i] = float(att['score'])


    table = table.drop(['file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes'], 1)
    table['x_coord'] = x_coord
    table['y_coord'] = y_coord

    if scores is not None:
        table['score'] = scores

    return table

def write_via_csv(path, table):
    # write the particles as VIA format CSV
    filename = table['image_name'].apply(lambda x: x + '.png') # need to add .png extension
    via_table = pd.DataFrame({'filename': filename})

    via_table['file_size'] = -1
    via_table['file_attributes'] = '{}'

    via_table['region_count'] = 0
    via_table['region_id'] = 0
    for im,group in table.groupby('image_name'):
        count = len(group)
        ident = np.arange(count)
        where = via_table['filename'] == im + '.png'
        via_table.loc[where,'region_count'] = count
        via_table.loc[where,'region_id'] = ident 

    regions = []
    template = '{{"name":"point","cx":{},"cy":{}}}'
    for i in range(len(table)):
        region = template.format(table['x_coord'].iloc[i], table['y_coord'].iloc[i])
        regions.append(region)

    via_table['region_shape_attributes'] = regions

    if 'score' in table.columns:
        scores = []
        template = '{{"score":"{}"}}'
        for i in range(len(table)):
            score = template.format(table['score'].iloc[i])
            scores.append(score)
        via_table['region_attributes'] = scores
    else:
        via_table['region_attributes'] = '{}'

    via_table.to_csv(path, index=False)


def read_box(path):
    # columns are separated by some number of spaces
    # and are ordered as x,y,width,height
    table = []
    with open(path, 'r') as f:
        for line in f:
            if line != '':
                tokens = line.split()
                x = int(tokens[0])
                y = int(tokens[1])
                w = int(tokens[2])
                h = int(tokens[3])
                table.append([x,y,w,h])
    table = np.array(table, dtype=int)
    return table


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
             star.VOLTAGE: 'voltage',
             star.DETECTOR_PIXEL_SIZE: 'detector_pixel_size',
             star.MAGNIFICATION: 'magnification',
             star.AMPLITUDE_CONTRAST: 'amplitude_contrast',
             }

        for k,v in d.items():
            if k in table.columns:
                table[v] = table[k]
                table = table.drop(k, axis=1)
        # strip off image extension, but save this for later
        table['image_name'] = table['image_name'].apply(strip_ext) 
        particles = table

    elif format == 'box':
        box = read_box(path)
        image_name = os.path.basename(os.path.splitext(path)[0])
        particles = boxes_to_coordinates(box, image_name=image_name)
    elif format == 'csv':
        # this is VIA CSV format
        particles = read_via_csv(path)
    else: # default to coordiantes table format
        particles = pd.read_csv(path, sep='\t')

    return particles


def write_coordinates(path, table, format='auto', boxsize=0, image_ext='.mrc', suffix=''):
    if format == 'box' or format == 'json':
        # writing separate file per image
        for image_name,group in table.groupby('image_name'):
            if format == 'box':
                this_path = path + '/' + image_name + suffix + '.box'
                xy = group[['x_coord', 'y_coord']].values.astype(np.int32)
                boxes = coordinates_to_boxes(xy, boxsize, boxsize)
                boxes = pd.DataFrame(boxes)
                boxes.to_csv(this_path, sep='\t', header=False, index=False)
            else: # json format
                this_path = path + '/' + image_name + suffix + '_info.json'
                xy = group[['x_coord','y_coord']].values.astype(int)
                boxes = coordinates_to_eman2_json(xy)
                with open(this_path, 'w') as f:
                    json.dump({'boxes': boxes}, f, indent=0)

    elif format == 'star':
        table = coordinates_to_star(table, image_ext=image_ext)
        star.write(table, path)

    elif format == 'csv':
        # write as VIA CSV
        write_via_csv(path, table)

    else: # write default coordinates format
        # filter columns to only include image name, x, y, score (if score is present)
        columns = ['image_name', 'x_coord', 'y_coord']
        if 'score' in table.columns:
            columns.append('score')
        table = table[columns]
        table.to_csv(path, sep='\t', index=False)
    

def write_table(f, table, format='auto', boxsize=0, image_ext=''):
    if format == 'box' or format == 'json':
        if format == 'box':
            xy = table[['x_coord', 'y_coord']].values.astype(np.int32)
            boxes = coordinates_to_boxes(xy, boxsize, boxsize)
            boxes = pd.DataFrame(boxes)
            boxes.to_csv(f, sep='\t', header=False, index=False)
        else: # json format
            xy = table[['x_coord','y_coord']].values.astype(int)
            boxes = coordinates_to_eman2_json(xy)
            json.dump({'boxes': boxes}, f, indent=0)

    elif format == 'star':
        table = coordinates_to_star(table, image_ext=image_ext)
        star.write(table, f)

    elif format == 'csv':
        # write as VIA CSV
        write_via_csv(f, table)

    else: # write default coordinates format
        # filter columns to only include image name, x, y, score (if score is present)
        columns = ['image_name', 'x_coord', 'y_coord']
        if 'score' in table.columns:
            columns.append('score')
        table = table[columns]
        table.to_csv(f, sep='\t', index=False)
    



