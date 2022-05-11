from __future__ import print_function,division

import pandas as pd
import sys

X_COLUMN_NAME = 'CoordinateX'
Y_COLUMN_NAME = 'CoordinateY'
SCORE_COLUMN_NAME = 'AutopickFigureOfMerit'
OLD_SCORE_COLUMN_NAME = 'ParticleScore'

VOLTAGE = 'Voltage'
DETECTOR_PIXEL_SIZE = 'DetectorPixelSize'
MAGNIFICATION = 'Magnification'
AMPLITUDE_CONTRAST = 'AmplitudeContrast'

def parse_star(f):
    return parse(f)

def parse(f):
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('data_'): 
            return parse_star_body(lines[i+1:])


def parse_star_body(lines):
    ## data_images line has been read, next is loop
    for i in range(len(lines)):
        if lines[i].startswith('loop_'):
            lines = lines[i+1:]
            break
    header,lines = parse_star_loop(lines)
    ## parse the body
    content = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('data'): # done with image block
            break
        if line.startswith('#') or line.startswith(';'): # comment lines
            continue
        if line != '':
            tokens = line.split()
            content.append(tokens)

    table = pd.DataFrame(content, columns=header)

    # check for old naming convention, 'ParticleScore'
    if OLD_SCORE_COLUMN_NAME in table.columns and SCORE_COLUMN_NAME not in table.columns:
        table[SCORE_COLUMN_NAME] = table[OLD_SCORE_COLUMN_NAME]
        table = table.drop(OLD_SCORE_COLUMN_NAME, axis=1)

    # convert columns to correct data type
    if X_COLUMN_NAME in table:
        table[X_COLUMN_NAME] = table[X_COLUMN_NAME].astype(float).astype(int)
    if Y_COLUMN_NAME in table:
        table[Y_COLUMN_NAME] = table[Y_COLUMN_NAME].astype(float).astype(int)
    if SCORE_COLUMN_NAME in table:
        table[SCORE_COLUMN_NAME] = table[SCORE_COLUMN_NAME].astype(float)
    if VOLTAGE in table:
        table[VOLTAGE] = table[VOLTAGE].astype(float)
    if DETECTOR_PIXEL_SIZE in table:
        table[DETECTOR_PIXEL_SIZE] = table[DETECTOR_PIXEL_SIZE].astype(float)
    if MAGNIFICATION in table:
        table[MAGNIFICATION] = table[MAGNIFICATION].astype(float)
    if AMPLITUDE_CONTRAST in table:
        table[AMPLITUDE_CONTRAST] = table[AMPLITUDE_CONTRAST].astype(float)

    return table


def parse_star_loop(lines):
    columns = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line.startswith('_'):
            break
        name = line[1:]
        # strip trailing comments from name
        loc = name.find('#')
        if loc >= 0:
            name = name[:loc]
        # strip 'rln' prefix
        if name.startswith('rln'):
            name = name[3:]
        name = name.strip()
        columns.append(name)
    return columns, lines[i:]


def write(table, f):
    ## write the star file
    print('data_images', file=f)
    print('loop_', file=f)
    for i,name in enumerate(table.columns):
        print('_rln' + name + ' #' + str(i+1), file=f)

    table.to_csv(f, sep='\t', index=False, header=False)


def threshold_star_particles(input_file, threshold, output_file=None):
    with open(input_file, 'r') as f:
        particles = parse_star(f)
    n = len(particles)
    particles['ParticleScore'] = [float(s) for s in particles['ParticleScore']]
    particles = particles.loc[particles['ParticleScore'] >= threshold]
    print('# filtered', n, 'particles to', len(particles), 'with treshold of', threshold, file=sys.stderr)

    ## write the star file
    f = sys.stdout if output_file is None else open(output_file, 'w')
    write(particles, f)
    if output_file is not None: 
        f.close() 