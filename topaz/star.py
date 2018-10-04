from __future__ import print_function,division

import sys
import pandas as pd

def parse_star(f):
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
            if len(tokens) > len(header):
                print('# Warning: too many columns in line', i, 'got', len(tokens), 'expected', len(header), file=sys.stderr)
                tokens = tokens[:len(header)]
            content.append(tokens)

    return pd.DataFrame(content, columns=header)


def parse_star_loop(lines):
    columns = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line.startswith('_'):
            break
        name = line[4:] # strip _rln
        # strip trailing comments from name
        loc = name.find('#')
        if loc >= 0:
            name = name[:loc]
        name = name.strip()
        columns.append(name)
    return columns, lines[i:]

