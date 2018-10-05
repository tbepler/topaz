from __future__ import print_function,division

import pandas as pd

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

    return pd.DataFrame(content, columns=header)


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
