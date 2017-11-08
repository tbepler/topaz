from __future__ import print_function,division

import os
import sys

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Script for generating an image list file from a list of files passed on the command line')
    parser.add_argument('paths', nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    paths = args.paths
    
    print('image_name\tpath')
    for path in paths:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        #source = name.split('_')[0]
        #print(source + '\t' + name + '\t' + path)
        print(name + '\t' + path)



