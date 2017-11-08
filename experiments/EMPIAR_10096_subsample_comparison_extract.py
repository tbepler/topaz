from __future__ import print_function,division

import os
import pandas as pd
import glob
import subprocess

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int, default=0)
    return parser.parse_args()

args = parse_args()
jobs = args.jobs

epoch = 10
radius = 10

paths = glob.glob('results/EMPIAR-10096_subsample_comparison/*_saved')

test_images = pd.read_csv('data/EMPIAR-10096/images_4a_test.txt', sep='\t')
test_images = test_images.image_name
test_images = ['data/EMPIAR-10096/images/4a/' + im + '.tiff' for im in test_images]

def output_file_exists(path):
    return os.path.exists(path) and (os.path.getsize(path) > 0)

def extract(path):
    name = os.path.basename(path)[:-6]
    model_name = name + '_epoch' + str(epoch) + '.sav'

    model_path = path + '/' + model_name
    output_path = path[:-6] + '_epoch' + str(epoch) + '_extracted_test.txt'
   
    cmd = ['python', 'scripts/extract.py', '-r'+str(radius), '-m', model_path
          , '-o', output_path]

    if os.path.exists(model_path) and not output_file_exists(output_path):
        print('#', ' '.join(cmd))
        cmd += test_images
        subprocess.call(cmd)

if jobs > 0:
    import multiprocessing
    pool = multiprocessing.Pool(jobs)
    for _ in pool.imap_unordered(extract, paths):
        pass
else:
    for path in paths:
        extract(path)
    

    




