#!/usr/bin/env python

from __future__ import print_function, division
import os


template = """#!/usr/bin/env bash

# make the saved model directory if it doesnt exist
mkdir -p {destdir}/{name}_saved/

python scripts/train.py \
        --train-images={train_images} \
        --train-targets={train_targets} \
        --test-images={test_images} \
        --test-targets={test_targets} \
        --radius={radius} \
        --minibatch-size=256 \
        --minibatch-balance=0.0625 \
        --num-workers=8 \
        --num-epochs=10 \
        --epoch-size=5000 \
        --model={model} \
        --autoencoder={autoencoder} \
        --ngf=32 \
        --method={method} \
        --slack={slack} \
        --pi={pi} \
        --units={units} \
        --unit-scaling={unit_scaling} \
        --learning-rate={learning_rate} \
        --save-prefix={destdir}/{name}_saved/{name} \
        > {destdir}/{name}_train.txt
"""

destdir = "results/EMPIAR-10096_subsample_comparison"
train_images = "data/EMPIAR-10096/images_4a_train.txt"

datasets = ['100']
train_target_template = "data/EMPIAR-10096/picks_4a_train{}.txt"

test_images = "data/EMPIAR-10096/images_4a_test.txt"
test_targets = "data/EMPIAR-10096/picks_4a_test.txt"

reps = 10 # reps for each number of picks

radius = 4

pi = 0.02

model = 'conv31'
unit_scaling = 2
units = 32

learning_rate = 0.0002

methods = [
           ('PN', 1, 0),
           ('PU', 1, 0),
           ('GE-KL', 100, 0),
           ('GE-binomial', 1, 0),
           ('GE-binomial', 1, 0.1),
          ]

script_dir = 'experiments/EMPIAR_10096_subsample_comparison_scripts'
if not os.path.exists(script_dir):
    os.makedirs(script_dir)

name_template = 'EMPIAR_10096_{data}_{model}_{units}_autoencoder{autoencoder}_{method}_pi{pi}_rep{rep}'

for data in datasets:
    for rep in range(reps):
      
        if data == '1000':
            train_rep = train_target_template.format('_1000')
            data_name = '1000'
        else:
            suffix = '_' + data + '_rep' + str(rep)
            train_rep = train_target_template.format(suffix)
            data_name = data

        for method,slack,autoencoder in methods:

            name = name_template.format( data=data_name
                                       , model=model
                                       , units=units
                                       , autoencoder=autoencoder
                                       , method=method
                                       , pi=pi
                                       , rep=rep
                                       )

            ## generate the script code
            code = template.format(name=name, destdir=destdir,
                                   train_images=train_images,
                                   train_targets=train_rep,
                                   test_images=test_images,
                                   test_targets=test_targets,
                                   radius=radius,
                                   model=model,
                                   autoencoder=autoencoder,
                                   method=method,
                                   pi=pi,
                                   slack=slack,
                                   units=units,
                                   unit_scaling=unit_scaling,
                                   learning_rate=learning_rate,
                                   )
            ## write to script file
            with open(script_dir+'/train_'+name+'.sh', 'w') as f:
                f.write(code) 

