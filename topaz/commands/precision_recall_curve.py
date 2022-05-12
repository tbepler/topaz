#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys

from topaz.metrics import particle_prc
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd
import argparse


name = 'precision_recall_curve'
help = 'calculate the precision-recall curve for a set of predicted particle coordinates with scores and a set of target coordinates'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Script for calculating the precision-recall curve for a set of predicted particle coordinates and a set of target coordinates.')

    parser.add_argument('--predicted', help='path to file containing predicted particle coordinates with scores')
    parser.add_argument('--targets', help='path to file specifying target particle coordinates') 

    parser.add_argument('-r', '--assignment-radius', required=True, type=int, help='maximum distance between prediction and labeled target allowed for considering them a match')
    parser.add_argument('--images', choices=['target', 'predicted', 'union'], default='target', help='only count particles on micrographs with coordinates labeled in the targets file, the predicted file, or the union of those (default: target)')

    return parser


def main(args):
    particle_prc(args.targets, args.predicted, args.assignment_radius, args.images)
    

if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)