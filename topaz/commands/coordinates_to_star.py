from __future__ import print_function,division

import sys
import os
import numpy as np
import pandas as pd

import topaz.utils.star as star

name = 'coordinates_to_star'
help = 'convert coordinates table to .star file format'


def add_arguments(parser):
    parser.add_argument('file', help='path to input coordinates file')
    parser.add_argument('--image-ext', default='.mrc', help='image file extension')
    parser.add_argument('--voltage', type=float, default=-1, help='voltage setting')
    parser.add_argument('--defocus-u', type=float, default=-1, help='defocus U setting')
    parser.add_argument('--defocus-v', type=float, default=-1, help='defocus V setting')
    parser.add_argument('--defocus-angle', type=float, default=-1, help='defocus angle setting')
    parser.add_argument('--spherical-aberation', type=float, default=-1, help='spherical aberation setting')
    parser.add_argument('--amplitude-contrast', type=float, default=-1, help='amplitude contrast setting')
    parser.add_argument('--detector-pixel-size', type=float, default=-1, help='detector pixel size setting')
    parser.add_argument('--magnification', type=float, default=-1, help='magnification setting')
    return parser


def main(args):
    particles = pd.read_csv(args.file, sep='\t')
    image_name = particles.image_name
    x_coord = particles.x_coord
    y_coord= particles.y_coord
    particle_score = None
    if 'score' in particles:
        particle_score = particles['score']


    #_,image_idx = np.unique(image_name, return_inverse=True)
    micrograph_name = [name + args.image_ext for name in image_name]
    image_name = ['1@' + name + args.image_ext for name in image_name]
    star_table = pd.DataFrame({'ImageName': image_name})
    star_table['MicrographName'] = micrograph_name
    star_table['CoordinateX'] = x_coord
    star_table['CoordinateY'] = y_coord
    
    if particle_score is not None:
        star_table['ParticleScore'] = particle_score

    if args.voltage >= 0:
        star_table['Voltage'] = args.voltage

    if args.defocus_u >= 0:
        star_table['DefocusU'] = args.defocus_u

    if args.defocus_v >= 0:
        star_table['DefocusV'] = args.defocus_v

    if args.defocus_angle >= 0:
        star_table['DefocusAngle'] = args.defocus_angle

    if args.spherical_aberation >= 0:
        star_table['SphericalAberration'] = args.spherical_aberation

    if args.amplitude_contrast >= 0:
        star_table['AmplitudeContrast'] = args.amplitude_contrast

    if args.detector_pixel_size >= 0:
        star_table['DetectorPixelSize'] = args.detector_pixel_size

    if args.magnification >= 0:
        star_table['Magnification'] = args.magnification


    ## write the star file
    star.write(star_table, sys.stdout)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert coordinates table to star file format')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


