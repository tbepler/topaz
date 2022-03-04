from __future__ import print_function,division

import sys
import os
import numpy as np
import pandas as pd

import topaz.mrc as mrc
import topaz.utils.star as star
from topaz.utils.image import downsample
from topaz.utils.data.loader import load_mrc, load_pil

name = 'particle_stack'
help = 'extract mrc particle stack given coordinates table'

def add_arguments(parser):
    parser.add_argument('file', help='path to input coordinates file')
    parser.add_argument('--image-root', help='root directory of the micrograph files')
    parser.add_argument('-o', '--output', help='path to write particle stack file')

    parser.add_argument('--size', type=int, help='size of particle stack images')
    parser.add_argument('--threshold', type=float, default=-np.inf, help='only take particles with scores >= this value (default: -inf)')
    parser.add_argument('--resize', default=-1, type=int, help='rescaled particle stack size. downsamples particle images from size to resize pixels. (default: off)')

    parser.add_argument('--image-ext', default='.mrc', help='image file extension (default=.mrc)')

    parser.add_argument('--metadata', help='path to .star file containing per-micrograph metadata, e.g. CTF parameters (optional)')

    return parser


def load_image(path):
    ext = os.path.splitext(path)[1]
    if ext == '.mrc':
        image = load_mrc(path)
    else:
        image = load_pil(path)
    return image 


def main(args):
    particles = pd.read_csv(args.file, sep='\t')

    print('#', 'Loaded', len(particles), 'particles', file=sys.stderr)

    # threshold the particles
    if 'score' in particles:
        particles = particles.loc[particles['score'] >= args.threshold]
        print('#', 'Thresholding at', args.threshold, file=sys.stderr)

    print('#', 'Extracting', len(particles), 'particles', file=sys.stderr)

    N = len(particles)
    size = args.size
    resize = args.resize
    if resize < 0:
        resize = size

    # 
    wrote_header = False
    read_metadata = False
    metadata = []

    # write the particles iteratively
    i = 0
    with open(args.output, 'wb') as f:
        for image_name,coords in particles.groupby('image_name'):

            print('#', image_name, len(coords), 'particles', file=sys.stderr)

            # load the micrograph
            image_name = image_name + args.image_ext
            path = os.path.join(args.image_root, image_name) 
            with open(path, 'rb') as fm:
                content = fm.read()
            micrograph, header, extended_header = mrc.parse(content)
            if len(micrograph.shape) < 3:
                micrograph = micrograph[np.newaxis] # add z dim if micrograph is image
        
            if not wrote_header: # load a/px and angles from micrograph header and write the stack header
                mz = micrograph.shape[0]

                dtype = micrograph.dtype

                cella = (header.xlen, header.ylen, header.zlen)
                cellb = (header.alpha, header.beta, header.gamma)
                shape = (N*mz,resize,resize)

                header = mrc.make_header(shape, cella, cellb, mz=mz, dtype=dtype)

                buf = mrc.header_struct.pack(*list(header))
                f.write(buf)
                wrote_header = True

            _,n,m = micrograph.shape

            x_coord = coords['x_coord'].values
            y_coord = coords['y_coord'].values
            scores = None
            if 'score' in coords:
                scores = coords['score'].values

            # crop out the particles
            for j in range(len(coords)):
                x = x_coord[j]
                y = y_coord[j]

                if scores is not None: 
                    metadata.append((image_name, x, y, scores[j]))
                else:
                    metadata.append((image_name, x, y)) 

                left = x - size//2
                upper = y - size//2
                right = left + size
                lower = upper + size

                c = micrograph[ : , max(0,upper):min(n,lower) , max(0,left):min(m,right) ]
                
                c = (c - c.mean())/c.std()
                stack = np.zeros((mz, size, size), dtype=dtype)

                #stack = np.zeros((mz, size, size), dtype=dtype) + c.mean().astype(dtype)
                stack[ : , max(0,-upper):min(size+n-lower,size), max(0,-left):min(size+m-right,size) ] = c

                # write particle to mrc file
                if resize != size:
                    restack = downsample(stack, 0, shape=(resize,resize))
                    #print(restack.shape, restack.mean(), restack.std())
                    restack = (restack - restack.mean())/restack.std()
                    f.write(restack.tobytes())
                else:
                    f.write(stack.tobytes())

                i += 1
                #print('# wrote', i, 'out of', N, 'particles', end='\r', flush=True)


    ## write the particle stack mrcs
    #with open(args.output, 'wb') as f:
    #    mrc.write(f, stack, ax=ax, ay=ay, az=az, alpha=alpha, beta=beta, gamma=gamma)

    image_name = os.path.basename(args.output)
    star_path = os.path.splitext(args.output)[0] + '.star'

    ## create the star file
    columns = ['MicrographName', star.X_COLUMN_NAME, star.Y_COLUMN_NAME]
    if 'score' in particles:
        columns.append(star.SCORE_COLUMN_NAME)
    metadata = pd.DataFrame(metadata, columns=columns)
    metadata['ImageName'] = [str(i+1) + '@' + image_name for i in range(len(metadata))]
    if mz > 1:
        metadata['NrOfFrames'] = mz

    micrograph_metadata = None
    if args.metadata is not None:
        with open(args.metadata, 'r') as f:
            micrograph_metadata = star.parse_star(f)
        metadata = pd.merge(metadata, micrograph_metadata, on='MicrographName', how='left')

    if resize != size and 'DetectorPixelSize' in metadata:
        # rescale the detector pixel size
        pix = metadata['DetectorPixelSize'].values.astype(float)
        metadata['DetectorPixelSize'] = pix*(size/resize)


    ## write the star file
    with open(star_path, 'w') as f:
        star.write(metadata, f)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for extracting mrc stack from particle coordinates')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


