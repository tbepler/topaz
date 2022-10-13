from __future__ import division, print_function

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import topaz.mrc as mrc
import topaz.utils.star as star
from topaz.utils.image import downsample


def as_mask(shape:Tuple[int], radii:List[float], x_coord:List[float], y_coord:List[float], z_coord:List[float]=None) -> np.ndarray:
    '''Given coordinates and bounding circle/sphere radii, return a binary mask about those points.'''
    dims = 3 if z_coord is not None else 2
    N = len(x_coord) #number of target coordinates

    #expand dims for vectorization
    x_coord = np.array(x_coord).reshape([1]*dims + [N]) 
    y_coord = np.array(y_coord).reshape([1]*dims + [N])  

    yrange = np.arange(shape[0])
    xrange = np.arange(shape[1])
    #create 2D or 3D meshgrids of all coordinates
    if dims == 3:
        z_coord = np.array(z_coord).reshape([1]*dims + [N])
        zrange = np.arange(shape[2])
        xgrid,ygrid,zgrid = np.meshgrid(xrange, yrange, zrange, indexing='xy')
        zgrid = np.expand_dims(zgrid, axis=-1)
    else:
        xgrid,ygrid = np.meshgrid(xrange, yrange, indexing='xy')
    xgrid = np.expand_dims(xgrid, axis=-1)
    ygrid = np.expand_dims(ygrid, axis=-1)

    #calculate distance tensor from each voxel to each target coordinate; X x Y x Z x N
    d2 = (xgrid - x_coord)**2 + (ygrid - y_coord)**2
    d2 += (zgrid - z_coord)**2 if dims == 3 else 0
    mask = (d2 <= np.array(radii)**2).sum(axis=-1) #sum over particles w/in threshold radius, binarize
    return np.clip(mask, 0, 1)


def scale_coordinates(input_file:str, scale:float, output_file:str=None):
    '''Scale pick coordinates for resized images
    '''
    ## load picks
    df = pd.read_csv(input_file, sep='\t')

    if 'diameter' in df:
        df['diameter'] = np.ceil(df.diameter*scale).astype(np.int32)
    df['x_coord'] = np.round(df.x_coord*scale).astype(np.int32)
    df['y_coord'] = np.round(df.y_coord*scale).astype(np.int32)
    
    ## write the scaled df
    out = sys.stdout if output_file is None else open(output_file, 'w')
    df.to_csv(out, sep='\t', header=True, index=False)
    if output_file is not None:
        out.close()
        
        
def create_particle_stack(input_file:str, output_file:str, threshold:float, size:int, 
                          resize:int, image_root:str, image_ext:str, metadata:str):
    particles = pd.read_csv(input_file, sep='\t')

    print('#', 'Loaded', len(particles), 'particles', file=sys.stderr)

    # threshold the particles
    if 'score' in particles:
        particles = particles.loc[particles['score'] >= threshold]
        print('#', 'Thresholding at', threshold, file=sys.stderr)

    print('#', 'Extracting', len(particles), 'particles', file=sys.stderr)

    N = len(particles)
    if resize < 0:
        resize = size

    wrote_header = False
    read_metadata = False
    metadata = []

    # write the particles iteratively
    i = 0
    with open(output_file, 'wb') as f:
        for image_name,coords in particles.groupby('image_name'):

            print('#', image_name, len(coords), 'particles', file=sys.stderr)

            # load the micrograph
            image_name = image_name + image_ext
            path = os.path.join(image_root, image_name) 
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

    image_name = os.path.basename(output_file)
    star_path = os.path.splitext(output_file)[0] + '.star'

    ## create the star file
    columns = ['MicrographName', star.X_COLUMN_NAME, star.Y_COLUMN_NAME]
    if 'score' in particles:
        columns.append(star.SCORE_COLUMN_NAME)
    metadata = pd.DataFrame(metadata, columns=columns)
    metadata['ImageName'] = [str(i+1) + '@' + image_name for i in range(len(metadata))]
    if mz > 1:
        metadata['NrOfFrames'] = mz

    micrograph_metadata = None
    if metadata is not None:
        with open(metadata, 'r') as f:
            micrograph_metadata = star.parse_star(f)
        metadata = pd.merge(metadata, micrograph_metadata, on='MicrographName', how='left')

    if resize != size and 'DetectorPixelSize' in metadata:
        # rescale the detector pixel size
        pix = metadata['DetectorPixelSize'].values.astype(float)
        metadata['DetectorPixelSize'] = pix*(size/resize)


    ## write the star file
    with open(star_path, 'w') as f:
        star.write(metadata, f)
