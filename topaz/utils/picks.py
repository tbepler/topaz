from __future__ import division, print_function

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import topaz.mrc as mrc
import topaz.utils.star as star
import torch
from torch.nn.functional import conv2d, conv3d
from topaz.utils.image import downsample


def as_mask(shape:Tuple[int], radius:float, x_coord:List[float], y_coord:List[float], z_coord:List[float]=None, 
            use_cuda:bool=False) -> torch.Tensor:
    '''Given coordinates and bounding circle/sphere radii, return a binary mask about those points.'''
    mask = torch.zeros(shape)
    dims = 3 if z_coord is not None else 2
    filter_width = int(np.floor(radius)) * 2 + 1
    center = filter_width // 2
    x_coord, y_coord = torch.Tensor(x_coord).long(), torch.Tensor(y_coord).long()
    z_coord = torch.Tensor(z_coord).long() if dims == 3 else None
    
    # places ones at coordinate centers
    coords = (z_coord, y_coord, x_coord) if dims == 3 else (y_coord, x_coord)
    mask[coords] += 1
    mask = mask.unsqueeze(0).unsqueeze(0) # add batch and channel dims
    
    # create convolutional mask
    filter_range = torch.arange(filter_width)
    grid = torch.meshgrid([filter_range]*dims, indexing='xy')
    xgrid, ygrid = grid[0], grid[1]
    zgrid = grid[2] if dims == 3 else None
    filter = (xgrid-center)**2 + (ygrid-center)**2
    filter += (zgrid-center)**2 if dims == 3 else 0
    filter = (filter <= radius**2).float()
    filter = filter.unsqueeze(0).unsqueeze(0) # add batch and channel dims
    
    # if GPU available convolve there
    if use_cuda:
        mask = mask.cuda() 
        filter = filter.cuda()
    
    # convolve filter with input
    conv = conv3d if dims == 3 else conv2d
    mask = conv(mask, filter, padding='same').squeeze()
    mask = (mask > 0).float() # binarize
    return mask


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
