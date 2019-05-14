from __future__ import print_function

import numpy as np
import struct
from collections import namedtuple

#int nx
#int ny
#int nz
fstr = '3i'
names = 'nx ny nz'

#int mode
fstr += 'i'
names += ' mode'

#int nxstart
#int nystart
#int nzstart
fstr += '3i'
names += ' nxstart nystart nzstart'

#int mx
#int my
#int mz
fstr += '3i'
names += ' mx my mz'

#float xlen
#float ylen
#float zlen
fstr += '3f'
names += ' xlen ylen zlen'

#float alpha
#float beta
#float gamma
fstr += '3f'
names += ' alpha beta gamma'

#int mapc
#int mapr
#int maps
fstr += '3i'
names += ' mapc mapr maps'

#float amin
#float amax
#float amean
fstr += '3f'
names += ' amin amax amean'

#int ispg
#int next
#short creatid
fstr += '2ih'
names += ' ispg next creatid'

#pad 30 (extra data)
# [98:128]
fstr += '30x'

#short nint
#short nreal
fstr += '2h'
names += ' nint nreal'

#pad 20 (extra data)
# [132:152]
fstr += '20x'

#int imodStamp
#int imodFlags
fstr += '2i'
names += ' imodStamp imodFlags'

#short idtype
#short lens
#short nd1
#short nd2
#short vd1
#short vd2
fstr += '6h'
names += ' idtype lens nd1 nd2 vd1 vd2'

#float[6] tiltangles
fstr += '6f'
names += ' tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz'

## NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
#float xorg
#float yorg
#float zorg
#char[4] cmap
#char[4] stamp
#float rms
fstr += '3f4s4sf'
names += ' xorg yorg zorg cmap stamp rms'

#int nlabl
#char[10][80] labels
fstr += 'i800s'
names += ' nlabl labels'

header_struct = struct.Struct(fstr)
MRCHeader = namedtuple('MRCHeader', names)

def parse(content):
    ## parse the header
    header = content[0:1024]
    header = MRCHeader._make(header_struct.unpack(content[:1024]))

    ## get the number of bytes in extended header
    extbytes = header.next
    start = 1024+extbytes # start of image data
    extended_header = content[1024:start]

    content = content[start:]
    if header.mode == 0:
        dtype = np.int8
    elif header.mode == 1:
        dtype = np.int16
    elif header.mode == 2:
        dtype = np.float32
    elif header.mode == 3:
        dtype = '2h' # complex number from 2 shorts
    elif header.mode == 4:
        dtype = np.complex64
    elif header.mode == 6:
        dtype = np.uint16
    elif header.mode == 16:
        dtype = '3B' # RGB values

    array = np.frombuffer(content, dtype=dtype) 
    # clip array to first nz*ny*nx elements
    array = array[:header.nz*header.ny*header.nx]
    ## reshape the array
    array = np.reshape(array, (header.nz, header.ny, header.nx)) # , order='F')
    if header.nz == 1:
        array = array[0]

    return array, header, extended_header

def get_mode(dtype):
    if dtype == np.int8:
        return 0
    elif dtype == np.int16:
        return 1
    elif dtype == np.float32:
        return 2
    elif dtype == np.dtype('2h'):
        return 3
    elif dtype == np.complex64:
        return 4
    elif dtype == np.uint16:
        return 6
    elif dtype == np.dtype('3B'):
        return 16
    
    raise "MRC incompatible dtype: " + str(dtype)
    

def make_header(shape, cella, cellb, mz=1, dtype=np.float32, order=(1,2,3), dmin=0, dmax=-1, dmean=-2, rms=-1
               , exthd_size=0, ispg=0):
    mode = get_mode(dtype)
    header = MRCHeader( shape[2], shape[1], shape[0], # nx, ny, nz
                        mode, # mode = 32-bit signed real
                        0, 0, 0, # nxstart, nystart, nzstart
                        1, 1, mz, # mx, my, mz
                        cella[0], cella[1], cella[2], # cella
                        cellb[0], cellb[1], cellb[2], # cellb
                        1, 2, 3, # mapc, mapr, maps
                        dmin, dmax, dmean, # dmin, dmax, dmean
                        ispg, # ispg, space group 0 means images or stack of images
                        exthd_size,
                        0, # creatid
                        0, 0, # nint, nreal
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, # xorg, yorg, zorg
                        b'\x00'*4, b'\x00'*4, #cmap, stamp
                        rms, # rms
                        0, # nlabl
                        b'\x00'*800, # labels
                      )
    return header



def write(f, array, header=None, extended_header=b'', ax=1, ay=1, az=1, alpha=0, beta=0, gamma=0):
    # make sure the array contains float32
    array = array.astype(np.float32)

    exthd_size = len(extended_header)
    if header is None:
        header = MRCHeader( array.shape[2], array.shape[1], array.shape[0], # nx, ny, nz
                            2, # mode = 32-bit signed real
                            0, 0, 0, # nxstart, nystart, nzstart
                            1, 1, 1, # mx, my, mz
                            ax, ay, az, # cella
                            alpha, beta, gamma, # cellb
                            1, 2, 3, # mapc, mapr, maps
                            array.min(), array.max(), array.mean(), # dmin, dmax, dmean
                            0, # ispg, space group 0 means images or stack of images
                            exthd_size,
                            0, # creatid
                            0, 0, # nint, nreal
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, # xorg, yorg, zorg
                            b'\x00'*4, b'\x00'*4, #cmap, stamp
                            array.std(), # rms
                            0, # nlabl
                            b'\x00'*800, # labels
                          )
    ## write the header
    buf = header_struct.pack(*list(header))
    f.write(buf)

    f.write(extended_header)

    f.write(array.tobytes())



