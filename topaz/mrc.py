from __future__ import print_function

import numpy as np
import struct
from collections import namedtuple

def parse(content):
    ## parse the header
    header = content[0:1024]

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

    MRCHeader = namedtuple('MRCHeader', names)
    header = MRCHeader._make(struct.unpack(fstr, content[:1024]))

    ## get the number of bytes in extended header
    extbytes = header.next
    start = 1024+extbytes # start of image data
    extended_header = content[1024:start]

    content = content[start:]
    if header.mode == 0:
        dtype = np.uint32
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
    ## reshape the array
    array = np.reshape(array, (header.nz, header.ny, header.nx)) # , order='F')
    if header.nz == 1:
        array = array[0]

    return array, header, extended_header


