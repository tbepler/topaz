from __future__ import print_function, division

cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport ceil, exp, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def as_mask( shape, np.ndarray[int] x_coord
              , np.ndarray[int] y_coord
              , np.ndarray[int] radii
              ):
    
    cdef np.ndarray[np.uint8_t, ndim=2] mask = np.zeros(shape, dtype=np.uint8)
    
    cdef int ydim, xdim
    ydim, xdim = shape
    
    cdef int j,x,y,xx,yy,size,xmin,xmax,ymin,ymax
    cdef double radius, threshold
    
    for j in range(len(x_coord)):
        xx = x_coord[j]
        yy = y_coord[j]
        radius = radii[j]
        size = radii[j]
        threshold = radius**2
        ## only consider coordinates within the box
        xmin = max(0, xx-size)
        xmax = min(xdim, xx+size+1)
        ymin = max(0, yy-size)
        ymax = min(ydim, yy+size+1)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if (y-yy)**2 + (x-xx)**2 <= threshold:
                    mask[y,x] = 1

    return mask

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def as_laplace( shape, np.ndarray[int] x_coord
               , np.ndarray[int] y_coord
               , np.ndarray[int] diameter
               , double scale=1.0
              ):
    
    cdef np.ndarray[np.float32_t, ndim=2] array = np.zeros(shape, dtype=np.float32)
    
    cdef int ydim, xdim
    ydim, xdim = shape
    
    cdef int j,x,y,xx,yy,size,xmin,xmax,ymin,ymax
    cdef double radius, threshold, d2
    
    for j in range(len(x_coord)):
        xx = x_coord[j]
        yy = y_coord[j]
        radius = diameter[j]/2.0
        size = <int> ceil(radius)
        threshold = radius**2
        ## only consider coordinates within the box
        xmin = max(0, xx-size)
        xmax = min(xdim, xx+size+1)
        ymin = max(0, yy-size)
        ymax = min(ydim, yy+size+1)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                d2 = (y-yy)**2 + (x-xx)**2
                if d2 <= threshold:
                    array[y,x] = max(array[y,x], exp(-sqrt(d2)/(scale*radius)))

    return array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def as_gaussian( shape, np.ndarray[int] x_coord
                , np.ndarray[int] y_coord
                , np.ndarray[int] diameter
                , double scale=1.0
               ):
    
    cdef np.ndarray[np.float32_t, ndim=2] array = np.zeros(shape, dtype=np.float32)
    
    cdef int ydim, xdim
    ydim, xdim = shape
    
    cdef int j,x,y,xx,yy,size,xmin,xmax,ymin,ymax
    cdef double radius, threshold, d2
    
    for j in range(len(x_coord)):
        xx = x_coord[j]
        yy = y_coord[j]
        radius = diameter[j]/2.0
        size = <int> ceil(radius)
        threshold = radius**2
        ## only consider coordinates within the box
        xmin = max(0, xx-size)
        xmax = min(xdim, xx+size+1)
        ymin = max(0, yy-size)
        ymax = min(ydim, yy+size+1)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                d2 = (y-yy)**2 + (x-xx)**2
                if d2 <= threshold:
                    array[y,x] = max(array[y,x], exp(-0.5*d2/(scale*radius)**2))

    return array







