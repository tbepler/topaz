from __future__ import print_function, division

def insize_from_outsize(layers, outsize):
    """ calculates in input size of a convolution stack given the layers and output size """
    for layer in layers[::-1]:
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size # assume square
            if type(kernel_size) is tuple:
                kernel_size = kernel_size[0]
        else:
            kernel_size = 1
        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) is tuple:
                stride = stride[0]
        else:
            stride = 1
        if hasattr(layer, 'padding'):
            pad = layer.padding
            if type(pad) is tuple:
                pad = pad[0]
        else:
            pad = 0
        if hasattr(layer, 'dilation'):
            dilation = layer.dilation
            if type(dilation) is tuple:
                dilation = dilation[0]
        else:
            dilation = 1

        outsize = (outsize-1)*stride + 1 + (kernel_size-1)*dilation - 2*pad 
    return outsize

