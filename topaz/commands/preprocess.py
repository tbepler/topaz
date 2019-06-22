from __future__ import print_function

from topaz.commands.normalize import add_arguments, main

name = 'preprocess'
help = 'downsample and normalize images in one step'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for performing image downsampling and normalization in one step')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


