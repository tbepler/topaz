from __future__ import print_function,division

import sys
import os
import pandas as pd
import numpy as np

import topaz.utils.star as star
import topaz.utils.files as file_utils
from topaz.utils.conversions import mirror_y_axis


name = 'convert'
help = 'convert particle coordinate files between various formats automatically'


def add_arguments(parser):
    parser.add_argument('files', nargs='+', help='path to input particle file(s). when multiple input files are given, they are concatentated into a single output file.')
    parser.add_argument('-o', '--output', help='path to output particle file (default: stdout)')

    parser.add_argument('--from', dest='_from', choices=['auto', 'coord', 'star', 'box'], default='auto'
                       , help='file format of the INPUT file (default: detect format automatically based on file extension)')
    parser.add_argument('--to', choices=['auto', 'coord', 'star', 'json', 'box'], default='auto'
                       , help='file format of the OUTPUT file. NOTE: when converting to JSON or BOX formats, OUTPUT must specify the destination directory. (default: detect format automatically based on file extension)')

    # arguments for file format specific parameters
    parser.add_argument('-t', '--threshold', type=float, default=-np.inf, help='threshold the particles by score (optional)')
    parser.add_argument('--invert-y', action='store_true', help='invert (mirror) the y-axis particle coordinates. requires also specifying --imagedir.')
    parser.add_argument('--imagedir', help='directory of images. only required to invert the y-axis - sometimes necessary for particles picked on .tiff images')
    parser.add_argument('--image-ext', default='.mrc', help='image file extension. required for converting to STAR and BOX formats and to find images when --invert-y is set. (default=.mrc)')
    parser.add_argument('--boxsize', default=0, type=int, help='size of particle boxes. required for converting to BOX format.')

    # verbose output?
    parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity of information printed (default: 0)')

    return parser

def main(args):

    verbose = args.verbose

    form = args._from
    from_forms = [form for _ in range(len(args.files))]

    # detect the input file formats
    if form == 'auto':
        try:
            from_forms = [file_utils.detect_format(path) for path in args.files]
        except file_utils.UnknownFormatError as e:
            print('Error: unrecognized input coordinates file extension ('+e.ext+')', file=sys.stderr)
            sys.exit(1)
    if verbose > 0:
        formats_detected = set(from_forms)
        print('# INPUT formats detected: '+str(list(formats_detected)), file=sys.stderr)

    # determine the output file format
    output_path = args.output
    output = None
    to_form = args.to
    if output_path is None:
        output = sys.stdout
        # if output is to stdout and form is not set
        # then raise an error
        if to_form == 'auto':
            print('Error: writing file to stdout and no output format (--to) set! Please tell me what format to write!')
            sys.exit(1)
        if to_form == 'box' or form == 'json':
            print('Error: writing BOX or JSON output files requires a destination directory. Please set the --output parameter!')
            sys.exit(1)

    image_ext = args.image_ext
    boxsize = args.boxsize
    if to_form == 'auto':
        # first check for directory
        if output_path[-1] == '/':
            # image-ext must be set for these file formats
            if image_ext is None:
                print('Error: writing BOX or JSON output files requires setting the image file extension!')
                sys.exit(1)
            # format is either json or box, check for boxsize to decide
            if boxsize > 0:
                # write boxes!
                if verbose > 0:
                    print('# Detected output format is BOX, because OUTPUT is a directory and boxsize > 0.', file=sys.stderr)
                to_form = 'box'
            else:
                if verbose > 0:
                    print('# Detected output format is JSON, because OUTPUT is a directory and no boxsize set.', file=sys.stderr)
                to_form = 'json'
        else:
            try:
                to_form = file_utils.detect_format(output_path)
            except file_utils.UnkownFormatError as e:
                print('Error: unrecognized output coordinates file extension ('+e.ext+')', file=sys.stderr)
                sys.exit(1)
    if verbose > 0:
        print('# OUTPUT format: ' + to_form)

    # read the input files
    dfs = []
    for i in range(len(args.files)):
        path = args.files[i]
        coords = file_utils.read_coordinates(path, format=from_forms[i])
        dfs.append(coords)
    coords = pd.concat(dfs, axis=0)

    # threshold particles by score (if there is a score)
    t = args.threshold
    if 'score' in coords.columns:
        coords = coords.loc[coords['score'] >= t]

    # invert y-axis coordinates if specified
    invert_y = args.invert_y
    if invert_y:
        if args.imagedir is None:
            print('Error: --imagedir must specify the directory of images in order to mirror the y-axis coordinates', file=sys.stderr)
            sys.exit(1)
        dfs = []
        for image_name,group in coords.groupby('image_name'):
            impath = os.path.join(args.imagedir, image_name) + '.' + args.image_ext
            # use glob incase image_ext is '*'
            impath = glob.glob(impath)[0]
            im = load_image(impath)
            height = im.height

            group = mirror_y_axis(group, height)
            dfs.append(group)
        coords = pd.concat(dfs, axis=0)

    # output file format is decided and coordinates are processed, now write files
    if output is None and to_form != 'box' and to_form != 'json':
        output = open(output_path, 'w')
    if to_form == 'box' or to_form == 'json':
        output = output_path

    file_utils.write_coordinates(output, coords, format=to_form, boxsize=boxsize, image_ext=image_ext)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script to ' + help)
    add_argument(parser)
    args = parser.parse_args()
    main(args)

