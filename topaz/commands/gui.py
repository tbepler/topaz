from __future__ import print_function

import os
import sys
import topaz.main

name = 'gui'
help = 'opens the topaz GUI in a web browser'


def add_arguments(parser):
    return parser


def main(args):
    # open the GUI
    import webbrowser

    # where is the GUI
    root = os.path.dirname(topaz.main.__file__)
    root = os.path.dirname(root)
    gui_path = os.path.join(root, 'gui', 'full.html')

    # open the GUI
    webbrowser.open('file://' + os.path.realpath(gui_path), new=2)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Opens the topaz GUI in a web browser.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


