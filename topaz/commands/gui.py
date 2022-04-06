from __future__ import print_function

import os
import sys
import argparse

name = 'gui'
help = 'opens the topaz GUI in a web browser'


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser('Opens the topaz GUI in a web browser.')
    return parser


def main(args):
    # open the GUI
    import webbrowser

    # where is the GUI
    root = os.path.dirname(__file__) # this is the commands dir
    root = os.path.dirname(root) # now in the topaz root dir
    gui_path = os.path.join(root, 'gui', 'topaz.html')

    # open the GUI
    webbrowser.open('file://' + os.path.realpath(gui_path), new=2)



if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)