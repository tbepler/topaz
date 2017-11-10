from __future__ import print_function,division

import sys

def report(*args):
    print('#', *args, file=sys.stderr)
