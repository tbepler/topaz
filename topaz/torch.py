from __future__ import absolute_import, print_function, division

import torch

def set_num_threads(num_threads):
    if num_threads < 0: # use all cores
        from multiprocessing import cpu_count
        num_threads = cpu_count()
    if num_threads > 0:
        # set number of threads in pytorch
        torch.set_num_threads(num_threads)
    return num_threads
