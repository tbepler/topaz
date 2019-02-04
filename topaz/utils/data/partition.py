from __future__ import print_function, division

import numpy as np
import pandas as pd

"""
Methods for paritioning labels into train/test sets by image. Partitions are stratified on image source and number
of objects per image.
"""

def stratify(labels, nbins=5):
    strata = []
    for source,group in labels.groupby('source'):
        counts = group['count'].rank(method='first')
        buckets = pd.qcut(counts, nbins, labels=False) #, duplicates='drop')
        for ident in buckets.unique():
            I = buckets == ident
            g = group.loc[I]
            strata.append(g)
    return strata


def kfold(k, labels, nbins=5, random=np.random):
    """
    Split the labels in k train/test partitions by image.
    Labels should contain columns of source, image_name, and count, where count is the number of objects in the image.
    """

    strata = stratify(labels, nbins=nbins)
    strata = [g.iloc[random.permutation(len(g))] for g in strata]
    strata = pd.concat(strata, axis=0)

    folds = []
    for i in range(k):
        folds.append(strata.iloc[i:].iloc[::k])

    ## form the partitions and yield them
    for i in range(k):
        test = folds[i]
        train = pd.concat([folds[j] for j in range(k) if j != i], axis=0)
        yield train, test






