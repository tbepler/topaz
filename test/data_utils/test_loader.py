import os
from logging import root

import numpy as np
from topaz.utils.data.loader import (ImageDirectoryLoader, ImageTree,
                                     LabeledImageCropDataset,
                                     LabeledRegionsDataset,
                                     SegmentedImageDataset, load_image,
                                     load_images_from_directory,
                                     load_images_from_list, load_jpeg,
                                     load_mrc, load_pil, load_png, load_tiff)


class TestImageDirectoryLoader():
    def test_init(self):
        pass

    def test_get(self):
        pass


class TestImageTree():
    def test_init(self):
        pass

    def test_get(self):
        pass


class TestLabeledImageCropDataset():
    def test_init(self):
        pass

    def test_len(self):
        pass

    def test_get_item(self):
        pass


class TestLabeledRegionsDataset():
    def test_init(self):
        pass

    def test_get_item(self):
        pass


class TestSegmentedImageDataset():
    def test_init(self):
        pass

    def test_len(self):
        pass

    def test_get_item(self):
        pass



def test_load_mrc():
    # path = 'test/test_data/EMPIAR-10025/rawdata/micrographs/14sep05c_c_00003gr_00014sq_00004hl_00004es_c.mrc'
    # image = load_mrc(path, standardize=False)
    # assert image is not None
    # array = np.asarray(image)
    # assert len(array.shape) == 2
    # assert np.isfinite(array.mean())
    # assert np.isfinite(array.std())
    pass


def test_load_mrc_standardize():
    # path = 'test/test_data/EMPIAR-10025/rawdata/micrographs/14sep05c_c_00003gr_00014sq_00004hl_00004es_c.mrc'
    # image = load_mrc(path, standardize=True)
    # assert image is not None
    # array = np.asarray(image)
    # assert len(array.shape) == 2
    # #normalization
    # assert np.isfinite(array.mean())
    # assert np.isfinite(array.std())
    # assert np.isclose(array.mean(), 0.0)
    # assert np.isclose(array.std(), 1.0)
    pass


def test_load_tiff():
    pass


def test_load_png():
    pass


def test_load_jpeg():
    pass


def test_load_pil():
    #not really necessary
    pass


def test_load_image():
    #not really necessary
    pass


def test_load_images_from_directory():
    # names = os.listdir('test/test_data/EMPIAR-10025/rawdata/micrographs/')
    # #remove extensions
    # names = [name.split('.')[0] for name in names]
    # rootdir = 'test/test_data/EMPIAR-10025/rawdata/micrographs/'
    # images = load_images_from_directory(names, rootdir)
    
    # for im in images.values():
    #     assert im is not None
    #     array = np.asarray(im)
    #     assert len(array.shape) == 2
    pass


def test_load_images_from_list():
    # rootdir = 'test/test_data/EMPIAR-10025/rawdata/micrographs/'
    # names = os.listdir(rootdir)
    # #remove extensions
    # names = [name.split('.')[0] for name in names]
    # paths = [os.path.join(rootdir, file) for file in os.listdir(rootdir)]    
    # images = load_images_from_list(names, paths)
    
    # for im in images.values():
    #     assert im is not None
    #     array = np.asarray(im)
    #     assert len(array.shape) == 2
    pass