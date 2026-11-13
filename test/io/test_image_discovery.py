"""Finding training/test micrographs in a directory or list file, in any supported format."""
import numpy as np
import pandas as pd
import pytest

from _support import write_mrc
from topaz import training
from topaz.utils.image import save_image


def make_dir(tmp_path, files):
    d = tmp_path / 'imgs'
    d.mkdir()
    for name, shape in files.items():
        if name.endswith('.mrc'):
            write_mrc(d / name, np.zeros(shape, dtype=np.float32))
        elif name.endswith(('.tiff', '.png', '.jpg')):
            save_image(np.zeros(shape, dtype=np.float32), str(d / name))
        else:
            (d / name).write_text('not an image')
    return d


def test_directory_finds_every_supported_format_sorted(tmp_path):
    d = make_dir(tmp_path, {'c.mrc': (8, 8), 'a.tiff': (8, 8), 'b.png': (8, 8), 'd.jpg': (8, 8), 'notes.txt': None})
    table = training.find_images_in_directory(str(d))
    assert table.image_name.tolist() == ['a', 'b', 'c', 'd']
    assert [p.rsplit('/', 1)[1] for p in table.path] == ['a.tiff', 'b.png', 'c.mrc', 'd.jpg']


@pytest.mark.parametrize('image_ext', ['.tiff', 'tiff'])
def test_image_ext_selects_one_format(tmp_path, image_ext):
    d = make_dir(tmp_path, {'mic.mrc': (8, 8), 'mic.tiff': (8, 8), 'other.tiff': (8, 8)})
    table = training.find_images_in_directory(str(d), image_ext)
    assert table.image_name.tolist() == ['mic', 'other'] and all(p.endswith('.tiff') for p in table.path)


def test_same_name_in_two_formats_is_an_error(tmp_path):
    d = make_dir(tmp_path, {'mic.mrc': (8, 8), 'mic.tiff': (8, 8)})
    with pytest.raises(ValueError, match='--image-ext'):
        training.find_images_in_directory(str(d))


@pytest.mark.parametrize('name', ['m.mrc', 'm.tiff', 'm.png'])
def test_image_num_pixels(tmp_path, name):
    d = make_dir(tmp_path, {name: (24, 40)})
    assert training.image_num_pixels(str(d / name)) == 24 * 40


def test_image_num_pixels_3d_mrc(tmp_path):
    write_mrc(tmp_path / 'v.mrc', np.zeros((5, 6, 7), dtype=np.float32))
    assert training.image_num_pixels(str(tmp_path / 'v.mrc')) == 5 * 6 * 7


def test_directory_and_list_file_agree(tmp_path):
    d = make_dir(tmp_path, {'a.mrc': (8, 8), 'b.tiff': (8, 8)})
    targets = pd.DataFrame({'image_name': ['a', 'b'], 'x_coord': [1, 2], 'y_coord': [3, 4]})
    listing = tmp_path / 'images.txt'
    training.find_images_in_directory(str(d))[['image_name', 'path']].to_csv(listing, sep='\t', index=False)
    from_dir = training.convert_path_to_grouped_list(str(d), targets.copy())
    from_list = training.convert_path_to_grouped_list(str(listing), targets.copy())
    assert from_dir == from_list == [[str(d / 'a.mrc'), str(d / 'b.tiff')]]
