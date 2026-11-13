"""`topaz convert` through the CLI."""
import numpy as np
import pandas as pd
import pytest

from _support import run_topaz_inprocess, write_mrc
from topaz.utils.image import save_image

pytestmark = pytest.mark.cli

HEIGHT, WIDTH = 100, 120
PARTICLES = pd.DataFrame({'image_name': ['mic', 'mic', 'mic'], 'x_coord': [10, 60, 119], 'y_coord': [20, 0, 99]})


@pytest.mark.parametrize('image_format, ext_arg', [('mrc', None), ('mrc', 'mrc'), ('mrc', '.mrc'), ('tiff', 'tiff')])
def test_invert_y(tmp_path, image_format, ext_arg):
    """--invert-y mirrors y about the image's last row; applying it twice is the identity.
    Regression: it crashed on every input through 0.3.20 (path with a doubled dot, MRC loaded as a tuple, DataFrame.clone)."""
    image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if image_format == 'mrc':
        write_mrc(tmp_path / 'mic.mrc', image)
    else:
        save_image(image, str(tmp_path / 'mic.tiff'))
    PARTICLES.to_csv(tmp_path / 'in.txt', sep='\t', index=False)
    ext = [] if ext_arg is None else ['--image-ext', ext_arg]

    run_topaz_inprocess('convert', '--invert-y', '--imagedir', tmp_path, *ext, '-o', tmp_path / 'once.txt', tmp_path / 'in.txt')
    once = pd.read_csv(tmp_path / 'once.txt', sep='\t')
    np.testing.assert_array_equal(once.y_coord, HEIGHT - 1 - PARTICLES.y_coord)
    np.testing.assert_array_equal(once.x_coord, PARTICLES.x_coord)

    run_topaz_inprocess('convert', '--invert-y', '--imagedir', tmp_path, *ext, '-o', tmp_path / 'twice.txt', tmp_path / 'once.txt')
    twice = pd.read_csv(tmp_path / 'twice.txt', sep='\t')
    np.testing.assert_array_equal(twice.y_coord, PARTICLES.y_coord)
