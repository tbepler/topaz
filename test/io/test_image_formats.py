"""Image file formats through topaz's own writer/reader, plus regression tests."""
import numpy as np
import pytest

from topaz.utils.data.loader import load_image
from topaz.utils.image import save_image

QUANT_STEP = 6 / 255  # PNG/JPEG store [-3, 3] quantized to uint8


def gradient_image(shape=(32, 48)):
    """Smooth ramp spanning [-3, 3] (the range save_png/save_jpeg quantize)."""
    return np.linspace(-3, 3, shape[0] * shape[1], dtype=np.float32).reshape(shape)


def load_array(path, **kwargs):
    return load_image(str(path), make_image=False, return_header=False, **kwargs)


def test_tiff_roundtrip_exact(tmp_path):
    x = np.random.default_rng(0).standard_normal((32, 48)).astype(np.float32)
    save_image(x, str(tmp_path / 'mic.tiff'))
    y = load_array(tmp_path / 'mic.tiff')
    assert isinstance(y, np.ndarray) and y.dtype == np.float32
    np.testing.assert_array_equal(y, x)


def test_png_roundtrip_unquantized(tmp_path):
    """PNG loads back as float values on the original scale, not raw uint8.
    Regression: from 2022 until 0.3.21 load_png returned the raw PIL image (0-255)."""
    x = gradient_image()
    save_image(x, str(tmp_path / 'mic.png'))
    y = load_array(tmp_path / 'mic.png')
    assert isinstance(y, np.ndarray) and y.dtype == np.float32
    assert np.abs(y - x).max() <= QUANT_STEP


def test_jpeg_roundtrip_unquantized(tmp_path):
    """JPEG is lossy; check the scale and dtype, and that values track the source closely."""
    x = gradient_image()
    save_image(x, str(tmp_path / 'mic.jpg'))
    y = load_array(tmp_path / 'mic.jpg')
    assert isinstance(y, np.ndarray) and y.dtype == np.float32
    assert -3 <= y.min() and y.max() <= 3
    assert np.abs(y - x).mean() < 2 * QUANT_STEP


@pytest.mark.parametrize('ext', ['tiff', 'png', 'jpg'])
def test_standardize(tmp_path, ext):
    x = gradient_image()
    save_image(x, str(tmp_path / f'mic.{ext}'))
    y = load_array(tmp_path / f'mic.{ext}', standardize=True)
    assert y.mean() == pytest.approx(0, abs=1e-5)
    assert y.std() == pytest.approx(1, rel=1e-5)


@pytest.mark.parametrize('ext', ['tiff', 'png', 'jpg'])
def test_make_image_returns_pil(tmp_path, ext):
    from PIL import Image
    save_image(gradient_image(), str(tmp_path / f'mic.{ext}'))
    assert isinstance(load_image(str(tmp_path / f'mic.{ext}'), make_image=True, return_header=False), Image.Image)


@pytest.mark.parametrize('source_ext', ['tiff', 'png'])
def test_headerless_image_saved_as_mrc(tmp_path, source_ext):
    """A TIFF/PNG has no MRC header; re-saving it as MRC (what downsample/preprocess/normalize/denoise
    do by default) passes header=None, extended_header=None through to mrc.write.
    Regression: that crashed with TypeError from 2022 until 0.3.21."""
    save_image(gradient_image(), str(tmp_path / f'mic.{source_ext}'))
    loaded = load_array(tmp_path / f'mic.{source_ext}')

    save_image(loaded, str(tmp_path / 'mic.mrc'), header=None, extended_header=None)
    roundtrip = load_array(tmp_path / 'mic.mrc')
    np.testing.assert_array_equal(roundtrip, loaded)
