"""MRC reading and writing with topaz's own implementation and header bookkeeping on write."""
import numpy as np
import pytest

from _support import run_topaz_inprocess, write_mrc
from topaz import mrc
from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample_file


def parse_file(path):
    with open(path, 'rb') as f:
        return mrc.parse(f.read())


def data_stats(array):
    a = np.asarray(array, dtype=np.float32)
    return {'amin': a.min(), 'amax': a.max(), 'amean': a.mean(), 'rms': a.std()}


@pytest.mark.parametrize('shape', [(5, 7), (4, 5, 7)], ids=['2d', '3d'])
def test_write_read_roundtrip(tmp_path, shape):
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    write_mrc(tmp_path / 'x.mrc', x)
    array, header, extended = parse_file(tmp_path / 'x.mrc')
    np.testing.assert_array_equal(array, x)
    nz = shape[0] if len(shape) == 3 else 1
    assert (header.nx, header.ny, header.nz, header.mode, header.next) == (shape[-1], shape[-2], nz, 2, 0)
    assert extended == b''
    for key, value in data_stats(x).items():
        assert getattr(header, key) == pytest.approx(float(value), rel=1e-5), key


@pytest.mark.parametrize('dtype, mode', [(np.int8, 0), (np.int16, 1), (np.float32, 2), (np.uint16, 6), (np.float16, 12)])
def test_read_modes(tmp_path, dtype, mode):
    """Files from other software use other modes; topaz writes only float32 but must read these."""
    x = (np.arange(6 * 8).reshape(1, 6, 8) % 50).astype(dtype)
    header = mrc.make_header(x.shape, (8, 6, 1), (90, 90, 90))._replace(mode=mode)
    (tmp_path / 'x.mrc').write_bytes(mrc.header_struct.pack(*header) + x.tobytes())
    array, parsed, _ = parse_file(tmp_path / 'x.mrc')
    assert array.dtype == np.dtype(dtype) and parsed.mode == mode
    np.testing.assert_array_equal(array, x[0])
    loaded = load_image(str(tmp_path / 'x.mrc'), make_image=False, return_header=False)
    np.testing.assert_array_equal(loaded, x[0].astype(loaded.dtype))
    if dtype == np.float16:
        assert loaded.dtype == np.float32  # the loader upcasts half precision


def test_extended_header_roundtrip(tmp_path):
    x = np.ones((1, 4, 4), dtype=np.float32)
    ext = bytes(range(256)) * 4
    with open(tmp_path / 'x.mrc', 'wb') as f:
        mrc.write(f, x, extended_header=ext)
    array, header, extended = parse_file(tmp_path / 'x.mrc')
    assert header.next == len(ext) and extended == ext
    np.testing.assert_array_equal(array, x[0])


@pytest.mark.parametrize('name', ['v025_float32_2d', 'v025_float32_3d'])
def test_read_v025_written_mrc(fixtures_dir, name):
    """Files written by v0.2.5 read back identically."""
    expected = np.load(fixtures_dir / 'io' / f'{name}.npy')
    array, header, _ = parse_file(fixtures_dir / 'io' / f'{name}.mrc')
    np.testing.assert_array_equal(array, expected)
    np.testing.assert_array_equal(load_image(str(fixtures_dir / 'io' / f'{name}.mrc'), make_image=False, return_header=False),
                                  expected)


def test_write_with_header_describes_new_data(tmp_path):
    """Passing a source header keeps its metadata but must describe the array actually written.
    Regression: through 0.3.20, a passed header kept its old amin/amax/amean/rms (and relied on callers for nx/ny)."""
    source = np.random.default_rng(1).standard_normal((1, 40, 50)).astype(np.float32) * 5 + 2
    header = mrc.make_header(source.shape, (50 * 1.06, 40 * 1.06, 1.06), (90, 90, 90), **{
        'dmin': source.min(), 'dmax': source.max(), 'dmean': source.mean(), 'rms': source.std()})
    new = np.random.default_rng(2).standard_normal((1, 20, 25)).astype(np.float32)
    with open(tmp_path / 'x.mrc', 'wb') as f:
        mrc.write(f, new, header=header)
    array, written, _ = parse_file(tmp_path / 'x.mrc')
    np.testing.assert_array_equal(array, new[0])
    assert (written.nx, written.ny, written.nz, written.mode) == (25, 20, 1, 2)
    for key, value in data_stats(new).items():
        assert getattr(written, key) == pytest.approx(float(value), rel=1e-5), key
    assert (written.xlen, written.ylen, written.alpha) == pytest.approx((header.xlen, header.ylen, header.alpha))


def realistic_header(shape, pixel_size):
    """MRC2014-style header: sampling equals the grid size, cell length = size * pixel size."""
    nz, ny, nx = (1,) + tuple(shape) if len(shape) == 2 else shape
    return mrc.make_header((nz, ny, nx), (nx * pixel_size, ny * pixel_size, nz * pixel_size), (90, 90, 90))._replace(
        mx=nx, my=ny, mz=nz)


@pytest.mark.parametrize('convention', ['mrc2014', 'v025_unit_cell'])
def test_resize_header_scales_pixel_size(convention):
    if convention == 'mrc2014':
        header = realistic_header((256, 320), 1.06)
    else:  # v0.2.5 wrote cell lengths of 1 and sampling of 1
        header = mrc.make_header((1, 256, 320), (1, 1, 1), (0, 0, 0))
    before = (header.xlen / header.mx, header.ylen / header.my)
    resized = mrc.resize_header(header, (64, 80))
    assert (resized.nx, resized.ny) == (80, 64)
    after = (resized.xlen / resized.mx, resized.ylen / resized.my)
    assert after == pytest.approx((before[0] * 4, before[1] * 4))


@pytest.mark.parametrize('shape, new_shape', [((256, 320), (64, 80)), ((8, 256, 320), (4, 64, 80))])
def test_resize_header_without_pixel_size(shape, new_shape):
    """Many real MRC files (the tutorial micrographs among them) leave the sampling and cell lengths at 0.
    The grid size must still be updated, and the cell lengths stay 0 (no pixel size to carry over).
    Regression: this divided by zero, so `topaz preprocess` crashed on such files."""
    nz, ny, nx = (1,) + shape if len(shape) == 2 else shape
    header = mrc.make_header((nz, ny, nx), (0, 0, 0), (90, 90, 90))._replace(mx=0, my=0, mz=0)
    resized = mrc.resize_header(header, new_shape)
    assert (resized.nx, resized.ny) == new_shape[::-1][:2]
    assert (resized.mx, resized.my) == (resized.nx, resized.ny)
    assert (resized.xlen, resized.ylen, resized.zlen) == (0, 0, 0)
    if len(new_shape) == 3:
        assert (resized.nz, resized.mz) == (new_shape[0], new_shape[0])


def test_resize_header_sampling_missing_but_cell_length_present():
    """A missing sampling is taken to equal the grid size, so the recorded cell length still gives the right pixel size."""
    header = realistic_header((256, 320), 1.06)._replace(mx=0, my=0)
    resized = mrc.resize_header(header, (64, 80))
    assert (resized.xlen / resized.mx, resized.ylen / resized.my) == pytest.approx((4.24, 4.24))


def test_resize_header_with_empty_grid():
    """A header whose grid size and sampling are both 0 has nothing to rescale by: no error, cell lengths kept."""
    header = realistic_header((256, 320), 1.06)._replace(nx=0, ny=0, mx=0, my=0)
    resized = mrc.resize_header(header, (64, 80))
    assert (resized.nx, resized.ny, resized.mx, resized.my) == (80, 64, 80, 64)
    assert (resized.xlen, resized.ylen) == pytest.approx((header.xlen, header.ylen))


def test_downsample_file_header(tmp_path):
    """Regression: through 0.3.20 a downsampled MRC kept the original pixel size and data statistics."""
    x = np.random.default_rng(0).standard_normal((1, 256, 256)).astype(np.float32)
    with open(tmp_path / 'mic.mrc', 'wb') as f:
        mrc.write(f, x, header=realistic_header((256, 256), 1.06))
    small = downsample_file(str(tmp_path / 'mic.mrc'), 4, str(tmp_path / 'small.mrc'), False)
    array, header, _ = parse_file(tmp_path / 'small.mrc')
    assert array.shape == (64, 64)
    np.testing.assert_array_equal(array, small)
    assert header.xlen / header.mx == pytest.approx(4.24)
    assert header.rms == pytest.approx(float(small.std()), rel=1e-5)


def test_preprocess_cli_header(tmp_path):
    x = np.random.default_rng(0).standard_normal((1, 128, 128)).astype(np.float32) * 3 + 10
    with open(tmp_path / 'mic.mrc', 'wb') as f:
        mrc.write(f, x, header=realistic_header((128, 128), 0.8))
    out = tmp_path / 'out'
    out.mkdir()
    run_topaz_inprocess('preprocess', '-s', 4, '--affine', '-o', out, tmp_path / 'mic.mrc')
    array, header, _ = parse_file(out / 'mic.mrc')
    assert array.shape == (32, 32)
    assert header.xlen / header.mx == pytest.approx(3.2)
    assert header.amean == pytest.approx(float(array.mean()), abs=1e-5)
    assert header.rms == pytest.approx(float(array.std()), rel=1e-5)
