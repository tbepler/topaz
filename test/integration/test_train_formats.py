"""`topaz train` on non-MRC micrographs (a full CLI run, in-process).

Regression: from 0.3.0 to 0.3.20 training only globbed *.mrc and read MRC headers, so a TIFF/PNG
micrograph directory crashed with ZeroDivisionError (v0.2.5 trained on them)."""
import numpy as np
import pandas as pd
import pytest

from _support import run_topaz_inprocess
from topaz.utils.data.loader import load_image
from topaz.utils.image import save_image

pytestmark = pytest.mark.cli


def convert_fixture_micrographs(fixtures_dir, dest, ext, dtype=None):
    dest.mkdir()
    for split in ('train', 'test'):
        (dest / split).mkdir()
        for path in sorted((fixtures_dir / '2d' / split).glob('*.mrc')):
            x = load_image(str(path), make_image=False, return_header=False)
            if dtype is not None:  # integer-valued TIFF, as from some detectors/exports
                x = np.round((x - x.min()) / (x.max() - x.min()) * np.iinfo(dtype).max).astype(dtype)
            save_image(x, str(dest / split / f'{path.stem}.{ext}'))
    return dest


@pytest.mark.parametrize('ext, dtype', [('tiff', None), ('tiff', np.uint16), ('png', None)])
def test_train_on_directory_of_images(fixtures_dir, tmp_path, capsys, ext, dtype):
    images = convert_fixture_micrographs(fixtures_dir, tmp_path / 'imgs', ext, dtype)
    curve = tmp_path / 'curve.txt'
    run_topaz_inprocess('train', '-d', -1, '-n', 15, '-r', 3, '-m', 'conv31', '--units', 8, '--no-pretrained',
                        '--train-images', images / 'train', '--train-targets', fixtures_dir / '2d' / 'particles_train.txt',
                        '--test-images', images / 'test', '--test-targets', fixtures_dir / '2d' / 'particles_test.txt',
                        '--num-epochs', 1, '--epoch-size', 2, '--minibatch-size', 64, '-o', curve)
    log = capsys.readouterr().err
    assert 'Loaded 7 training micrographs' in log
    assert 'Loaded 3 testing micrographs' in log
    table = pd.read_csv(curve, sep='\t')
    test_rows = table[table.split == 'test']
    assert len(test_rows) == 1
    assert np.isfinite(test_rows[['loss', 'precision', 'tpr', 'fpr', 'auprc']].to_numpy(dtype=float)).all()
