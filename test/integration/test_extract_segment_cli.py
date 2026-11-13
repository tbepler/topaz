"""`topaz extract` and `topaz segment` through the CLI."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from _support import require, run_topaz_inprocess

pytestmark = pytest.mark.cli

THRESHOLD = 1.5


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    return out


def test_extract_finds_known_peaks(fixtures_dir, output_dir, ndim):
    """Extracting from a pre-scored map (-m none) returns exactly the known peaks above the threshold, with their
    scores. At radius 5 both peaks of the pair 6 apart are kept.
    Regression: an existing directory as --output crashed (sys.path.join) through 0.3.20."""
    require('extract_3d', ndim)
    name = f'scores{ndim}d'
    run_topaz_inprocess('extract', '-m', 'none', '--dims', ndim, '-r', 5, '-t', THRESHOLD, '-o', output_dir,
                        fixtures_dir / 'scores' / f'{name}.mrc')
    columns = ['x_coord', 'y_coord', 'z_coord'][:ndim] + ['score']
    peaks = pd.read_csv(fixtures_dir / 'scores' / 'peaks.txt', sep='\t')
    expected = peaks[(peaks.image_name == name) & (peaks.score > THRESHOLD)][columns]
    picks = pd.read_csv(output_dir / 'extracted_particles.txt', sep='\t')
    assert list(picks.columns) == ['image_name'] + columns
    assert (picks.image_name == name).all()
    assert len(expected) == 3
    np.testing.assert_allclose(picks[columns].sort_values(columns).to_numpy(),
                               expected.sort_values(columns).to_numpy().astype(float))


def test_segment_in_patches_matches_whole_image(fixtures_dir, tmp_path):
    """`segment --patch-size` writes the same score map as segmenting the whole micrograph at once.
    Regression: --patch-size crashed (an unexpected keyword argument) through 0.3.20."""
    micrograph = fixtures_dir / '2d' / 'test' / 'mic07.mrc'
    for label, patch in [('whole', []), ('patched', ['--patch-size', 48])]:
        run_topaz_inprocess('segment', '-m', 'resnet8', '-d', -1, *patch, '-o', tmp_path / label, micrograph)
    whole, patched = (np.array(Image.open(tmp_path / label / 'mic07.tiff')) for label in ('whole', 'patched'))
    assert whole.shape == (128, 128)
    assert np.ptp(whole) > 1  # a real score map, not a constant
    np.testing.assert_allclose(patched, whole, atol=1e-5)
