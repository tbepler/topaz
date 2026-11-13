"""Particle coordinate file formats.

The oracle for each format is the file v0.2.5's `topaz convert` wrote from the ground-truth table
(test/fixtures/2d, see generate.py). Files are compared as parsed tables, not bytes.
"""
import json

import numpy as np
import pandas as pd
import pytest

from topaz.utils import files as file_utils

KEY = ['image_name', 'x_coord', 'y_coord']


def write_file(path, table, fmt):
    """write_coordinates takes an open file for single-file formats (as `topaz convert` passes it)."""
    with open(path, 'w') as f:
        file_utils.write_coordinates(f, table, format=fmt)


def ground_truth(fixtures_dir):
    return pd.read_csv(fixtures_dir / '2d' / 'particles.txt', sep='\t', dtype={'image_name': str})


def normalized(table, columns=KEY):
    t = table[columns].copy()
    for c in columns:
        if c.endswith('_coord'):
            t[c] = t[c].astype(float)
    return t.sort_values(columns).reset_index(drop=True)


def assert_same_particles(actual, expected, columns=KEY):
    pd.testing.assert_frame_equal(normalized(actual, columns), normalized(expected, columns), check_dtype=False)


@pytest.mark.parametrize('ext, fmt', [('.star', 'star'), ('.box', 'box'), ('.json', 'json'), ('.csv', 'csv'),
                                      ('.txt', 'coord'), ('.tab', 'coord')])
def test_detect_format(ext, fmt):
    assert file_utils.detect_format(f'particles{ext}') == fmt


def test_detect_format_unknown():
    with pytest.raises(file_utils.UnknownFormatError):
        file_utils.detect_format('particles.xyz')


# Read v0.2.5-written files

@pytest.mark.parametrize('name', ['particles.star', 'particles.csv', 'particles.txt'])
def test_read_v025_file(fixtures_dir, name):
    """Regression (csv): reading VIA csv crashed on pandas >= 2 (positional axis in DataFrame.drop) until 0.3.21."""
    table = file_utils.read_coordinates(str(fixtures_dir / '2d' / name))
    assert_same_particles(table, ground_truth(fixtures_dir))


def test_read_v025_box_files(fixtures_dir):
    boxes = [file_utils.read_coordinates(str(p)) for p in sorted((fixtures_dir / '2d' / 'box').glob('*.box'))]
    assert len(boxes) == 10
    assert_same_particles(pd.concat(boxes, ignore_index=True), ground_truth(fixtures_dir))


# Write every format and compare with v0.2.5's output for the same table

@pytest.mark.parametrize('name', ['particles.star', 'particles.csv'])
def test_write_matches_v025(fixtures_dir, tmp_path, name):
    out = tmp_path / name
    write_file(out, ground_truth(fixtures_dir), file_utils.detect_format(name))
    ours = file_utils.read_coordinates(str(out))
    theirs = file_utils.read_coordinates(str(fixtures_dir / '2d' / name))
    assert list(ours.columns) == list(theirs.columns)
    assert_same_particles(ours, theirs)


def test_write_box_matches_v025(fixtures_dir, tmp_path):
    file_utils.write_coordinates(str(tmp_path), ground_truth(fixtures_dir), format='box', boxsize=24)
    for golden in sorted((fixtures_dir / '2d' / 'box').glob('*.box')):
        np.testing.assert_array_equal(file_utils.read_box(str(tmp_path / golden.name)), file_utils.read_box(str(golden)))


def test_write_json_matches_v025(fixtures_dir, tmp_path):
    file_utils.write_coordinates(str(tmp_path), ground_truth(fixtures_dir), format='json')
    goldens = sorted((fixtures_dir / '2d' / 'json').glob('*.json'))
    assert len(goldens) == 10
    for golden in goldens:
        assert json.loads((tmp_path / golden.name).read_text()) == json.loads(golden.read_text())


def test_box_suffix(fixtures_dir, tmp_path):
    file_utils.write_coordinates(str(tmp_path), ground_truth(fixtures_dir), format='box', boxsize=24, suffix='_picks')
    assert sorted(p.name for p in tmp_path.iterdir()) == [f'mic{i:02d}_picks.box' for i in range(10)]


# Round trips, including scores and 3D

def scored_table(ndim=2):
    rng = np.random.default_rng(0)
    t = pd.DataFrame({'image_name': ['a'] * 4 + ['b'] * 3, 'x_coord': rng.integers(0, 500, 7),
                      'y_coord': rng.integers(0, 500, 7), 'score': rng.normal(size=7).round(4)})
    if ndim == 3:
        t.insert(3, 'z_coord', rng.integers(0, 200, 7))
    return t


@pytest.mark.parametrize('fmt, ext', [('coord', '.txt'), ('star', '.star'), ('csv', '.csv')])
def test_roundtrip_with_scores(tmp_path, fmt, ext):
    table = scored_table()
    write_file(tmp_path / f'p{ext}', table, fmt)
    back = file_utils.read_coordinates(str(tmp_path / f'p{ext}'))
    assert_same_particles(back, table, KEY)
    merged = normalized(table, KEY + ['score']).merge(normalized(back, KEY + ['score']), on=KEY)
    np.testing.assert_allclose(merged.score_x, merged.score_y, rtol=1e-6)


@pytest.mark.parametrize('writer', ['write_coordinates', 'write_table'])
def test_roundtrip_3d_keeps_z(tmp_path, writer):
    """Regression: the coord-format writers dropped z_coord (e.g. `extract --dims 3 --per-micrograph`) until 0.3.21."""
    table = scored_table(ndim=3)
    path = tmp_path / 'p.txt'
    if writer == 'write_coordinates':
        write_file(path, table, 'coord')
    else:
        with open(path, 'w') as f:
            file_utils.write_table(f, table, format='coord')
    back = file_utils.read_coordinates(str(path))
    assert list(back.columns) == ['image_name', 'x_coord', 'y_coord', 'z_coord', 'score']
    assert_same_particles(back, table, KEY + ['z_coord'])


# Image names

@pytest.mark.parametrize('name, expected', [('mic', 'mic'), ('mic.mrc', 'mic'), ('mic.tiff', 'mic'), ('mic.tif', 'mic'),
                                            ('mic.png', 'mic'), ('a.b.jpeg', 'a.b'), ('mic.v2', 'mic.v2'),
                                            ('14sep05c_00024sq_00003hl_00002es.frames', '14sep05c_00024sq_00003hl_00002es.frames')])
def test_image_extension_stripping(tmp_path, name, expected):
    """Only real image extensions are removed; dots that are part of a micrograph name stay."""
    path = tmp_path / 'p.txt'
    pd.DataFrame({'image_name': [name], 'x_coord': [1], 'y_coord': [2]}).to_csv(path, sep='\t', index=False)
    assert file_utils.read_coordinates(str(path)).image_name.tolist() == [expected]


def test_numeric_image_names_stay_strings(tmp_path):
    path = tmp_path / 'p.txt'
    path.write_text('image_name\tx_coord\ty_coord\n0001\t1\t2\n')
    assert file_utils.read_coordinates(str(path)).image_name.tolist() == ['0001']
