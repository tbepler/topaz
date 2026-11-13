"""Pixel/voxel counts and shapes of a particle of radius r.

The oracle is the number of integer lattice points inside a disc/sphere, computed here by brute
force and pinned against a literal table. Three independent topaz implementations of the same
footprint (stats.pixels_given_radius, training.expand_target_points, utils.picks.as_mask) must
all agree with it.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from topaz.stats import calculate_pi, pixels_given_radius
from topaz.training import expand_target_points
from topaz.utils.picks import as_mask

# integer lattice points with |d|^2 <= r^2 (Gauss circle / sphere problem)
LATTICE_COUNTS = {
    2: {1: 5, 2: 13, 3: 29, 5: 81, 7: 149},
    3: {1: 7, 2: 33, 3: 123, 5: 515, 7: 1419},
}
RADII = sorted(LATTICE_COUNTS[2])


def lattice_offsets(r, ndim):
    """Brute force: all integer offsets (z?, y, x) within radius r."""
    rng = range(-r, r + 1)
    return {d for d in itertools.product(rng, repeat=ndim) if sum(c * c for c in d) <= r * r}


@pytest.mark.parametrize('r', RADII)
def test_literal_table_matches_brute_force(r, ndim):
    assert len(lattice_offsets(r, ndim)) == LATTICE_COUNTS[ndim][r]


@pytest.mark.parametrize('r', RADII)
def test_pixels_given_radius(r, ndim):
    assert pixels_given_radius(r, dims=ndim) == LATTICE_COUNTS[ndim][r]


@pytest.mark.parametrize('r', RADII)
def test_expand_target_points_footprint(r, ndim):
    """One particle expands to exactly the lattice points within r, and mask_size agrees."""
    center = {'x_coord': 40, 'y_coord': 50, 'z_coord': 60}
    columns = ['z_coord', 'y_coord', 'x_coord'][-ndim:]
    targets = pd.DataFrame([{'image_name': 'img', **{c: center[c] for c in columns}}])
    expanded, mask_size = expand_target_points(targets, r, dims=ndim)
    offsets = {tuple(int(row[c]) - center[c] for c in columns) for _, row in expanded.iterrows()}
    assert len(expanded) == len(offsets)  # no duplicate pixels
    assert offsets == lattice_offsets(r, ndim)
    assert int(mask_size) == LATTICE_COUNTS[ndim][r]


@pytest.mark.parametrize('r', RADII)
def test_as_mask_footprint(r, ndim):
    """as_mask of one centred particle marks exactly the lattice points within r."""
    size = 2 * r + 5
    c = size // 2
    coords = {'x_coord': np.array([c]), 'y_coord': np.array([c]),
              'z_coord': np.array([c]) if ndim == 3 else None}
    mask = as_mask((size,) * ndim, r, **coords).numpy()
    marked = {tuple(int(i) - c for i in idx) for idx in zip(*np.nonzero(mask))}
    assert marked == lattice_offsets(r, ndim)
    assert set(np.unique(mask)) <= {0.0, 1.0}


@pytest.mark.parametrize('r', RADII)
def test_implementations_agree(r, ndim):
    targets = pd.DataFrame([{'image_name': 'img', 'x_coord': 20, 'y_coord': 20, 'z_coord': 20}])
    if ndim == 2:
        targets = targets.drop(columns='z_coord')
    _, mask_size = expand_target_points(targets, r, dims=ndim)
    assert pixels_given_radius(r, dims=ndim) == int(mask_size)


@pytest.mark.parametrize('r', [1, 3, 7])
@pytest.mark.parametrize('n_particles, total', [(15, 128 * 128), (400, 7 * 4096 * 4096), (10, 64 ** 3)])
def test_calculate_pi(r, n_particles, total, ndim):
    expected = LATTICE_COUNTS[ndim][r] * n_particles / total
    assert calculate_pi(n_particles, r, total, dims=ndim) == pytest.approx(expected, rel=1e-12)
