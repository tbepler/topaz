"""Negative crops never land on a particle.

CroppableImage.get_random_negative_crop_indices rejection-samples positions until the nearest
labeled positive pixel (a KDTree over the *expanded* particle pixels) is at distance > 0.
"""
import numpy as np
import pandas as pd
import pytest

from _support import array_image
from topaz.training import expand_target_points

SIZE = 64
RADIUS = 3
N_DRAWS = 2000


def particle_grid(ndim):
    """Non-overlapping particles on a regular grid, covering roughly 35-45% of the image."""
    spacing = 8 if ndim == 2 else 7
    centers = np.arange(RADIUS, SIZE - RADIUS, spacing)
    columns = ['z_coord', 'y_coord', 'x_coord'][-ndim:]
    rows = [dict(zip(columns, c)) for c in np.array(np.meshgrid(*[centers] * ndim, indexing='ij')).reshape(ndim, -1).T]
    targets = pd.DataFrame(rows).assign(image_name='img')
    expanded, mask_size = expand_target_points(targets, RADIUS, dims=ndim)
    positives = set(map(tuple, expanded[columns].to_numpy().astype(int)))
    return expanded, int(mask_size), positives


def make_image(split, ndim, seed=0):
    expanded, mask_size, positives = particle_grid(ndim)
    image = array_image(np.zeros((SIZE,) * ndim, dtype=np.float32), expanded, crop_size=9,
                        split=split, dims=ndim, mask_size=mask_size)
    image.rng = np.random.default_rng(seed)
    return image, positives


def draw(sample, ndim, n=N_DRAWS):
    points = []
    for _ in range(n):
        z, y, x = sample()
        points.append((z, y, x) if ndim == 3 else (y, x))
    return points


def test_fixture_coverage(ndim):
    _, positives = make_image('pn', ndim)
    coverage = len(positives) / SIZE ** ndim
    assert 0.3 < coverage < 0.5


def test_negatives_avoid_positives(ndim):
    image, positives = make_image('pn', ndim)
    points = draw(image.get_random_negative_crop_indices, ndim)
    on_particle = sum(p in positives for p in points)
    assert on_particle == 0, f'{on_particle}/{len(points)} negative crop centres landed on a particle'


def test_uniform_sampling_hits_positives(ndim):
    """Power check: unconstrained sampling does land on particles at about the coverage rate,
    so the test above would notice if the rejection step stopped working."""
    image, positives = make_image('pu', ndim)
    points = draw(image.get_random_crop_indices, ndim)
    coverage = len(positives) / SIZE ** ndim
    hit_rate = sum(p in positives for p in points) / len(points)
    sd = np.sqrt(coverage * (1 - coverage) / len(points))
    assert abs(hit_rate - coverage) < 5 * sd


@pytest.mark.parametrize('split', ['pn', 'pu'])
def test_unlabeled_crop_uses_rejection_only_for_pn(split, ndim):
    """get_UN_crop samples negatives for 'pn' and unconstrained positions for 'pu'.

    Particle pixels are set to 1, so a crop's centre value says whether it landed on a particle."""
    image, positives = make_image(split, ndim)
    image.array = np.zeros((SIZE,) * ndim, dtype=np.float32)
    for idx in positives:
        image.array[idx] = 1.0
    centre = (image.size // 2,) * ndim
    crops = [image.get_UN_crop() for _ in range(300)]
    on_particle = sum(float(c[centre]) == 1.0 for c in crops)
    if split == 'pn':
        assert on_particle == 0
    else:  # about 40% coverage, so 300 unconstrained draws land on particles many times
        assert on_particle > 0
