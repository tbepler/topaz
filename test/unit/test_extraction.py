"""Non-maximum suppression, prediction-target matching and the extraction radius search.

NMS coordinates are returned as (x, y) in 2D and (x, y, z) in 3D.
"""
import numpy as np
import pandas as pd
import pytest

from topaz.algorithms import match_coordinates, non_maximum_suppression, non_maximum_suppression_3d
from topaz.extract import NonMaximumSuppression, extract_auprc, find_opt_radius


def nms(x, r, threshold):
    return non_maximum_suppression(x, r, threshold=threshold) if x.ndim == 2 else \
        non_maximum_suppression_3d(x, r, threshold=threshold)


def reference_nms(x, r, threshold):
    """Brute-force greedy NMS: take the highest remaining value, suppress everything within
    Euclidean distance r of it, repeat. Returns (scores, coords) with coords in topaz's (x, y[, z]) order."""
    idx = np.argwhere(np.ones_like(x, dtype=bool))
    order = np.argsort(-x.ravel(), kind='stable')
    suppressed = np.zeros(x.size, dtype=bool)
    scores, coords = [], []
    for i in order:
        if x.flat[i] <= threshold:
            break
        if suppressed[i]:
            continue
        scores.append(x.flat[i])
        coords.append(idx[i][::-1])
        suppressed |= ((idx - idx[i]) ** 2).sum(axis=1) <= r * r
    return np.array(scores, dtype=np.float32), np.array(coords)


def interior_random_map(ndim, size, r, seed):
    """Distinct random scores with a border of width r below threshold, so no suppression
    window reaches an edge. NMS suppresses by flat-index offsets, so near the left/right edge a window
    wraps onto the neighbouring row (and slice, in 3D); that is a known, accepted quirk, not tested here."""
    rng = np.random.default_rng(seed)
    x = rng.permutation(size ** ndim).reshape((size,) * ndim).astype(np.float32)
    border = np.ones_like(x, dtype=bool)
    border[(slice(r, size - r),) * ndim] = False
    x[border] = -1.0
    return x


@pytest.mark.parametrize('r', [1, 2, 3, 5])
@pytest.mark.parametrize('seed', [0, 1])
def test_nms_matches_bruteforce(r, seed, ndim):
    size = 32 if ndim == 2 else 16
    x = interior_random_map(ndim, size, r, seed)
    scores, coords = nms(x, r, threshold=0.0)
    ref_scores, ref_coords = reference_nms(x, r, threshold=0.0)
    np.testing.assert_array_equal(scores, ref_scores)
    np.testing.assert_array_equal(coords, ref_coords)


def test_nms_known_peaks(ndim):
    """Hand-placed peaks: those farther apart than r survive; the weaker one within r is suppressed."""
    size = 40
    x = np.zeros((size,) * ndim, dtype=np.float32)
    # positions in (x, y[, z]) order
    peaks = {(10, 10, 10): 5.0, (25, 12, 20): 4.0, (13, 10, 10): 3.0, (30, 30, 30): 2.0}
    for pos, value in peaks.items():
        x[tuple(reversed(pos[:ndim]))] = value
    scores, coords = nms(x, 4, threshold=0.5)
    kept = [tuple(c) for c in coords]
    assert kept == [p[:ndim] for p in [(10, 10, 10), (25, 12, 20), (30, 30, 30)]]
    np.testing.assert_array_equal(scores, [5.0, 4.0, 2.0])


def test_nms_threshold(ndim):
    x = np.zeros((20,) * ndim, dtype=np.float32)
    x[(5,) * ndim] = 2.0
    x[(15,) * ndim] = -1.0
    scores, _ = nms(x, 2, threshold=0.0)
    np.testing.assert_array_equal(scores, [2.0])


@pytest.mark.parametrize('offset, matched', [(2.9, True), (3.0, False), (3.1, False)])
def test_match_radius_is_strict(offset, matched, ndim):
    """A prediction matches a target only when strictly closer than the radius (unchanged since v0.2.5)."""
    target = np.zeros((1, ndim))
    pred = target.copy()
    pred[0, 0] = offset
    assignment, dist = match_coordinates(target, pred, 3)
    assert assignment.tolist() == [1.0 if matched else 0.0]
    assert dist[0] == pytest.approx(offset)


def test_match_is_one_to_one(ndim):
    target = np.zeros((1, ndim))
    preds = np.zeros((2, ndim))
    preds[0, 0], preds[1, 0] = 1.0, 2.0
    assignment, _ = match_coordinates(target, preds, 3)
    assert sorted(assignment.tolist()) == [0.0, 1.0]


def test_match_assigns_each_target(ndim):
    targets = np.array([[0] * ndim, [10] + [0] * (ndim - 1)], dtype=float)
    preds = targets[::-1] + 0.5
    assignment, dist = match_coordinates(targets, preds, 3)
    assert assignment.tolist() == [1.0, 1.0]
    np.testing.assert_allclose(dist, np.sqrt(0.25 * ndim))


def two_peak_scores(ndim, spacing):
    """A score map with two equal-ish peaks `spacing` voxels apart along x, and matching targets."""
    size = 48
    x = np.full((size,) * ndim, -10.0, dtype=np.float32)
    centre = [20] * ndim  # (z,) y, x
    x[tuple(centre)] = 5.0
    other = list(centre)
    other[-1] += spacing
    x[tuple(other)] = 4.0
    columns = ['x_coord', 'y_coord', 'z_coord'][:ndim]
    targets = pd.DataFrame([dict(zip(columns, reversed(c))) for c in (centre, other)]).assign(image_name='m')
    return {'m': x}, targets


def test_radius_search_scores_the_extraction_radius(ndim):
    """With peaks 1.5 r apart, NMS at radius r keeps both, so AUPRC at r must be 1."""
    r = 4
    scores, targets = two_peak_scores(ndim, spacing=6)
    auprc, _, n_hits, n_targets = extract_auprc(targets, scores, r, threshold=0.0, dims=ndim)
    assert (n_hits, n_targets) == (2, 2)
    assert auprc == pytest.approx(1.0)


def test_find_opt_radius_prefers_non_suppressing_radius(ndim, capsys):
    """Radii below the peak spacing keep both peaks (AUPRC 1); larger ones lose one."""
    scores, targets = two_peak_scores(ndim, spacing=6)
    radius, auprc = find_opt_radius(targets, scores, threshold=0.0, lo=2, hi=10, step=2, match_radius=3, dims=ndim)
    assert radius < 6
    assert auprc == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason='known bug: the patch-by-patch mode of NonMaximumSuppression fails on any input; '
                                       'no topaz command uses it')
def test_patched_nms_matches_unpatched(ndim):
    size = 128 if ndim == 2 else 64
    x = np.full((size,) * ndim, -10.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    positions = [tuple(p) for p in rng.integers(8, size - 8, size=(6, ndim))]
    for i, pos in enumerate(positions):
        x[pos] = 5.0 - 0.1 * i
    _, ref_scores, ref_coords = NonMaximumSuppression(3, 0.0, dims=ndim, patch_size=0)(('m', x))
    _, scores, coords = NonMaximumSuppression(3, 0.0, dims=ndim, patch_size=32, patch_overlap=8)(('m', x))
    assert sorted(map(tuple, np.asarray(coords).tolist())) == sorted(map(tuple, ref_coords.tolist()))
