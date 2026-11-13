"""Precision-recall and average precision, against sklearn and hand-computed cases."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from topaz.metrics import average_precision, precision_recall_curve


def random_scores(seed, n=500, positive_rate=0.3):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < positive_rate
    s = rng.standard_normal(n) + 1.5 * y
    return y, s


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_average_precision_matches_sklearn(dtype, seed):
    """Regression: before 0.3.21, float64 inputs were mis-sorted (the 'f4,f4' row view), e.g. 0.14 instead of 0.71."""
    y, s = random_scores(seed)
    s = s.astype(np.float32)  # compare on the precision topaz computes in
    expected = average_precision_score(y, s)
    assert average_precision(y.astype(dtype), s.astype(dtype)) == pytest.approx(expected, rel=1e-6)
    assert precision_recall_curve(y.astype(dtype), s.astype(dtype))[3] == pytest.approx(expected, rel=1e-6)


def test_hand_computed_curve():
    # ranked: hit, miss, hit, miss, miss -> precision at the hits 1/1 and 2/3
    target = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    precision, recall, threshold, auprc = precision_recall_curve(target, pred)
    np.testing.assert_allclose(precision, [1, 1 / 2, 2 / 3, 2 / 4, 2 / 5])
    np.testing.assert_allclose(recall, [1 / 2, 1 / 2, 1, 1, 1])
    np.testing.assert_allclose(threshold, pred)
    assert auprc == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(target, pred) == pytest.approx(auprc)


def test_unmatched_targets_lower_recall():
    """N counts all targets, including ones no prediction matched."""
    target = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    _, recall, _, auprc = precision_recall_curve(target, pred, N=4)
    assert recall[-1] == pytest.approx(0.5)
    assert auprc == pytest.approx((1 + 2 / 3) / 4)
    assert average_precision(target, pred, N=4) == pytest.approx(auprc)


def test_tied_scores_form_one_bucket():
    target = np.array([1, 0, 1, 1], dtype=np.float32)
    pred = np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32)
    precision, recall, threshold, auprc = precision_recall_curve(target, pred)
    np.testing.assert_allclose(threshold, [0.5, 0.2])
    np.testing.assert_allclose(precision, [1 / 2, 3 / 4])
    assert auprc == pytest.approx((1 / 2) * (1 / 3) + (3 / 4) * (2 / 3))
    assert average_precision_score(target, pred) == pytest.approx(auprc)
