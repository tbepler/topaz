"""Training setup: the data loader wiring and the pi bookkeeping per method."""
import numpy as np
import pandas as pd
import pytest

from topaz import methods, training
from topaz.model.classifier import LinearClassifier
from topaz.model.factory import get_feature_extractor
from topaz.utils.image import save_image


def write_training_set(tmp_path, n_images=2, size=64):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_images):
        save_image(rng.standard_normal((size, size)).astype(np.float32), str(image_dir / f'img{i}.mrc'))
        rows += [{'image_name': f'img{i}', 'x_coord': x, 'y_coord': y} for x, y in [(16, 16), (40, 20), (30, 45)]]
    targets = tmp_path / 'targets.txt'
    pd.DataFrame(rows).to_csv(targets, sep='\t', index=False)
    return image_dir, targets


def test_train_loader_passes_worker_init_fn(tmp_path):
    """Without worker_init_fn every DataLoader worker forks the same numpy RNG state and
    produces identical batches (the v0.3.x regression fixed in 67cecc2)."""
    image_dir, targets = write_training_set(tmp_path)
    train_loader, test_loader = training.make_data_iterators(
        str(image_dir), str(targets), crop=31, split='pn', minibatch_size=4, epoch_size=2,
        num_workers=2, balance=0.5, dims=2, radius=3, preload=True)
    assert test_loader is None
    assert train_loader.num_workers == 2
    assert train_loader.worker_init_fn is training.worker_init_fn


def small_classifier():
    return LinearClassifier(get_feature_extractor('conv31', units=2, bn=False))


@pytest.mark.parametrize('method, pi, p_observed, expected_type, expected_pi, expected_split', [
    # GE methods model only the unlabeled data: pi excludes the labeled positives
    ('GE-binomial', 0.10, 0.02, methods.GE_binomial, 0.08, 'pn'),
    ('GE-KL', 0.10, 0.02, methods.GE_KL, 0.08, 'pn'),
    # pi <= observed positive fraction: GE falls back to PN with pi = p_observed
    ('GE-binomial', 0.01, 0.02, methods.PN, 0.02, 'pn'),
    ('GE-KL', 0.02, 0.02, methods.PN, 0.02, 'pn'),
    ('PN', 0.10, 0.02, methods.PN, 0.10, 'pn'),
    ('PU', 0.10, 0.02, methods.PU, 0.10, 'pu'),
])
def test_training_step_method(method, pi, p_observed, expected_type, expected_pi, expected_split):
    trainer, criteria, split = training.make_training_step_method(
        small_classifier(), num_positive_regions=1000, positive_fraction=p_observed, method=method, pi=pi)
    assert type(trainer) is expected_type
    assert trainer.pi == pytest.approx(expected_pi, rel=1e-12)
    assert split == expected_split


def test_ge_fallback_warns(capsys):
    training.make_training_step_method(small_classifier(), 1000, 0.02, method='GE-binomial', pi=0.01)
    assert 'setting method to PN' in capsys.readouterr().err
