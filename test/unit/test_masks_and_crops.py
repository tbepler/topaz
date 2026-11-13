"""Particle masks, crop extraction and the training dataset's sampling."""
import numpy as np
import pandas as pd
import pytest
import torch

from _support import array_image, write_mrc
from topaz.training import expand_target_points
from topaz.utils.data.memory_mapped_data import MultipleImageSetDataset
from topaz.utils.picks import as_mask


def brute_force_mask(shape, radius, centers):
    """Every voxel within `radius` of any center (centers in array index order)."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1)
    mask = np.zeros(shape, dtype=bool)
    for c in centers:
        mask |= ((grid - np.array(c)) ** 2).sum(axis=-1) <= radius * radius
    return mask.astype(np.float32)


@pytest.mark.parametrize('radius', [1, 3, 4])
def test_as_mask_overlapping_and_edge_particles(radius, ndim):
    size = 24 if ndim == 2 else 16
    shape = (size,) * ndim
    # overlapping pair, one on the low edge, one in the far corner (index order: (z,) y, x)
    centers = [(8,) * ndim, (8,) * (ndim - 1) + (10,), (0,) * ndim, (size - 1,) * ndim, (3,) * (ndim - 1) + (size - 2,)]
    coords = np.array(centers)
    x, y = coords[:, -1].copy(), coords[:, -2].copy()
    z = coords[:, 0].copy() if ndim == 3 else None
    before = [a.copy() for a in (x, y) + ((z,) if ndim == 3 else ())]

    mask = as_mask(shape, radius, x, y, z).numpy()
    np.testing.assert_array_equal(mask, brute_force_mask(shape, radius, centers))
    for a, b in zip((x, y) + ((z,) if ndim == 3 else ()), before):
        np.testing.assert_array_equal(a, b)  # inputs untouched


@pytest.mark.parametrize('center', ['interior', 'low_edge', 'high_edge', 'corner'])
def test_get_crop_matches_padded_slice(center, ndim):
    size, crop = 20, 7
    array = np.arange(size ** ndim, dtype=np.float32).reshape((size,) * ndim) + 1  # nonzero, so padding is visible
    image = array_image(array, pd.DataFrame(columns=['image_name', 'x_coord', 'y_coord', 'z_coord'][:3 + (ndim == 3)]),
                        crop_size=crop, split='pu', dims=ndim)
    idx = {'interior': (10,) * ndim, 'low_edge': (0,) + (10,) * (ndim - 1), 'high_edge': (10,) * (ndim - 1) + (size - 1,),
           'corner': (size - 1,) * ndim}[center]
    z, y, x = (idx if ndim == 3 else (None,) + idx)
    half = crop // 2
    padded = np.pad(array, half)
    expected = padded[tuple(slice(i, i + crop) for i in idx)]
    got = image.get_crop((z, y, x))
    assert isinstance(got, torch.Tensor)
    np.testing.assert_array_equal(got.numpy(), expected)


def particle_dataset(tmp_path, ndim, positive_balance, number_samples=2000, radius=2):
    """Two images whose particle pixels are 1 and background 0, and a dataset over them (no augmentation)."""
    size = 40 if ndim == 2 else 24
    columns = ['x_coord', 'y_coord', 'z_coord'][:ndim]
    centers = [(10, 12, 8), (25, 20, 15), (15, 30, 18)] if ndim == 3 else [(10, 12), (25, 20), (15, 30)]
    centers = [c for c in centers if max(c) < size - radius]
    paths, rows = [], []
    for name in ('a', 'b'):
        targets = pd.DataFrame([dict(zip(columns, c)) for c in centers]).assign(image_name=name)
        expanded, mask_size = expand_target_points(targets, radius, dims=ndim)
        array = np.zeros((size,) * ndim, dtype=np.float32)
        for _, row in expanded.iterrows():
            array[tuple(int(row[c]) for c in reversed(columns))] = 1.0
        write_mrc(tmp_path / f'{name}.mrc', array)
        paths.append(str(tmp_path / f'{name}.mrc'))
        rows.append(expanded)
    expanded = pd.concat(rows, ignore_index=True)
    dataset = MultipleImageSetDataset([paths], expanded, number_samples, crop_size=5, positive_balance=positive_balance,
                                      split='pn', rotate=False, flip=False, dims=ndim, radius=radius,
                                      mask_size=int(mask_size), preload=True)
    dataset.rng = np.random.default_rng(0)
    for group in dataset.images:
        for image in group:
            image.rng = np.random.default_rng(1)
    return dataset


@pytest.mark.parametrize('positive_balance', [0.1, 0.5])
def test_dataset_positive_fraction_and_centres(tmp_path, ndim, positive_balance):
    dataset = particle_dataset(tmp_path, ndim, positive_balance)
    n = len(dataset)
    labels, centres = [], []
    for i in range(n):
        crop, label = dataset[i]
        assert tuple(crop.shape) == (5,) * ndim
        labels.append(label)
        centres.append(float(crop[(2,) * ndim]))
    labels = np.array(labels)
    sd = np.sqrt(positive_balance * (1 - positive_balance) / n)
    assert abs(labels.mean() - positive_balance) < 5 * sd
    centres = np.array(centres)
    assert (centres[labels == 1] == 1.0).all(), 'positive crops must be centred on a particle pixel'
    assert (centres[labels == 0] == 0.0).all(), "'pn' negative crops must be centred off particles"
