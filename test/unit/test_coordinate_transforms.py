"""Box <-> centre conversions and coordinate scaling. The conventions themselves (box corner, y inversion,
rounding) are additionally pinned against v0.2.5 output in test/io/test_coordinate_formats.py."""
import numpy as np
import pandas as pd
import pytest

from topaz.utils.conversions import boxes_to_coordinates, coordinates_to_boxes
from topaz.utils.picks import scale_coordinates

CENTRES = np.array([[10, 20], [0, 0], [55, 7], [31, 63]])
SHAPE = (64, 80)  # (height, width)


@pytest.mark.parametrize('box', [8, 9, 32])
def test_box_corner_is_centre_minus_half_box(box):
    boxes = coordinates_to_boxes(CENTRES, box, box)
    np.testing.assert_array_equal(boxes[:, :2], CENTRES - box // 2)
    np.testing.assert_array_equal(boxes[:, 2:], box)


@pytest.mark.parametrize('box', [8, 9, 32])
@pytest.mark.parametrize('invert_y', [False, True])
def test_box_roundtrip(box, invert_y):
    boxes = coordinates_to_boxes(CENTRES, box, box, shape=SHAPE, invert_y=invert_y)
    back = boxes_to_coordinates(boxes, shape=SHAPE, invert_y=invert_y)
    np.testing.assert_array_equal(back, CENTRES)


def test_box_y_inversion_mirrors_about_last_row():
    box = 10
    inverted = coordinates_to_boxes(CENTRES, box, box, shape=SHAPE, invert_y=True)
    np.testing.assert_array_equal(inverted[:, 1] + box // 2, SHAPE[0] - 1 - CENTRES[:, 1])


def test_boxes_to_coordinates_table():
    boxes = coordinates_to_boxes(CENTRES, 8, 8)
    table = boxes_to_coordinates(boxes, image_name='mic')
    assert list(table.columns) == ['image_name', 'x_coord', 'y_coord']
    assert (table.image_name == 'mic').all()
    np.testing.assert_array_equal(table[['x_coord', 'y_coord']].to_numpy(), CENTRES)


@pytest.mark.parametrize('scale', [0.125, 0.5, 2.0, 8.0])
def test_scale_coordinates(tmp_path, scale):
    src = tmp_path / 'in.txt'
    pd.DataFrame({'image_name': ['a'] * 4, 'x_coord': [3, 10, 17, 100], 'y_coord': [5, 11, 250, 0]}).to_csv(src, sep='\t', index=False)
    scale_coordinates(str(src), scale, str(tmp_path / 'out.txt'))
    out = pd.read_csv(tmp_path / 'out.txt', sep='\t')
    np.testing.assert_array_equal(out.x_coord, np.round(np.array([3, 10, 17, 100]) * scale).astype(int))
    np.testing.assert_array_equal(out.y_coord, np.round(np.array([5, 11, 250, 0]) * scale).astype(int))


def test_mirror_y_axis():
    """Regression: mirror_y_axis called DataFrame.clone() (never worked, through 0.3.20)."""
    from topaz.utils.conversions import mirror_y_axis
    table = pd.DataFrame({'image_name': ['a', 'a'], 'x_coord': [10, 3], 'y_coord': [20, 0]})
    mirrored = mirror_y_axis(table, 100)
    np.testing.assert_array_equal(mirrored.y_coord, [79, 99])
    np.testing.assert_array_equal(mirrored.x_coord, table.x_coord)
    np.testing.assert_array_equal(mirror_y_axis(mirrored, 100).y_coord, table.y_coord)
    assert table.y_coord.tolist() == [20, 0]  # input untouched
