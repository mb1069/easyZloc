from data.datasets import norm_coords
import pytest
import numpy as np

@pytest.mark.parametrize("xy_coords, frame_dim, norm", [
    ([[0, 0]], [200, 200], [[0, 0]]),
    ([[100, 100]], [200, 200], [[0.5, 0.5]]),
    ([[50, 50]], [200, 200], [[0.25, 0.25]]),
    ([[150, 50]], [200, 200], [[0.75, 0.25]]),
    ([[150, 50]], [200, 100], [[0.75, 0.5]]),
    ([[50, 150]], [100, 200], [[0.5, 0.75]]),
])
def test_norm_coords_should_norm_coords(xy_coords, frame_dim, norm):
    frame_height, frame_width = frame_dim
    np.testing.assert_almost_equal(norm_coords(np.array(xy_coords), frame_height, frame_width), np.array(norm))

@pytest.mark.parametrize("xy_coords, frame_dim, norm", [
    ([[-1, -1]], [200, 200], [[0, 0]]),
    ([[300, 300]], [200, 200], [[0.5, 0.5]]),
])
def test_norm_coords_should_raise_error_if_invalid_coords(xy_coords, frame_dim, norm):
    frame_height, frame_width = frame_dim
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(norm_coords(np.array(xy_coords), frame_height, frame_width), np.array(norm))

