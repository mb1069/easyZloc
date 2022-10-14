from data.datasets import frame_xy_to_polar
import pytest
import numpy as np

@pytest.mark.parametrize("xy_coords, polar_coords", [
    ([[100, 100]], [[0, 0]]),
    ([[50, 50]], [[0.5, -3/8]]),
    ([[150, 50]], [[0.5, -1/8]]),
    ([
        [100, 100],
        [50, 50]
    ], [
        [0, 0],
        [0.5, -3/8]
    ])
])
def test_frame_xy_to_polar_should_return_correct_coords(xy_coords, polar_coords):
    frame_width = 200
    frame_height = 200
    
    np.testing.assert_almost_equal(frame_xy_to_polar(np.array(xy_coords), frame_height, frame_width), np.array(polar_coords))