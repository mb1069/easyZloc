from pyotf.otf import HanserPSF, apply_aberration
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tifffile import imread
import pytest 

from data.align_psfs import tf_find_optimal_roll, norm_zero_one
from test.extract_psfs import TEST_DATA_DIR



def load_experimental_bead_stacks():
    fpaths = glob.glob(f'{TEST_DATA_DIR}/*.tif')
    return {
        os.path.basename(f): imread(f).astype(float) for f in fpaths
    }

exp_stacks = load_experimental_bead_stacks()
if len(exp_stacks.keys()) == 0:
    raise Exception('Test files missing')

kwargs = dict(
    wl=647,
    na=1.3,
    ni=1.51,
    res=106,
    zres=10,
    size=32,
    zsize=200,
    vec_corr="none",
    condition="none",
)
psf = HanserPSF(**kwargs)
psf = apply_aberration(psf, np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 1]))

blank_psf = psf.PSFi


@pytest.mark.parametrize("blank_psf", [
    blank_psf,
    *[v[0] for k, v in exp_stacks.items()],
    *[v[3] for k, v in exp_stacks.items()],
    *[v[5] for k, v in exp_stacks.items()],
    *[v[7] for k, v in exp_stacks.items()],
])
def test_tf_find_optimal_roll_retrieves_correct_roll(blank_psf):
    blank_psf = norm_zero_one(blank_psf)
    for offset in [-10, -5, 5, 10, 15, 19]:
        rolled_psf = np.roll(blank_psf, offset, axis=0)
        for _ in range(10):
            noised_rolled_psf = rolled_psf + np.random.normal(0, 5e-2, size=rolled_psf.shape)
            assert tf_find_optimal_roll(blank_psf, noised_rolled_psf, 1) == offset