import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import pywt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from tifffile import imshow
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

SEED = 42

DEFAULT_WAVELET = 'sym4'

DEFAULT_WAVELET_LEVEL = 4


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def limit_data_range(psfs, peaks, z_range=1000, min_z=None, max_z=None):
    if min_z and max_z:
        lower_lim = min_z
        upper_lim = max_z
    else:
        lower_lim = -z_range
        upper_lim = z_range
    psfs, peaks = zip(*[(psf, p) for psf, p in zip(psfs, peaks) if lower_lim < float(p) < upper_lim])
    return [np.array(psfs), np.array(peaks)]


def min_max_norm(psf):
    min_z = psf.min(axis=(1, 2))[:, None, None]
    max_z = psf.max(axis=(1, 2))[:, None, None]
    psf = (psf - min_z) / (max_z - min_z)
    return psf


def get_base_data_path():
    bpaths = [
        '/Volumes/Samsung_T5/uni/smlm/experimental_data/',
        '/data/mdb119/smlm/data/experimental_data/',
        '/home/miguel/Projects/uni/data/smlm_3d/',
        '/Users/miguelboland/Projects/uni/phd/smlm_z/final_project/smlm_3d/mini_data_dir'
    ]
    try:
        while not os.path.exists(bpaths[-1]):
            bpaths.pop()
    except IndexError:
        print('BPath not found')
        return ''
    return Path(bpaths[-1])


def split_for_training(X, y):
    train_size = 0.7
    val_size = 0.2
    test_size = 0.1
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=SEED)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=(val_size / (train_size + val_size)),
                                                      shuffle=True, random_state=SEED)

    train_ds = [train_x, train_y]
    val_ds = [val_x, val_y]
    test_ds = [test_x, test_y]
    return {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }


def dwt_transform(img, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
    img = img / img.max()
    coeffs2 = pywt.wavedecn(img, wavelet, level=level)
    coeffs = pywt.coeffs_to_array(coeffs2)[0].flatten()
    return coeffs


def dwt_dataset(psfs, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
    print(f'Running Wavelet transform for dataset at level: {level}')
    func = partial(dwt_transform, wavelet=wavelet, level=level)
    x = list(map(func, psfs.squeeze()))
    x = np.stack(x)
    return x


def dwt_inverse_transform(dwt, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
    dwt = pywt.array_to_coeffs(dwt)
    img = pywt.waverecn(dwt, wavelet, level)
    return img


def dwt_inverse_dataset(dwt, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
    print(f'Running Wavelet transform for dataset at level: {level}')
    func = partial(dwt_inverse_transform, wavelet=wavelet, level=level)
    x = list(map(func, dwt.squeeze()))
    x = np.stack(x)
    return x
