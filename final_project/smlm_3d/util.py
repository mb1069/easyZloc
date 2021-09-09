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
    bpath = '/Volumes/Samsung_T5/uni/smlm/experimental_data/'
    if not os.path.exists(bpath):
        bpath = '/data/mdb119/smlm/data/experimental_data/'
        if not os.path.exists(bpath):
            bpath = '/home/miguel/Projects/uni/data/smlm_3d/'
    base_data_dir = Path(bpath)
    return base_data_dir


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


def dwt_transform(img, wavelet='sym4', level=8):
    img = img / img.max()
    coeffs2 = pywt.wavedecn(img, wavelet, level=level)
    coeffs = pywt.coeffs_to_array(coeffs2)[0].flatten()
    print(coeffs.shape)
    print(pywt.ravel_coeffs(coeffs2).shape)
    quit()
    return coeffs


def dwt_dataset(psfs, wavelet='sym4', level=8):
    print(f'Running Wavelet transform for dataset at level: {level}')
    func = partial(dwt_transform, wavelet=wavelet, level=level)
    x = list(map(func, psfs.squeeze()))
    x = np.stack(x)
    return x

def dwt_inverse_transform(dwt, wavelet='sym4', level=8):
    dwt = pywt.array_to_coeffs(dwt)
    img = pywt.waverecn(dwt, wavelet, level)
    print(img.shape)
    return img

def dwt_inverse_dataset(dwt, wavelet='sym4', level=8):
    print(f'Running Wavelet transform for dataset at level: {level}')
    func = partial(dwt_inverse_transform, wavelet=wavelet, level=level)
    x = list(map(func, dwt.squeeze()))
    x = np.stack(x)
    return x