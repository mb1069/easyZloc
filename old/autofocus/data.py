from functools import partial
import glob
import numpy as np
import os
from numpy.random import shuffle
from tifffile import imread
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pywt

from src.autofocus.estimate_offset import estimate_offset, resize_img
from src.autofocus.config import cfgs, get_cfg_images, dwt_level

def dwt_transform(img, wavelet='sym4', level=3):
    try:
        img = img / img.max()
    except RuntimeWarning:
        img = 0
    coeffs2 = pywt.wavedecn(img, wavelet, level=level)
    coeffs = pywt.coeffs_to_array(coeffs2)[0].flatten()
    return coeffs

def dwt_dataset(psfs, wavelet='sym4', level=3):
    print(f'Running Wavelet transform for dataset at level: {level}')
    func = partial(dwt_transform, wavelet=wavelet, level=level)
    # with Pool(8) as p:
    x = list(map(func, psfs.squeeze()))
    print(np.argwhere(np.isnan(x)).shape)

    x = np.stack(x)

    return x


def remove_low_variance_features(xs):
    # Remove features with low variance to reduce memory footprint
    xs_vars = np.var(xs, axis=0)
    cols = np.where(xs_vars > 0.1)[0]
    print(f'Removing {xs.shape[1] - len(cols)} features with low variance.')
    xs = xs[:, cols]

    print(f'X: {round(xs.nbytes / (10 ** 9), 3)} GB')
    return xs, cols    

def split_dataset(xs, ys):
    x_train, x_other, y_train, y_other = train_test_split(xs, ys, train_size=0.8)
    x_val, x_test, y_val, y_test = train_test_split(x_other, y_other, train_size=0.5)

    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }


def transform_img(impath, dwt_level):
    outpath = get_dwt_transform_path(impath, dwt_level)
    if not os.path.exists(outpath):
        img = imread(impath)
        print(impath)
        quit()
        img = resize_img(img)

        dwt = dwt_dataset(img, level=dwt_level, wavelet='sym4')
        dwt = dwt.astype(np.float16)
        np.savez(outpath, dwt)
        del img
        del dwt


def transform_data(imgs, dwt_level):
    for img in tqdm(imgs):
        transform_img(img, dwt_level)
    # process_map(transform_img, imgs, max_workers=4)


def get_dwt_transform_path(impath, dwt_level):
    return impath.replace('.tif', f'_{dwt_level}.npz')


def get_axial_position_file(impath):
    return impath.replace('.tif', '.offset.npz')


def prepare_axial_positions(imgs, voxel_sizes, row_avg):
    for impath in tqdm(imgs):
        outpath = get_axial_position_file(impath)

        if not os.path.exists(outpath):
            print(impath)
            img = imread(impath)

            y = estimate_offset(img, voxel_sizes=voxel_sizes, row_avg=row_avg)
            np.savez(outpath, y)
            del img


def gather_data(imgs, slc, dwt_level):
    xs = []
    ys = []

    for impath in tqdm(imgs[slc]):
        dwt_path = get_dwt_transform_path(impath, dwt_level)
        xs.append(np.load(dwt_path)['arr_0'])

        axial_position_path = get_axial_position_file(impath)
        ys.append(np.load(axial_position_path)['arr_0'])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    
    return xs, ys


if __name__ == '__main__':
    for cfg in cfgs.values():
        print(f'Processing {cfg}')
        vs = cfg['z_voxel']
        images = get_cfg_images(cfg)
        
        print(f'Found {len(images)} images.')

        all_vs = (vs, None, None)
        transform_data(images, dwt_level)
        prepare_axial_positions(images, all_vs, cfg['row_avg'])
