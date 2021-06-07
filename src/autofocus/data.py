import glob
from multiprocessing import Pool

import cv2
import numpy as np
import os
from tifffile import imread
from tqdm import tqdm, trange

from tqdm.contrib.concurrent import process_map

from src.autofocus.estimate_offset import estimate_offset
from src.wavelets.wavelet_data.util import dwt_dataset

# img_glob = '/home/miguel/Projects/uni/data/autofocus/cylindrical_lenses_openframe/*/*/*.tif'

img_type = '50'
img_glob = f'/home/miguel/Projects/uni/data/autofocus/202006*_{img_type}nm*7ms/*/*.tif'

imgs = glob.glob(img_glob)

# shuffle(imgs)
#
# imgs = imgs[:100]

voxel_sizes = (1000, None, None)
cutoff = int(len(imgs) * 0.8)


def resize_img(img):
    height = 256
    width = 256
    # width = int((height / img.shape[1]) * img.shape[2])

    new_img = np.zeros((img.shape[0], height, width))
    for i in range(len(img)):
        img_z = img[i]
        new_img[i, :, :] = cv2.resize(img_z, (width, height), interpolation=cv2.INTER_CUBIC)
    return new_img


def transform_img(impath):
    for dwt_level in list(range(7)):
        outpath = get_dwt_transform_path(impath, dwt_level)
        if not os.path.exists(outpath):
            print(impath)
            img = imread(impath)
            img = resize_img(img)
            dwt = dwt_dataset(img, level=dwt_level, wavelet='sym4')
            dwt = dwt.astype(np.float16)

            np.savez(outpath, dwt)
            del img
            del dwt


def transform_data(imgs):
    for img in tqdm(imgs):
        transform_img(img)
    # process_map(transform_img, imgs, max_workers=4)


def get_dwt_transform_path(impath, dwt_level):
    return impath.replace('.tif', f'_{dwt_level}.npz')


def read_img(impath):
    # from src.autofocus.readTrainingStacks import createReader, getImageRead, getImageInfo, closeReader
    reader = createReader(impath)
    pT, pX, pY = getImageInfo(impath)

    arrX_save = np.zeros((int(pT), int(1536), int(2048)))

    for i in range(0, pT):
        new_img = getImageRead(reader, i)
        new_img = new_img / np.linalg.norm(new_img)
        arrX_save[i, :, :] = new_img
        # fft1 = np.fft.fft2(rez_img)
        # fft1 = fft1/np.linalg.norm(fft1)
        # fft_save[i, :, :] = fft1

    # e2 = cv2.getTickCount()
    # e = (e2-e1)/cv2.getTickFrequency()
    closeReader(reader)
    return arrX_save


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


def gather_data(slc, dwt_level):
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
    # Slit
    cfgs = [{
        'z_voxel': 50,
        'row_avg': False # columns
    },
        {'z_voxel': 1000,
         'row_avg': True
         }
    ]

    # Cylindrical lens
    # cfgs = [{
    #     'z_voxel': 200,
    #     'row_avg': False
    # }, {
    #     'z_voxel': 50,
    #     'row_avg': True
    # }]

    for cfg in cfgs:
        vs = cfg['z_voxel']
        # img_glob = f'/home/miguel/Projects/uni/data/autofocus/202006*_{vs}nm*/*/*.tif'
        img_glob = f'/home/miguel/Projects/uni/data/autofocus/202006*_50nm*7ms/*/*.tif'

        # img_glob = f'/home/miguel/Projects/uni/data/autofocus/cylindrical_lenses_openframe/20210525_*um_{vs}nm_cylindrical_lenses/*.tif'
        images = glob.glob(img_glob)

        all_vs = (vs, None, None)
        transform_data(images)
        prepare_axial_positions(images, all_vs, cfg['row_avg'])
