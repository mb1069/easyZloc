import matplotlib
from matplotlib import cm
from tifffile import imwrite, imread
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from src.config.optics import bounds
from src.data.data_source import JonnyDataSource
from src.zernike_decomposition.gen_psf import visualise_modelling_results
from src.zernike_decomposition.model_psf import fit_multiple_psfs
from src.config.datafiles import psf_modelling_file, data_dir

emitter_dir = '/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large'
os.makedirs(emitter_dir, exist_ok=True)


def extract_emitters():
    ds = JonnyDataSource(data_dir)
    for i, psf in enumerate(ds.get_all_emitter_stacks(bound=bounds, pixel_size=85)):
        # imshow(psf[int(psf.shape[0]/2)])
        # plt.show()
        imwrite(os.path.join(emitter_dir, f'{i}.tif'), psf, compress=6)


min_contrast_threshold = 10000


def model_psfs():
    psfs = []
    for psf_path in list(glob.glob(os.path.join(emitter_dir, '*.tif'))):
        psf = imread(psf_path)
        if psf.max() - psf.min() > min_contrast_threshold:
            psfs.append(psf)
    fit_multiple_psfs(psfs, 32)


def check_modelling_results():
    cmap = cm.get_cmap('Spectral')

    coords = pd.read_csv('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/coords.csv')
    coords = coords[['x', 'y']]

    df = pd.read_csv(psf_modelling_file)
    df = df.loc[df['mse'] <= 0.005]

    df = pd.concat((coords, df), axis=1)

    df.plot.scatter('x', 'y', c='mse', cmap=cmap)
    plt.show()

    center = (700, 700)
    df['from_center'] = np.sqrt(np.power(center[0] - df['x'], 2) + np.power(center[1] - df['y'], 2))
    for c in [f'pcoef_{i}' for i in range(0, 9)]:
        # df2 = df.loc[(df[c].mean() * 0.75 <= df[c]) & (df[c] <= df[c].mean() * 1.25)]
        print(df.shape)
        print(df[c].min(), df[c].max())
        df.plot.scatter('x', 'y', c=c, cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.title(c)
        plt.show()
        input()
    plt.axis('on')
    df.boxplot(column=[f'pcoef_{i}' for i in range(32)])
    plt.xticks(rotation=45)
    plt.show()
    df.boxplot(column=[f'mcoef_{i}' for i in range(32)])
    plt.xticks(rotation=45)
    plt.show()
    df.boxplot(column='mse')
    plt.xticks(rotation=45)
    plt.show()


def main():
    # extract_emitters()
    # model_psfs()
    # visualise_modelling_results()
    check_modelling_results()


if __name__ == '__main__':
    main()
