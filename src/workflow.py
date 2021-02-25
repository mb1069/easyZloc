from tifffile import imwrite, imread
import glob
import pandas as pd
import matplotlib.pyplot as plt

from src.data.data_manager import jonny_data_dir
from src.data.data_source import JonnyDataSource
from src.zernike_decomposition.model_psf import fit_multiple_psfs
from src.config.datafiles import psf_modelling_file


def extract_emitters():
    ds = JonnyDataSource(jonny_data_dir)
    psfs, _ = ds.get_all_emitter_stacks(bound=20, pixel_size=85)
    for i, psf in enumerate(psfs):
        # imshow(psf[int(psf.shape[0]/2)])
        # plt.show()
        imwrite(f'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/{i}.tif', psf,
                compress=6)


def model_psfs():
    psfs = []
    for psf_path in list(
            glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif')):
        psfs.append(imread(psf_path))
    fit_multiple_psfs(psfs, 120)


def check_modelling_results():
    df = pd.read_csv(psf_modelling_file)
    # df = df.loc[df['mse'] <= df['mse'].quantile(0.25)]
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
    check_modelling_results()

if __name__ == '__main__':
    main()
