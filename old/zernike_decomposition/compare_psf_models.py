import glob
import os

from skimage import img_as_ubyte, io
from tifffile import imread
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from src.config.optics import model_kwargs
from src.zernike_decomposition.ga import ga_fit_psf, prepare_target_psf, sim_psf
from src.zernike_decomposition.model_psf import fit_psf_lsquares
import numpy as np

csv_file = os.path.join(os.path.dirname(__file__), 'results.csv')


def plot_results(target_psf, ga_psf, lsquares_psf):
    print(target_psf.shape)
    print(ga_psf.shape)
    print(lsquares_psf.shape)
    _psfs = np.concatenate((target_psf, ga_psf, lsquares_psf), axis=2)
    sub_psf = np.concatenate(_psfs[slice(0, target_psf.shape[0], 3)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    sub_psf = img_as_ubyte(sub_psf)
    print(sub_psf.max())
    io.imsave('out.png', sub_psf)
    io.imshow(sub_psf)
    plt.show()
    plt.show()


def main():

    psfs = list(glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/data/jonny_psf_emitters/*.tif'))

    psfs = random.sample(psfs, k=50)

    psf_dict = {os.path.basename(p): prepare_target_psf(imread(p)) for p in psfs}
    records = []

    N_ZERNS = 16
    for imname, psf in tqdm(psf_dict.items()):
        model_kwargs['zsize'] = psf.shape[0]
        model_kwargs['size'] = psf.shape[1]
        lsquares_res = fit_psf_lsquares(N_ZERNS, psf)
        ga_res = ga_fit_psf(psf, N_ZERNS, plot_best=False)
        records.append({
            'name': imname,
            'l_squares': lsquares_res['mse'],
            'ga': ga_res['mse']
        })

        print(psf.shape)
        ga_psf = sim_psf(model_kwargs, ga_res['mcoefs'], ga_res['pcoefs'])
        lsquares_psf = sim_psf(model_kwargs, lsquares_res['mcoefs'], lsquares_res['pcoefs'])
        plot_results(psf, ga_psf, lsquares_psf)
        print(records[-1])

    df = pd.DataFrame.from_records(records)

    df.to_csv(csv_file)
    plt.axis('on')
    df.boxplot()
    plt.show()
    print(df)


if __name__ == '__main__':
    main()
