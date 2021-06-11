import copy
import math
import pickle
import random
from collections import OrderedDict
from functools import partial
from itertools import product

from pyotf.utils import center_data, prep_data_for_PR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from multiprocessing import Pool
from src.config.optics import model_kwargs, voxel_sizes, target_psf_shape
from src.wavelets.wavelet_data.util import min_max_norm
from src.data.visualise import show_psf_axial

from src.config.datafiles import psf_modelling_file

py_otf = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'cnnSTORM', 'src'))
sys.path.append(py_otf)
from pyotf.otf import HanserPSF, apply_aberration, apply_named_aberrations

n_psfs = 10


def show_all_psfs(psfs):
    _psfs = np.concatenate(psfs, axis=2)
    show_psf_axial(_psfs)


def gen_psf_named_params_raw_psf(params, psf_kwargs=model_kwargs):
    psf = HanserPSF(**psf_kwargs)
    psf = apply_named_aberrations(psf, params)
    psf = psf.PSFi
    psf = psf.astype(np.float)
    psf = psf / psf.max()
    return psf


def gen_psf_named_params(params, psf_kwargs=model_kwargs):
    psf = HanserPSF(**psf_kwargs)
    psf = apply_named_aberrations(psf, params)
    psf = psf.PSFi
    psf = center_data(psf)

    # Normalise PSF and cast to uint8
    psf = psf / psf.max()
    psf *= 255
    psf = psf.astype(np.uint8)

    psf = prep_data_for_PR(psf, multiplier=1.1)
    return psf


def gen_psf_modelled_param(mcoefs, pcoefs, kwargs=model_kwargs):
    psf = HanserPSF(**kwargs)
    psf = apply_aberration(psf, mcoefs, pcoefs)
    return psf.PSFi


class NamedAbberationConfigGenerator:
    def __init__(self, params):
        self.params = params
        self.param_range = params.values()
        self.param_names = params.keys()
        self.config_iter = None

        self.reset()

    def __next__(self):
        config = next(self.config_iter)
        return {c: v for c, v in zip(self.param_names, config)}

    def __iter__(self):
        while True:
            try:
                yield self.__next__()
            except StopIteration:
                break

    def __len__(self):
        total = 1
        for v in self.params.values():
            total *= len(v)
        return total

    def reset(self):
        self.config_iter = product(*self.param_range)


psf_param_config = OrderedDict({
    'oblique astigmatism': np.linspace(0.9, 1.1, 1),
    'defocus': np.linspace(0.9, 1.1, 1),
    # 'vertical astigmatism': np.linspace(0.9, 1.1, 10),
    # 'tilt': np.linspace(0.9, 1.1, 10),
})

cg = NamedAbberationConfigGenerator(psf_param_config)


def load_psf_model():
    ppath = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model.p')
    with open(ppath, 'rb') as f:
        model = pickle.load(f)
    return model.mcoefs, model.pcoefs


class AbberationConfigGenerator:
    N_datasets = 100
    mu = 1
    sigma = 0.01

    def __init__(self, base_mcoefs, base_pcoefs):
        self.base_mcoefs = base_mcoefs
        self.base_pcoefs = base_pcoefs
        self.n_coefs = len(base_mcoefs)
        self.datasets = [np.random.normal(self.mu, self.sigma, self.n_coefs * 2) for _ in range(self.N_datasets)]
        # self.datasets = [np.ones((self.n_coefs*2,)) for _ in range(self.N_datasets)]
        self.config_iter = None

        self.reset()

    def __next__(self):
        config = next(self.config_iter)
        mcoef_mod = config[0:self.n_coefs]
        pcoef_mod = config[self.n_coefs:]
        mcoefs = np.multiply(self.base_mcoefs, mcoef_mod)
        pcoefs = np.multiply(self.base_pcoefs, pcoef_mod)
        return mcoefs, pcoefs

    def __iter__(self):
        while True:
            try:
                yield self.__next__()
            except StopIteration:
                break
        self.reset()

    def __len__(self):
        return self.N_datasets

    def reset(self):
        self.config_iter = iter(self.datasets)


def gen_dataset_named_params(cg):
    with Pool(8) as p:
        psfs = list(tqdm(p.imap_unordered(gen_psf_named_params, cg), total=len(cg)))
    # psfs = [gen_psf_named_params(cfg) for cfg in tqdm(cg)]
    z_pos = np.linspace(0, (target_psf_shape[0] - 1) * voxel_sizes[0], target_psf_shape[0])
    return psfs, z_pos


def apply_normal_noise(coefs):
    noise = np.random.normal(0, 0.01, coefs.shape[0])
    return coefs + noise


def _gen_dataset(mcoefs, pcoefs, normalise, num_ref_psfs, custom_kwargs, noise, *args):
    i = random.randint(0, num_ref_psfs - 1)
    # mcoef = apply_normal_noise(mcoefs[i])
    # pcoef = apply_normal_noise(pcoefs[i])
    mcoef = mcoefs[i]
    pcoef = pcoefs[i]
    psf = gen_psf_modelled_param(mcoef, pcoef, kwargs=custom_kwargs)

    psf = psf.astype(np.float)
    psf = psf / psf.max()

    if noise:
        noise_level = 0.3 * np.max(psf, axis=(1,2))
        psf += noise_level[:, np.newaxis, np.newaxis]
        psf = (psf / psf.max() * 255)

        psf = np.random.poisson(psf)

    # Stack normalisation
    if normalise == 'stack':
        psf = psf / psf.max()
    else:
        # Image-wise normalisation
        # Max value
        # axial_max = psf.max(axis=(1, 2))
        # psf = psf / axial_max[:, None, None]

        # Min-max
        psf = min_max_norm(psf)
    return psf


def get_highres_model_kwargs(zres=40):
    zrange = 1000
    zsize = (2 * zrange) / zres

    high_res_kwargs = copy.deepcopy(model_kwargs)

    high_res_kwargs['zsize'] = math.ceil(zsize)
    high_res_kwargs['zres'] = zres
    return high_res_kwargs


def poisson_psf(psf):
    return np.random.poisson(psf)


def gen_dataset(n_psfs, normalise='img-wise', override_kwargs=None, noise=False):
    df = pd.read_csv(psf_modelling_file)
    print(f'{df.shape[0]} datapoints')
    df = df.loc[df['mse'] <= 0.05]
    del df['mse']
    print(f'{df.shape[0]} after filter by MSE')
    # df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
    # print(f'{df.shape[0]} after filter by outliers')

    print(df.shape)

    cols = list(df)
    pcoef_cols = [c for c in cols if 'pcoef' in c]
    mcoef_cols = [c for c in cols if 'mcoef' in c]

    pcoefs = df[pcoef_cols].to_numpy()
    mcoefs = df[mcoef_cols].to_numpy()

    # pcoefs = pcoefs.mean(axis=0)
    # mcoefs = mcoefs.mean(axis=0)
    num_ref_psfs = len(pcoefs)
    custom_kwargs = get_highres_model_kwargs(10)

    if override_kwargs:
        custom_kwargs.update(override_kwargs)

    pfunc = partial(_gen_dataset, mcoefs, pcoefs, normalise, num_ref_psfs, custom_kwargs, noise)
    with Pool(8) as p:
        psfs = list(tqdm(p.imap_unordered(pfunc, range(n_psfs)), total=n_psfs))

    psfs = np.stack(psfs, axis=0)

    min_zpos = -custom_kwargs['zsize'] / 2 * custom_kwargs['zres']
    max_zpos = (custom_kwargs['zsize'] / 2 - 1) * custom_kwargs['zres']
    z_pos = np.linspace(min_zpos, max_zpos , custom_kwargs['zsize'])
    return psfs, z_pos


def visualise_modelling_results():
    df = pd.read_csv(psf_modelling_file)
    print(df)
    mcoefs = df[[c for c in list(df) if 'mcoef' in c]].to_numpy()
    pcoefs = df[[c for c in list(df) if 'pcoef' in c]].to_numpy()

    for mcoef, pcoef in zip(mcoefs, pcoefs):
        psf = gen_psf_modelled_param(mcoef, pcoef)
        show_psf_axial(psf)
        plt.show()


if __name__ == '__main__':
    psfs, zpos = gen_dataset(1, noise=True)
    plt.imshow(psfs[0][30])
    plt.show()

    print(psfs.shape)
    print(zpos.shape)
    print(zpos.min(), zpos.max())
    show_psf_axial(psfs[0])
    quit()
    psf2, _ = gen_dataset(1, noise=True)

    psfs = [psfs[0], psf2[0]]
    show_all_psfs(psfs)

    # for psf, z in zip(psfs, z_pos):
    #     print(psf.shape)
    #     fname = f'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/tmp/{z}.png'
    #     imwrite(fname, psf)
    # psfs, z_pos = gen_dataset()
    # show_all_psfs(psfs)

    # cfg = {'oblique astigmatism': 1}
    # psf = gen_psf_named_params(cfg)
    # show_psf_axial(psf)
