import pickle
import random
from collections import OrderedDict
from itertools import product

from imageio import imwrite
from tqdm import trange
from pyotf.utils import center_data, prep_data_for_PR
from scipy.stats import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from multiprocessing import Pool
from src.config.optics import model_kwargs, voxel_sizes, target_psf_shape
from src.data.visualise import show_psf_axial
from natsort import natsorted

from src.config.datafiles import psf_modelling_file

py_otf = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'cnnSTORM', 'src'))
sys.path.append(py_otf)
from pyotf.otf import apply_named_aberration, HanserPSF, apply_aberration, apply_named_aberrations

n_psfs = 10

plt.axis('off')


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


def gen_psf_modelled_param(mcoefs, pcoefs):
    psf = HanserPSF(**model_kwargs)
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
    z_pos = np.linspace(0, target_psf_shape[0] * voxel_sizes[0], target_psf_shape[0])
    return psfs, z_pos


def apply_normal_noise(coefs):
    noise = np.random.normal(0, 0.01, coefs.shape[0])
    return coefs + noise


def gen_dataset(n_psfs):
    df = pd.read_csv(psf_modelling_file)
    df = df.loc[df['mse'] <= df['mse'].quantile(0.25)]
    cols = list(df)
    pcoef_cols = [c for c in cols if 'pcoef' in c]
    mcoef_cols = [c for c in cols if 'mcoef' in c]

    pcoefs = df[pcoef_cols].to_numpy()
    mcoefs = df[mcoef_cols].to_numpy()

    # pcoefs = pcoefs.mean(axis=0)
    # mcoefs = mcoefs.mean(axis=0)
    num_ref_psfs = len(pcoefs)

    psfs = []

    for _ in trange(n_psfs):
        i = random.randint(0, num_ref_psfs-1)
        mcoef = apply_normal_noise(mcoefs[i])
        pcoef = apply_normal_noise(pcoefs[i])
        psf = gen_psf_modelled_param(mcoef, pcoef)
        psf = psf.astype(np.float)
        psf = psf/psf.max()
        psfs.append(psf)

    psfs = np.stack(psfs, axis=0)
    z_pos = np.linspace(0, target_psf_shape[0] * voxel_sizes[0], target_psf_shape[0])

    # show_all_psfs(psfs)

    # Save
    return psfs, z_pos

# def gen_dataset(*args, **kwargs):
#     # DEBUG to generate pyOTF datasets
#     configs = cg
#     psfs = [gen_psf_named_params_raw_psf(cfg) for cfg in tqdm(configs)]
#
#
#     # psfs = [gen_psf_named_params_raw_psf({'oblique astigmatism': 1}, model_kwargs)]
#
#     psfs = np.stack(psfs, axis=0)
#     z_pos = np.linspace(0, target_psf_shape[0] * voxel_sizes[0], target_psf_shape[0])
#
#     show_all_psfs(psfs)
#
#     # Save
#     return psfs, z_pos


if __name__ == '__main__':
    psfs, z_pos = gen_dataset(1)
    z_pos = np.tile(z_pos, psfs.shape[0])
    psfs = np.concatenate(psfs, axis=0)

    for psf, z in zip(psfs, z_pos):
        print(psf.shape)
        fname = f'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/tmp/{z}.png'
        imwrite(fname, psf)
    # psfs, z_pos = gen_dataset()
    # show_all_psfs(psfs)

    # cfg = {'oblique astigmatism': 1}
    # psf = gen_psf_named_params(cfg)
    # show_psf_axial(psf)
