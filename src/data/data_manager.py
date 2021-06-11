import pickle
import shutil

from sklearn.model_selection import train_test_split
from tifffile import imwrite
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
import numpy as np
import os

from src.config.datafiles import storm_data_dir, matlab_data_dir, jonny_data_dir
from src.config.optics import bounds
from src.data.data_processing import process_STORM_datadir, process_MATLAB_data, process_multiple_MATLAB_data, \
    process_jonny_datadir, load_jonny_datasource
from src.wavelets.wavelet_data.datasets import JonnyDataSource
from src.wavelets.wavelet_data.util import limit_data_range, min_max_norm
from src.zernike_decomposition.gen_psf import gen_dataset

dtype = np.float32


class CustomDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]

        self.x = x.astype(dtype)
        self.y = y.astype(dtype)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]


def reorder_channel(x_train):
    return x_train[:, np.newaxis, :, :]


def split_datasets(X, y, test_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=1 - test_size)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    return train_dataset, val_dataset


def load_storm_datasets(test_size):
    X, y = process_STORM_datadir(storm_data_dir)
    X = X[0:34]
    y = y[0:34]
    X = reorder_channel(X)
    return split_datasets(X, y, test_size)


def load_jonny_datasets(test_size, datasets=None, limit_range=None):
    X, y = process_jonny_datadir(jonny_data_dir, datasets=datasets, bound=bounds)
    X = reorder_channel(X)
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    X = min_max_norm(X)
    if limit_range:
        X, y = limit_data_range(X, y, limit_range)
    if test_size != 0:
        return split_datasets(X, y, test_size)
    return X, y


def load_experimental_stacks(n_psfs, test_size):
    ds = JonnyDataSource()
    iter_dataset = ds.get_all_emitter_stacks(bound=20, z_type=True)
    psfs = []
    z_vals = []
    for i, (psf, z_pos) in zip(list(range(n_psfs)), iter_dataset):
        psfs.append(psf)
        z_vals.append(z_pos)

    psfs = np.concatenate(psfs)

    y = np.concatenate(z_vals)

    X = reorder_channel(psfs)

    if test_size == 0:
        return X, y
    return split_datasets(X, y, test_size)


def load_matlab_datasets(test_size, debug):
    matlab_files = [
        ('PSF_2_0to1in9_2in51_100.mat', 'Zpos_2_0to1in9_2in51_100.mat'),
        # ('PSF_2_0to1in9_2in101_100.mat', 'Zpos_2_0to1in9_2in101_100.mat'),
        # ('PSF_2to6_0to1in9_2in51_100.mat', 'Zpos_2to6_0to1in9_2in51_100.mat'),
        # (['PSF_2to6_0to1in9_2in101_100_1.mat', 'PSF_2to6_0to1in9_2in101_100_2.mat'],
        #  'Zpos_2to6_0to1in9_2in101_100.mat'),
    ]
    X = []
    ys = []
    for mfs, zpos in matlab_files:
        if isinstance(mfs, list):
            x, y = process_multiple_MATLAB_data(matlab_data_dir, mfs, zpos, normalise_images=True)
        else:
            x, y = process_MATLAB_data(matlab_data_dir, mfs, zpos, normalise_images=True)
        X.append(x)
        ys.append(y)

        # Load only a single dataset in debug mode
        if debug:
            break
    X = np.concatenate(X)
    ys = np.concatenate(ys)[:, np.newaxis]
    X = reorder_channel(X)

    X = X[0:10]
    ys = ys[0:10]

    print(f'Collected {X.shape[0]} datapoints')
    print(f'Z-range: {ys.min()} - {ys.max()}')

    return split_datasets(X, ys, test_size)
    # return CustomDataset(X, ys), CustomDataset(X, ys)


dump_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'raw_data', 'tmp')


def dump_training_psfs(psfs, z_pos):
    try:
        shutil.rmtree(dump_dir)
    except FileNotFoundError:
        pass
    os.makedirs(dump_dir, exist_ok=True)
    for psf, z in zip(psfs, z_pos):
        psf = psf.squeeze()
        fname = f'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/tmp/{z}.png'
        imwrite(fname, psf)


def load_custom_psfs(n_psfs, test_size):
    psfs, z_pos = gen_dataset(n_psfs, noise=True)
    n_stacks = psfs.shape[0]
    X = np.concatenate(psfs, axis=0)
    ys = np.tile(z_pos, n_stacks)

    X = reorder_channel(X)
    if test_size == 0:
        return CustomDataset(X, ys)
    return split_datasets(X, ys, test_size)


def load_corrected_datasets(test_size):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, 'psf.pickle'), 'rb') as f:
        X = pickle.load(f)

    with open(os.path.join(dirname, 'zpos.pickle'), 'rb') as f:
        y = pickle.load(f)

    if test_size == 0:
        return CustomDataset(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=1 - test_size)

    # synth_dataset = load_custom_psfs(1000, test_size=0)
    # X_train = np.concatenate((X_train, synth_dataset.x), axis=0)
    # y_train = np.concatenate((y_train, synth_dataset.y), axis=0)

    return CustomDataset(X_train, y_train), CustomDataset(X_val, y_val)


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, test_size=0.2, debug=False, jonny_datasets=None):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.debug = debug
        self.jonny_datasets = jonny_datasets

    def prepare_data(self, n_psfs):
        # Old approaches
        # self.train, self.val = load_storm_datasets(self.test_size)
        # self.train, self.val = load_matlab_datasets(self.test_size, self.debug)
        # self.train, self.val = load_test_datasets()
        # self.train, self.val = load_jonny_datasets(self.test_size, self.jonny_datasets)

        # Synth only
        # self.train, self.val = load_custom_psfs(n_psfs, self.test_size)

        # Exp only
        # self.train, self.val = load_corrected_datasets(self.test_size)

        # Mixed
        # self.train = load_custom_psfs(n_psfs, 0)
        # self.val = load_corrected_datasets(0)

        # Even / Odd mix
        # Even/Odd scheme
        # train_imgs = [f'{letter}{number}' for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] for number in range(1, 13, 2)]
        # test_imgs = [f'{letter}{number}' for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] for number in range(2, 13, 2)]
        #
        # X, y = load_jonny_datasource(img_names=train_imgs, z_type='synth')
        # X_val, y_val = load_jonny_datasource(img_names=test_imgs, z_type='synth')
        #
        # X, y = limit_data_range(X, y, z_range=1000)
        # X_val, y_val = limit_data_range(X_val, y_val, z_range=1000)
        #
        #
        # X = X[:, np.newaxis, :, :]
        # X_val = X_val[:, np.newaxis, :, :]
        #
        # self.train = CustomDataset(X, y)
        # self.val = CustomDataset(X_val, y_val)

        # Single well

        X, y = load_jonny_datasource(img_names=['A1'])
        X = X[:, np.newaxis, :, :]
        self.train, self.val = split_datasets(X, y, self.test_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def print_stats(self):
        print('Train')
        print(self.train.x.shape, self.train.y.shape)
        print(f'{self.train.y.min()}, {self.train.y.max()}')

        print('Val')
        print(self.val.x.shape, self.val.y.shape)
        print(f'{self.val.y.min()}, {self.val.y.max()}')

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=64)


if __name__ == '__main__':
    load_corrected_datasets(0.2)
    # psf = imread('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/0.tif')
    # dataset = load_experimental_stacks(100, 0.01)
    # data = list(zip(dataset.x, dataset.y))
    # data.sort(key=lambda d: d[1])
    #
    # psfs = [d[0] for d in data]
    # psfs = np.concatenate(psfs, axis=0)
    # psfs = psfs.astype(np.float32)
    # psfs = min_max_norm(psfs)
    # imwrite('/Users/miguelboland/Projects/uni/phd/smlm_z/src/tmp/tmp.tif', psfs, compress=6)
    # print('/Users/miguelboland/Projects/uni/phd/smlm_z/src/tmp/tmp.tif')
