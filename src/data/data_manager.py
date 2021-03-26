import shutil

from sklearn.model_selection import train_test_split
from imageio import imwrite
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
import numpy as np
import os

from src.config.datafiles import storm_data_dir, matlab_data_dir, jonny_data_dir
from src.config.optics import bounds
from src.data.data_processing import process_STORM_datadir, process_MATLAB_data, process_multiple_MATLAB_data, \
    process_jonny_datadir
from src.zernike_decomposition.gen_psf import gen_dataset, min_max_norm

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


def load_jonny_datasets(test_size, datasets=None):
    X, y = process_jonny_datadir(jonny_data_dir, datasets=datasets, bound=bounds)
    X = reorder_channel(X)
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    X = min_max_norm(X)
    if test_size != 0:
        return split_datasets(X, y, test_size)
    return X, y


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
    print(f'Norm ratio: {z_pos.max()}')
    # z_pos = z_pos/z_pos.max()
    n_stacks = psfs.shape[0]
    X = np.concatenate(psfs, axis=0)
    ys = np.tile(z_pos, n_stacks)[:, np.newaxis]
    # dump_training_psfs(X, ys.squeeze())

    X = reorder_channel(X)
    return split_datasets(X, ys, test_size)


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, test_size=0.2, debug=False, jonny_datasets=None):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.debug = debug
        self.jonny_datasets = jonny_datasets

    def prepare_data(self, n_psfs):
        # self.train, self.val = load_storm_datasets(self.test_size)
        # self.train, self.val = load_matlab_datasets(self.test_size, self.debug)
        # self.train, self.val = load_test_datasets()
        # self.train, self.val = load_jonny_datasets(self.test_size, self.jonny_datasets)
        self.train, self.val = load_custom_psfs(n_psfs, self.test_size)
        print('Train', self.train.x.shape, self.train.y.shape)
        self.val = CustomDataset(*load_jonny_datasets(0))
        print('Val', self.val.x.shape, self.val.y.shape)



    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=64)
