from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import os
import pytorch_lightning as pl

from src.data.data_processing import process_STORM_datadir, process_jonny_tif
from src.data_manager import storm_data_dir, reorder_channel, jonny_data_dir
from src.reptile.dataset import TaskDataset
from src.reptile.task_sampler import TaskSampler


def split_datasets(X, y, test_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=1 - test_size)

    train_dataset = TaskDataset(X_train, y_train)
    val_dataset = TaskDataset(X_val, y_val)
    return train_dataset, val_dataset


def load_storm_datasets(test_size):
    X, y = process_STORM_datadir(storm_data_dir)
    X = X[0:34]
    y = y[0:34]
    X = reorder_channel(X)
    return split_datasets(X, y, test_size)


def process_jonny_datadir_by_task(bound=16, y_dims=1, pixel_size=106, normalise_images=True,
                                  datasets=None, test_size=0.1):
    print("Processing Jonny data")
    # Storage arrays
    sample_images = glob(os.path.join(jonny_data_dir, '*', 'MMStack_Default.ome.tif'))
    if datasets:
        sample_images = [sample_images[i] for i in datasets]

    x_train = []
    x_test = []

    y_train = []
    y_test = []

    for image in sample_images:
        truth = os.path.join(os.path.dirname(image), 'stack', 'MMStack_Default.csv')
        x, y = process_jonny_tif(image, truth, bound=bound, pixel_size=pixel_size, normalise_images=normalise_images)

        _x_train, _x_test, _y_train, _y_test = train_test_split(x, y, train_size=1 - test_size)

        x_train.append(_x_train)
        x_test.append(_x_test)
        y_train.append(_y_train)
        y_test.append(_y_test)

    # Reshape x_train for direct input into CNN
    # Add channel in
    print(f'Collected {x_train.shape[0]} datapoints')
    print(f'Z-range: {y_train.min()} - {y_train.max()}')
    return x_train, y_train


class TaskDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers):

class ReptileDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, test_size=0.2, debug=False, jonny_datasets=None):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.debug = debug
        self.jonny_datasets = jonny_datasets

    def prepare_data(self):
        # self.train, self.val = load_storm_datasets(self.test_size)
        # self.train, self.val = load_matlab_datasets(self.test_size, self.debug)
        # self.train, self.val = load_test_datasets()
        self.train, self.val = process_jonny_datadir_by_task(self.test_size, self.jonny_datasets)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8,
                          sampler=TaskSampler(self.train, num_samples=3))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)
