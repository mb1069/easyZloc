import glob
from collections import OrderedDict

import numpy as np
from tifffile import imread

from src.config.optics import target_psf_shape, voxel_sizes
from src.data.data_processing import process_jonny_datadir
from src.data.data_manager import reorder_channel, jonny_data_dir
import os
import torch
import matplotlib.pyplot as plt
from src.z_estimation.train_alt import Model
from src.zernike_decomposition.gen_psf import gen_psf_named_params_raw_psf, gen_dataset

model_path = os.path.join(os.path.dirname(__file__), 'model.pth')


def get_available_devices():
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.device_count()
    return torch.device('cpu'), 0


def load_trained_model(device):
    model = Model()

    state_dict = torch.load(model_path, map_location=device[0])

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def normalise_dataset(y):
    return y / y.max()


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def eval_results(y_true, y_pred):
    y_true = normalise_dataset(y_true)
    y_pred = normalise_dataset(y_pred)

    print(f'RMSE: {rmse(y_pred, y_true)}')


def eval_jonny_datasets(model):
    X, y_true = process_jonny_datadir(jonny_data_dir, datasets=[0])
    X = reorder_channel(X)

    y_pred = model.pred(X)

    eval_results(y_true, y_pred)


def eval_emitter_stack(model):
    imgs = glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif')
    for img in imgs[0:5]:
        X = imread(img)
        img, z_pos = gen_dataset(1)
        z_pos_2 = np.linspace(0, target_psf_shape[0] * voxel_sizes[0], target_psf_shape[0])
        X = normalise_dataset(X)
        X = reorder_channel(X)
        z_pred = model.pred(X) * 4100

        x = np.linspace(0, X.shape[0], X.shape[0])
        plt.title(img)
        plt.axis('on')
        plt.plot(x, z_pos)
        plt.plot(x, z_pred)

        diff = z_pos - z_pred.squeeze()
        plt.plot(x, np.abs(diff))
        plt.show()
        quit()


def main():
    device = get_available_devices()
    model = load_trained_model(device)
    # eval_jonny_datasets(model)
    eval_emitter_stack(model)


if __name__ == '__main__':
    main()
