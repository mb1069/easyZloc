from scipy.sparse import data
from torch.utils.data.dataloader import DataLoader
from data.visualise import scatter_3d
import os

import matplotlib.pyplot as plt
import numpy as np

from data.datasets import TrainingDataSet, ExperimentalDataSet
from config.datafiles import res_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules import Conv2d
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset
from experiments.deep_learning import train_model, save_model
from debug_tools.est_calibration_stack_error import fit_plane


def extract_flattened_psfs(dataset):
    psfs, xyz_coords = dataset.estimate_ground_truth()
    dists_to_plane, _, subset_idx = fit_plane(xyz_coords)
    print(np.stack(psfs).shape)
    print(xyz_coords.shape)
    quit()


def main():
    
    z_range = 1000

    dataset = 'paired_bead_stacks'

    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=False, lazy=True)
    # exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False, lazy=True)

    training_data = extract_flattened_psfs(train_dataset)

    model = train_model(train_dataset.data, train_dataset.data['val'])
    save_model(model)

    model = load_model()
    eval_model(model, train_dataset.data['train'], 'Bead test (bead training)')
    
    eval_model(model, exp_dataset.data['train'], 'Bead test 2 (bead training)', shift_correction=True)
    # 67.44nm


    # eval_model(model, exp_dataset.data['train'], 'Bead test 2 (bead training)', shift_correction=False)

    # model = load_model()
    # coords = exp_dataset.predict_dataset(model)
    # scatter_3d(coords)

    # eval_model(model, exp_dataset.data['test'], 'Sphere (sphere training)')

    # eval_model(model, train_dataset.data['test'], 'Bead stack')

# MAE: 72.4392
# MAE: 136.6815

if __name__ == '__main__':
    main()
