
from data.datasets import TrainingDataSet, ExperimentalDataSet
from config.datafiles import res_file
from config.datasets import dataset_configs
from data.visualise import scatter_3d, show_psf_axial
from workflow_v2 import eval_model
from experiments.deep_learning import load_model
from debug_tools.est_calibration_stack_error import fit_plane

import numpy as np
import matplotlib.pyplot as plt
import dill
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
import os

if __name__=='__main__':
    z_range = 1000
    dataset = 'paired_bead_stacks'

    sub_dataset = 'experimental'
    cfg = dataset_configs[dataset][sub_dataset]

    dataset = TrainingDataSet(cfg, transform_data=False, lazy=True, z_range=1000)
    model = load_model()

    coords = dataset.estimate_ground_truth()
    print(coords.shape)
    coords = fit_plane(coords)[2]

    out_csv = os.path.join(cfg['bpath'], cfg['csv'].replace('.csv', '_flattened.csv'))

    print('CSV_DATA', dataset.csv_data.shape)
    sub_df = dataset.csv_data.iloc[coords]
    print('NEW CSV_DATA', sub_df.shape)
    sub_df.to_csv(out_csv, sep=',')
    print(out_csv)