from final_project.smlm_3d.data.visualise import scatter_3d, show_psf_axial
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.util import get_base_data_path
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs

def main():

    # Run on exp data
    z_range = 2000

    train_dataset = TrainingDataSet(dataset_configs['olympus']['training'], z_range, lazy=True)
    psfs, coords = train_dataset.fetch_emitters()
    # plt.scatter(train_dataset.csv_data['x [nm]'], train_dataset.csv_data['y [nm]'])
    # plt.show()
    for psf, coord in zip(psfs, coords):
        # plt.title(", ".join([str(round(n/1000, 2)) for n in coord[-2:]]))
        show_psf_axial(psf, subsample_n=3)

if __name__ == '__main__':
    main()


