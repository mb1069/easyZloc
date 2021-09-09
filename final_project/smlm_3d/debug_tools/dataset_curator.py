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
from final_project.smlm_3d.data.visualise import concat_psf_axial
from tqdm import trange

def get_key():
    a = input('keep?')
    if a == 'y':
        return True
    elif a == 'n':
        return False
    elif a == 'x':
        quit(0)

def main():
    # Run on exp data
    z_range = 2000

    cfg = dataset_configs['other']['single_slice']

    train_dataset = TrainingDataSet(cfg, z_range, lazy=True)
    train_dataset.prepare_debug()
    # plt.scatter(train_dataset.csv_data['x [nm]'], train_dataset.csv_data['y [nm]'])
    # plt.show()
    records = []
    i = -1
    plt.ion()
    plt.show()

    csv_path = os.path.join(cfg['bpath'], cfg['csv'].replace('.csv', '_filtered.csv'))

    for i in trange(train_dataset.total_emitters):
        try:
            psf, dwt, coords, z, record = train_dataset.debug_emitter(i, 1000)
        except RuntimeError:
            continue
        sub_psf = concat_psf_axial(psf, 3)
        plt.imshow(sub_psf)
        plt.draw()
        
        if get_key():
            records.append(record)
            print(f'kept {len(records)}')

    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
    print(csv_path)
    # for psf, coord in zip(psfs, coords):
    #     plt.title(", ".join([str(round(n/1000, 2)) for n in coord[-2:]]))
    #     show_psf_axial(psf, subsample_n=3)

if __name__ == '__main__':
    main()


