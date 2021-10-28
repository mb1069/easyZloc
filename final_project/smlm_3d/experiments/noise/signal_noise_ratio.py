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

z_range = 10000


def estimate_nsr_ratio(psf):
    psf = psf / psf.max()
    axial_max = psf.max(axis=(1,2))
    peak = axial_max.max()
    # Avoid division by 0 errors
    bkgd_noise = max(axial_max.min(), 1e-25)
    nsr = bkgd_noise / peak
    return nsr

def gather_nsr_ratios(cfg):
    snrs = []
    dataset = TrainingDataSet(cfg, z_range, lazy=True, normalize_psf=False, transform_data=False)
    dataset.prepare_debug()
    for i in range(dataset.total_emitters):
        try:
            psf, _, _, _, record = dataset.debug_emitter(i, z_range)
            snrs.append(estimate_nsr_ratio(psf))
        except RuntimeError:
            # Ignore emitters which cannot be modelled as gaussian
            pass
    return snrs
    
def main():
    
    cfg = {
        'simulated_training': dataset_configs['simulated_ideal_psf']['training'],
        'simulated_sphere': dataset_configs['simulated_ideal_psf']['sphere_ground_truth'],
    }
    for ds in ['olympus', 'openframe', 'other']:
        cfg[ds+'_training'] = dataset_configs[ds]['training']
        cfg[ds+'_sphere'] = dataset_configs[ds]['sphere_ground_truth']

    snrs = {k: gather_nsr_ratios(cfg) for k, cfg in cfg.items()}
    labels, data = [*zip(*snrs.items())]
    for k, v in snrs.items():
        print(k)
        print(f'\t{np.mean(v)}')
        print(f'\t{np.std(v)}')
        print('\n')
    plt.boxplot(data)
    plt.xlabel(labels)
    plt.ylabel('Noise to signal ratio')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()


