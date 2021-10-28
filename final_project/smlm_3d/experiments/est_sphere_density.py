from final_project.smlm_3d.data.visualise import scatter_3d, scatter_yz
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.data.datasets import ExperimentalDataSet
from final_project.smlm_3d.debug_tools.dataset_simulator.sphere import radius
import pandas as pd
import numpy as np

def estimate_density(cfg):
    truth_dataset = ExperimentalDataSet(cfg, lazy=True)

    n_emitters = truth_dataset.csv_data.shape[0]
    area = np.power(cfg['voxel_sizes'][1] * truth_dataset.img.shape[1], 2)

    return n_emitters / area

def main():
    spheres = {}
    for dataset in ['olympus', 'openframe', 'other']:
        cfg = dataset_configs[dataset]['sphere_ground_truth']
        spheres[dataset] = "{:e}".format(estimate_density(cfg))
    
    densities = [1e-8, 2.5e-8, 5e-8, 7.5e-8, 1e-7, 2.5e-7, 5e-7]
    base_cfg = dataset_configs['simulated_ideal_psf']['sphere_ground_truth']
    for d in densities:
        new_cfg = {k: v for k, v in base_cfg.items()}
        new_cfg['bpath'] = str(new_cfg['bpath']) + f'_{d}'
        spheres[str(d)] = "{:e}".format(estimate_density(new_cfg))

    print('\n')
    for k, v in spheres.items():
        print(k, v)

if __name__=='__main__':
    main()

