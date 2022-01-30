from xmlrpc.client import TRANSPORT_ERROR
from final_project.smlm_3d.data.visualise import scatter_3d, show_psf_axial
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import imsave

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet, SphereTrainingDataset
from final_project.smlm_3d.util import get_base_data_path, chunks
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs

        
def grid_psfs(psfs, cols=10):
    rows = (len(psfs) // cols) + 1
    n_spaces = int(cols * rows)
    print(f'Rows {rows} Cols {cols} n_spaces {n_spaces} n_psfs {len(psfs)}')
    if n_spaces > len(psfs):
        black_placeholder = np.zeros((n_spaces-len(psfs), *psfs[0].shape))
        psfs = np.concatenate((psfs, black_placeholder))
        cols = len(psfs) // rows
    psfs = list(chunks(psfs, cols))

    psfs = [np.concatenate(p, axis=2) for p in psfs]
    psfs = np.concatenate(psfs, axis=1)
    return psfs


def main():

    # cfg = dataset_configs['paired_bead_stacks']['training']
    # train_dataset = TrainingDataSet(cfg, 100000, lazy=True)
    # psfs, coords = train_dataset.fetch_emitters_modelled_z()

    dataset = 'matched_index_sphere'
    cfg = dataset_configs[dataset]['sphere_ground_truth_647nm']
    exp_dataset = SphereTrainingDataset(cfg, transform_data=False, z_range=100000, split_data=True, add_noise=False, radius=5e5+50, lazy=True)
    psfs, coords = exp_dataset.fetch_emitters_modelled_z()

    
    psfs = np.array(psfs)
    coords = np.stack(coords)
    # for psf, coord in list(zip(psfs, coords))[120:]:
    #     plt.title(', '.join([str(c) for c in coord[0][1:3]]))
    #     plt.imshow(psf[100])
    #     plt.show()

    psfs = grid_psfs(psfs)
    impath = os.path.join(cfg['bpath'], cfg['img'].replace('.ome.tif', '_concat.ome.tif'))
    plt.imshow(psfs[psfs.shape[0]//2])
    plt.show()
    imsave(impath, psfs, compress=6)
    print(impath)

    # for psf, coord in zip(psfs, coords):
    #     # plt.title(", ".join([str(round(n/1000, 2)) for n in coord[-2:]]))
    #     show_psf_axial(psf, subsample_n=3)

if __name__ == '__main__':
    main()


