from final_project.smlm_3d.data.visualise import scatter_3d, show_psf_axial
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_project.smlm_3d.data.datasets import TrainingDataSet
from final_project.smlm_3d.util import get_base_data_path
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.data.visualise import concat_psf_axial
from tqdm import trange
from skimage.metrics import structural_similarity as ssim

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
    z_range = 200

    cfg = dataset_configs['paired_bead_stacks']['experimental']

    SMART_SELECTION = False

    train_dataset = TrainingDataSet(cfg, z_range, lazy=True, transform_data=False, normalize_psf=True)
    train_dataset.prepare_debug()
    # plt.scatter(train_dataset.csv_data['x [nm]'], train_dataset.csv_data['y [nm]'])
    # plt.show()
    records = []
    i = -1
    plt.ion()
    plt.show()

    csv_path = os.path.join(cfg['bpath'], cfg['csv'].replace('.csv', '_filtered.csv'))
    print(csv_path)

    accepted_images = []
    use_ssim_val = False
    accept_ssim_threshold = 0.93
    reject_ssim_threshold = 0.85
    for i in trange(train_dataset.total_emitters):
        try:
            psf, dwt, coords, z, record = train_dataset.debug_emitter(i, 1000)
        except RuntimeError:
            continue
        sub_psf = concat_psf_axial(psf, 7)
        if len(accepted_images) > 2:
            example_psf = np.stack(accepted_images).mean(axis=(0))
            val = round(ssim(example_psf, psf), 5)
            plt.title(str(val))
        else:
            val = 0
        
        if use_ssim_val and val <= reject_ssim_threshold:
            accept_image = False
            print(f'Auto rejecting {val}')
        elif use_ssim_val and val >= accept_ssim_threshold:
            accept_image = True
            print('Auto accepting')
        else:
            plt.imshow(sub_psf)
            plt.draw()
            accept_image = get_key()

        if accept_image:
            records.append(record)
            accepted_images.append(psf)
            print(f'kept {len(records)}')
        
        if SMART_SELECTION and len(accepted_images) == 10:
            use_ssim_val = True


    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
    print(csv_path)
    # for psf, coord in zip(psfs, coords):
    #     plt.title(", ".join([str(round(n/1000, 2)) for n in coord[-2:]]))
    #     show_psf_axial(psf, subsample_n=3)

if __name__ == '__main__':
    main()


