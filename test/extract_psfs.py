from config.datasets import dataset_configs
from data.datasets import TrainingPicassoDataset
import os
from tifffile import imwrite
import shutil
TEST_DATA_DIR = '/home/miguel/Projects/uni/phd/smlm_z/test/psfs'

if __name__=='__main__':
    shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)

    for k, v in dataset_configs.items():
        for sub_k in v.keys():
            if 'training' in sub_k:
                cfg = dataset_configs[k][sub_k]
                try:
                    ds = TrainingPicassoDataset(cfg, z_range=1000, raw_data_only=True)
                except KeyError:
                    continue
                bead_stacks = ds.slice_locs_to_stacks()

                img_name = f'{k}_{sub_k}_beads.tif'
                imwrite(os.path.join(TEST_DATA_DIR, img_name), bead_stacks, compress=6)
