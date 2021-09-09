
from final_project.smlm_3d.data.datasets import TrainingDataSet
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.config.datasets import dataset_configs

from final_project.smlm_3d.experiments.dataset_classifier import load_discriminator
from final_project.smlm_3d.util import dwt_inverse_dataset
import matplotlib.pyplot as plt

import pandas as pd
import ot
import numpy as np
def load_datasets(batch=True):
    z_range = 1000

    dataset = 'openframe'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=True)

    exp_dataset = TrainingDataSet(dataset_configs[dataset]['sphere_ground_truth'], z_range, transform_data=True)

    return train_dataset, exp_dataset

def plot_dwt_coeff(train_ds, train_z, exp_ds, exp_z, coeff):
    plt.scatter(train_z, train_ds[:, coeff], label='train')
    plt.scatter(exp_z, exp_ds[:, coeff], label='test')
    plt.show()

def main():
    train_ds, exp_ds = load_datasets(batch=False)
    src_dwt, src_z = train_ds.data['train']

    target_dwt, target_z = exp_ds.data['train']


    imgs = dwt_inverse_dataset(src_dwt)
    print(imgs.shape)
    quit()


    df = pd.DataFrame(target_dwt)
    df['z'] = target_z
    c = df.corrwith(df['z']).abs()
    print(src_dwt.shape)

    cols_to_keep = np.where(c > 0.25)[0][0:-2]
    src_dwt = src_dwt[:, cols_to_keep]
    target_dwt = target_dwt[:, cols_to_keep]
    print(src_dwt.shape)
    so = c.sort_values(kind="quicksort", ascending=False)
    best_fits = list(so.index)


    # for i in range(5):
    #     plot_dwt_coeff(src_dwt, src_z, target_dwt, target_z, i)

    ot_l1l2 = ot.da.SinkhornTransport(max_iter=20,
                                        verbose=True)
    ot_l1l2.fit(Xs=src_dwt, Xt=target_dwt)

    print(src_dwt.shape)
    print(target_dwt.shape)
    src_dwt = ot_l1l2.transform(src_dwt)
    print('transformed!')
    for i in range(5):
        plot_dwt_coeff(src_dwt, src_z, target_dwt, target_z, i)

    
if __name__ == '__main__':
    main()