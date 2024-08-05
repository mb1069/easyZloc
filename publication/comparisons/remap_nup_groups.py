from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np
import shutil
import os
import h5py

PIXEL_SIZE = 106

def main(args):
    df_fdloco = pd.read_hdf(args['target_nup_locs'], key='locs')
    df_ours = pd.read_hdf(args['src_nup_locs'], key='locs')

    fd_loco_groups = df_fdloco.groupby('group').mean().reset_index().sort_values('group')
    ours_groups = df_ours.groupby('group').mean().reset_index().sort_values('group')
    from sklearn.metrics import pairwise_distances

    fd_loco_xy = fd_loco_groups[['x', 'y']].to_numpy()
    ours_xy = ours_groups[['x', 'y']].to_numpy()
    fd_loco_groups = fd_loco_groups['group'].to_numpy().astype(int)

    dists = pairwise_distances(ours_xy, fd_loco_xy)
    inv_mapping = {}
    mapping_dist = {}
    for i, g in enumerate(fd_loco_groups):

        idx = np.argmin(dists[:, i])
        dist = np.min(dists[:, i])
        target_group = int(ours_groups['group'].to_numpy()[idx])

        previous_dist = mapping_dist.get(target_group) or np.inf
        if dist < 5 and dist < previous_dist:
            mapping_dist[target_group] = dist
            print(f'{g} -> {target_group}')
            inv_mapping[target_group] = g
        

    mapping = {v: k for k, v in inv_mapping.items()}
    df_fdloco['group'] = df_fdloco['group'].map(lambda x: mapping.get(x) or -1)
    df_fdloco = df_fdloco[df_fdloco['group']!=-1]


    df_fdloco['z'] = df_fdloco['z [nm]'] / PIXEL_SIZE


    del df_fdloco['index']
    outpath = args['target_nup_locs'].replace('.hdf5', '_matched.hdf5')
    print(outpath)
    with h5py.File(outpath, "w") as locs_file:
        locs_file.create_dataset("locs", data=df_fdloco.to_records())

    yaml_file = args['target_nup_locs'].replace('.hdf5', '.yaml')
    if os.path.exists(yaml_file):
        dest_yaml = outpath.replace('.hdf5', '.yaml')
        shutil.copy(yaml_file, dest_yaml)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_nup_locs')
    parser.add_argument('target_nup_locs')
    return parser.parse_args()


if __name__=='__main__':
    args = vars(parse_args())
    main(args)