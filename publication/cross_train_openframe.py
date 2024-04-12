from tifffile import imread
import pandas as pd
import os
import numpy as np

datasets = [
    '/home/miguel/Projects/data/20231020_20nm_beads_10um_range_10nm_step/combined/',
    '/home/miguel/Projects/data/20231128_tubulin_miguel/combined/',
    '/home/miguel/Projects/data/all_openframe_beads/combined/'
]


def load_dataset(dpath):
    args = {
        'stacks': imread(os.path.join(dpath, 'stacks.ome.tif')),
        'locs': pd.read_hdf(os.path.join(dpath, 'locs.hdf'))
    }

    psfs = imread(args['stacks'])[:, :, :, :, np.newaxis].astype(float)
    locs = pd.read_hdf(args['locs'], key='locs')
    locs['idx'] = np.arange(locs.shape[0])

    if args['debug']:
        idx = np.arange(psfs.shape[0])
        np.random.seed(42)
        idx = np.random.choice(idx, 2000)
        idx = idx[0:100]
        psfs = psfs[idx]
        locs = locs.iloc[idx]

    ys = []
    for offset in locs['offset']:
        zs = ((np.arange(psfs.shape[1])) * args['zstep']) - offset
        ys.append(zs)

    ys = np.array(ys)

    return psfs, locs, ys



def main():
    datasets = [load_dataset(d) for d in datasets]

    for i in range(len(datasets)):
        



if __name__=='__main__':
    main()