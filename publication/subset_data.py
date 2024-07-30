from tifffile import imread, imwrite
import os
import shutil
import pandas as pd
from sklearn.metrics import euclidean_distances
import numpy as np

dirname = '/home/miguel/Projects/smlm_z/publication/VIT_031_redo/'
# outdir = dirname.replace('combined', 'combined_subset')
outdir = '/home/miguel/Projects/smlm_z/publication/VIT_031_redo_subset/'
os.makedirs(outdir, exist_ok=True)


shutil.copy(dirname + 'stacks_config.json', outdir)

stacks = dirname + 'stacks.ome.tif'

locs = dirname + 'locs.hdf'

df = pd.read_hdf(locs, key='locs')

stacks = imread(stacks)

max_dist = 200

print(f'Starting from {df.shape[0]} PSFs')

center_x = (df['x'].max() - df['x'].min()) / 2
center_y = (df['y'].max() - df['y'].min()) / 2

dists = euclidean_distances(df[['x', 'y']].to_numpy(), [[center_x, center_y]]).squeeze()

print(dists.min(), dists.max())

print(dists.shape)

idx = np.argwhere(dists < max_dist).squeeze()

print(f'Retaining {len(idx)} PSFs')

df = df.iloc[idx]
stacks = stacks[idx]

df.to_hdf(outdir+'locs.hdf', key='locs')
imwrite(outdir+'stacks.ome.tif', stacks)
