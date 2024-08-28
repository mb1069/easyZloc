import pandas as pd
import h5py


PIXEL_SIZE = 106

df = pd.read_csv('./emitter.csv', skiprows=3)

df['y'] = 1200 - df['y']

df['z'] /= PIXEL_SIZE

for c in ['x', 'y']:
    df[f'{c} [nm]'] = df[c]*PIXEL_SIZE


for c in ['lpx', 'lpy', 'lpz']:
    df[c] = 0.05

for c in ['iterations', 'likelihood']:
    df[c] = 1

df['net_gradient'] = 1000

col_map = {
    'frame_ix': 'frame',
    'x_sig': 'sx',
    'y_sig': 'sy',
    'z_sig': 'sz',
    'phot': 'photons',
}

df[['x', 'x [nm]', 'y', 'y [nm]']] = df[['y', 'y [nm]', 'x', 'x [nm]']]

for c in ['id', 'bg_cr', 'bg_sig', 'x_cr', 'y_cr', 'z_cr', 'phot_cr', 'phot_sig']:
    del df[c]

if 'index' in list(df):
    del df['index']
df = df.rename(columns=col_map)
with h5py.File('./emitter_remapped.hdf5', 'w') as f:
    f.create_dataset('locs', data=df.to_records())

