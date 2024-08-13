import argparse
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
import h5py
import shutil

def plot_drift(df, outpath):
    group_df = df.groupby('frame').mean()
    fig = sns.scatterplot(data=group_df, x='frame', y='z [nm]', alpha=0.05, label='z locs (nm)').get_figure()
    fig.savefig(outpath)
    plt.close()
    print('Plotted drift')

def save_new_locs(df, args):
    print('Saving new locs')
    locs_undrift_path = args['locs'].replace('locs_3d.hdf5', f'locs_3d_undrift_z.hdf5')

    with h5py.File(locs_undrift_path, "w") as locs_file:
        locs_file.create_dataset("locs", data=df.to_records())
    shutil.copyfile(args['locs'].replace('.hdf5', '.yaml'), locs_undrift_path.replace('.hdf5', '.yaml'))
    print(f'Wrote locs to {locs_undrift_path}')


def undrift(df, outpath, args):
    print('Undrifting')

    df = df.copy(deep=True)

    def frame_center(x):
        return x.left + (args['n_frames']/2)
        
    df['frame_bin'] = (pd.cut(df['frame'].astype(int), np.arange(0, df['frame'].max()+args['n_frames'], args['n_frames']), include_lowest=True))
    df_group_bin = df.groupby('frame_bin').mean().reset_index()
    df_group_bin['frame_bin_center'] = df_group_bin['frame_bin'].map(frame_center)

    x = df_group_bin['frame_bin_center']
    y = df_group_bin['z [nm]']

    spline = BSpline(*splrep(x, y, s=len(y)*100))
    print('Fitted spline')

    _x = np.arange(df['frame'].min(), df['frame'].max()+1)
    _y = spline(_x)

    fig, ax = plt.subplots()
    group_df = df[['frame', 'z [nm]']].groupby('frame').mean()
    sns.scatterplot(data=group_df, x='frame', y='z [nm]', alpha=0.05, label='z locs (nm)', ax=ax)
    sns.lineplot(x=_x, y=_y, label='Spline fit', c='red', ax=ax)
    plt.ylabel('Mean localisation z (nm)')
    plt.savefig(os.path.join(outpath, 'z_drift_spline_fit.png'))
    plt.close()

    del df['frame_bin']
    del df['index']
    df['z [nm]'] -= df['frame'].map(lambda f: _y[f])
    df['z'] -= df['frame'].map(lambda f: _y[f])
    return df


def plot_hists(raw_df, corrected_df, outpath):
    fig, axs = plt.subplots(1, 2)
    sns.histplot(data=raw_df, x='z [nm]', ax=axs[0])
    sns.histplot(data=corrected_df, x='z [nm]', ax=axs[1])
    plt.title('Pre (left) and Post (right) drift correction')
    plt.savefig(os.path.join(outpath, 'z_drift_hists.png'))
    plt.close()


def main(args):
    outpath = os.path.dirname(args['locs'])
    # locs_path = '/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_resnet_fov-max_aug11/out_nup/locs_3d.hdf5'
    # locs_path = '/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_33/out_nup/locs_3d.hdf5'
    # locs_path = '/home/miguel/Projects/smlm_z/publication/VIT_fd-loco3/out_3/out_nup/locs_3d.hdf5'
    df = pd.read_hdf(args['locs'], key='locs')
    print(f'Loaded locs {df.shape}')

    
    plot_drift(df, os.path.join(outpath, 'init_drift.png'))

    df_out = undrift(df, outpath, args)

    save_new_locs(df_out, args)
    plot_drift(df_out, os.path.join(outpath, 'corrected_drift.png'))
    plot_hists(df, df_out, outpath)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('locs')
    parser.add_argument('--n-frames', type=int, default=500, help='N frames per chunk to estimate drift')
    return parser.parse_args()

def run_tool():
    args = vars(parse_args())
    main(args)

if __name__ == '__main__':
    run_tool()