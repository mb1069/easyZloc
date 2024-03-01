import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from picasso import io
from picasso.render import render
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import find_peaks
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=18)
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import shutil
from argparse import ArgumentParser


# old_locs = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/figure2/nup_cell_picked.hdf5'


# Openframe
# picked_locs = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted_picked_4.hdf5'
# PIXEL_SIZE = 86
# OVERSAMPLE = 30

# FD-LOCO
# picked_locs = '/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_locs_picked.hdf5'
# PIXEL_SIZE = 110
# OVERSAMPLE = 30


# Zeiss
# picked_locs = '/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_picked.hdf5'
# DEFAULT_PIXEL_SIZE = 106
# OVERSAMPLE = 30


min_sigma = 0
max_sigma = 3

z_min = 400
z_max = 600
min_log_likelihood = -100
# min_kde = np.log(0.007)
min_kde = 0.05

cmap_min_z = -600
cmap_max_z = -300
BLUR = 'gaussian'
color_by_depth = False

MIN_BLUR=0.001

records = []

def filter_locs(l):
    n_points = l.shape[0]
    print(f'From {n_points} points')

    l = l[(min_sigma < l['sx']) & (l['sx'] < max_sigma)]
    l = l[(min_sigma < l['sy']) & (l['sy'] < max_sigma)]
    # print(f'{n_points-l.shape[0]} removed by sx/sy')


    X = l[['z']]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    l['kde'] = kde.score_samples(X)

    # l = l[l['z [nm]'] > z_min]
    # l = l[l['z [nm]'] < z_max]
    # sns.scatterplot(data=l, x='z', y='kde')
    # plt.show()
    
    l = l[np.power(10, l['kde']) > min_kde]
    # print(f'{n_points-l.shape[0]} removed by kde')

    # l = l[l['likelihood']>min_log_likelihood]
    
    n_points2 = l.shape[0]
    # print(f'Removed {n_points-n_points2} pts')
    # print(f'{n_points2} remaining')
    print(f'N points: {n_points2}')

    return l


plt.rcParams['figure.figsize'] = [18, 6]


def apply_cmap_img(img, cmap_min_coord, cmap_max_coord, img_min_coord, img_max_coord, cmap='gist_rainbow', brightness_factor = 20):
    img = img.squeeze()
    
    cmap_zrange = cmap_max_coord - cmap_min_coord
    
    def map_z_to_cbar(z_val):
        return (z_val - cmap_min_coord) / cmap_zrange
        
    min_coord_color = map_z_to_cbar(img_min_coord)
    max_coord_color = map_z_to_cbar(img_max_coord)
    
    cmap = plt.get_cmap('gist_rainbow')
    
    gradient = np.repeat(np.linspace(min_coord_color, max_coord_color, img.shape[1])[np.newaxis, :], img.shape[0], 0)
    
    base = cmap(gradient)
    img = img[:, :, np.newaxis]
    cmap_img = img * base
    # cmap_img /= 2
    # Black background
    cmap_img = (cmap_img / cmap_img.max()) * 255
    cmap_img *= brightness_factor

    cmap_img[:, :, 3] = 255 
    
    cmap_img = cmap_img.astype(int)

    return cmap_img
    
def color_histplot(barplot, cmap_min_z, cmap_max_z):
    from matplotlib.colors import rgb2hex
    cmap = plt.get_cmap('gist_rainbow')
    
    bar_centres = [bar._x0 + bar._width/2 for bar in barplot.patches]
    bar_centres = np.array(list(map(lambda x: (x-cmap_min_z) / (cmap_max_z-cmap_min_z), bar_centres)))
    rgb_colors = cmap(bar_centres)
    hex_colors = [rgb2hex(x) for x in rgb_colors]
    
    for bar, hex_color in zip(barplot.patches, hex_colors):
        bar.set_facecolor(hex_color)
        

def center_view(locs, zrange=200):
    zs = locs['z [nm]']
    bin_width = 25
    hist, bins = np.histogram(zs, bins=np.arange(zs.min(), zs.max(), bin_width))
    try:
        max_bin_idx = np.argmax(hist)
        bin_val = bins[max_bin_idx] + (bin_width // 2)
    except ValueError:
        bin_val = np.mean(zs)

    locs = locs[(bin_val-zrange <=locs['z [nm]']) & (locs['z [nm]'] <= bin_val+zrange)]

    return locs

def get_viewport(locs, axes, margin=1):
    mins = np.array([locs[ax].min()-margin for ax in axes])
    maxs = np.array([locs[ax].max()+margin for ax in axes])
    # mins[:] = min(mins)
    # maxs[:] = max(maxs)
    return np.array([mins, maxs])

def disable_axis_ticks():
    plt.xticks([])
    plt.yticks([])

def get_extent(viewport, pixel_size):
    mins, maxs = viewport
    return np.array([mins[1], maxs[1], mins[0], maxs[0]]) * pixel_size


def render_locs(locs, args, ang_xyz=(0,0,0), barsize=None, ax=None):
    
    locs = locs.copy()
    locs['lpz'] = np.mean(locs[['lpx', 'lpy']].to_numpy()) / 2
    locs['sz'] = np.mean(locs[['sx', 'sy']].to_numpy()) / 3
    # locs['lpx'] = 0.1
    # locs['sx'] = 0.1
    # locs['lpy'] = 0.1
    # locs['sy'] = 0.1
    # disable_axis_ticks()
    locs['x [nm]'] -= locs['x [nm]'].mean()
    locs['y [nm]'] -= locs['y [nm]'].mean()
    locs['z [nm]'] -= locs['z [nm]'].mean()
    locs['x'] -= locs['x'].mean()
    locs['y'] -= locs['y'].mean()
    locs['z'] -= locs['z'].mean()

    viewport = get_viewport(locs, ('y', 'x'))

    _, img = render(locs.to_records(), blur_method=args['blur_method'], viewport=viewport, min_blur_width=args['min_blur'], ang=ang_xyz, oversampling=args['oversample'])
    if ang_xyz == (0, 0, 0):
        plt.xlabel('x [nm]')
        plt.ylabel('y [nm]')
    elif ang_xyz == (np.pi/2, 0, 0):
        plt.xlabel('z [nm]')
        plt.ylabel('x [nm]')
        img = img.T
        viewport = np.fliplr(viewport)

    elif ang_xyz == (0, np.pi/2, 0):
        plt.xlabel('z [nm]')
        plt.ylabel('y [nm]')
    else:
        print('Axis labels uncertain due to rotation angle')

    extent = get_extent(viewport, args['pixel_size'])
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal', 'box')
    img_plot = plt.imshow(img, extent=extent)
    plt.colorbar(img_plot)

    if barsize is not None:
        scalebar = AnchoredSizeBar(ax.transData,
                            barsize, f'{barsize} nm', 'lower center', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=1,
                            fontproperties=fontprops)
        ax.add_artist(scalebar)

def write_nup_plots(locs, args, good_dir, other_dir):
    for cid in set(locs['clusterID']):
        # if not cid in [23, 24, 4, 8, 10, 86, 75, 79, 88, 102]:
        #     continue
        print('Cluster ID', cid)

        df = locs[locs['clusterID']==cid]
        df = filter_locs(df)

        if df.shape[0] == 0:
            continue
        df = center_view(df)

        try:
            del df['index']
        except ValueError:
            pass

        if df.shape[0] < 5:
            print('No remaining localisations, continuing...')
            continue

        fig = plt.figure()
        gs = fig.add_gridspec(1, 4)
        plt.subplots_adjust(wspace=0.3, hspace=0)
        
        ax1 = fig.add_subplot(gs[0, 0])
        render_locs(df, args, (0,0,0), barsize=110, ax=ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        render_locs(df, args, (np.pi/2,0,0), barsize=50, ax=ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        render_locs(df, args, (0, np.pi/2,0), barsize=50, ax=ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        
        histplot = sns.histplot(data=df, x='z [nm]', ax=ax4, stat='density', legend=False)
        if color_by_depth:
            color_histplot(histplot, cmap_min_z, cmap_max_z)
        sns.kdeplot(data=df, x='z [nm]', ax=ax4, bw_adjust=0.5, color='black', bw_method='silverman')

        x = ax4.lines[0].get_xdata()
        y = ax4.lines[0].get_ydata()
        peaks, _ = find_peaks(y)

        sorted_peaks = sorted(peaks, key=lambda peak_index: y[peak_index], reverse=True)
        peak_vals = y[peaks]
        if len(peak_vals) == 1:
            n_peaks = 1
        else:
            n_peaks = 2
            
        sorted_peaks = sorted_peaks[:n_peaks]

        peak_x = x[sorted_peaks]
        peak_y = y[sorted_peaks]
        for x, y in zip(peak_x, peak_y):
            ax4.vlines(x, 0, y, label=str(round(x)), color='black')

        sep = abs(max(peak_x) - min(peak_x))

        septxt = 'Sep: '+ str(round(sep))+ 'nm'

        records.append({
            'id': cid,
            'seperation': sep,
        })

        margin=10
        if 50-margin <= sep and sep <= 50+margin:
            cluster_outdir = good_dir
        else:
            cluster_outdir = other_dir
        plt.suptitle(f'Nup ID: {cid}, N points: {df.shape[0]}, {septxt}')
        plt.savefig(os.path.join(cluster_outdir, f'nup_{cid}_{BLUR}.png'))
        plt.close()

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(args['outdir'], 'nup_report.csv'))


def prep_dirs(args):

    shutil.rmtree(args['outdir'], ignore_errors=True)
    os.makedirs(args['outdir'], exist_ok=True)

    good_dir = os.path.join(args['outdir'], 'good_results')
    other_dir = os.path.join(args['outdir'], 'other_results')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    return good_dir, other_dir


def load_and_filter_locs(args):
    locs, info = io.load_locs(args['locs'])
    locs = pd.DataFrame.from_records(locs)
    assert info[1]['Pixelsize'] == args['pixel_size']

    if args['picked_locs']:
        picked_locs, old_info = io.load_locs(args['picked_locs'])
        picked_locs = pd.DataFrame.from_records(picked_locs)
        locs = locs.merge(picked_locs, on=['x', 'y', 'photons', 'bg', 'lpx', 'lpy', 'net_gradient', 'iterations', 'frame', 'likelihood', 'sx', 'sy'])
    locs['clusterID'] = locs['group']
    locs['z'] = locs['z [nm]'] / args['pixel_size']
    return locs


def main(args):
    good_dir, other_dir = prep_dirs(args)
    # Save a copy of this script for reproducibility
    shutil.copy(os.path.abspath(__file__), os.path.join(args['outdir'], 'render_nup.py.bak'))

    locs = load_and_filter_locs(args)

    write_nup_plots(locs, args, good_dir, other_dir)

    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--locs', default='./locs_3d.hdf')
    parser.add_argument('-px', '--pixel-size', default=86)
    parser.add_argument('-p', '--picked-locs')
    parser.add_argument('-o', '--outdir', default='./nup_renders3')
    parser.add_argument('-os', '--oversample', default=30)
    parser.add_argument('-mb', '--min-blur', default=0.001)
    parser.add_argument('-b', '--blur-method', default='gaussian')
    return vars(parser.parse_args())


if __name__=='__main__':
    main(parse_args())