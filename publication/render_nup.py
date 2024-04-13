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
from scipy.stats import gaussian_kde
import wandb
import json

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
    if 'lpz' not in list(locs):
        locs['lpz'] = np.mean(locs[['lpx', 'lpy']].to_numpy()) / 2
    if 'sz' not in list(locs):
        if not ('sx' in list(locs) and 'sy' in list(locs)):
            locs['sx'] = 0.5
            locs['sy'] = 0.5
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
    # plt.colorbar(img_plot)

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
    good_nup = 0
    bad_nup = 0
    for cid in set(locs['clusterID']):
        # if not cid in [6, 18, 19, 21, 22]:
        #     continue
        print('Cluster ID', cid, end='')

        df = locs[locs['clusterID']==cid]
        if not args['disable_filter']:
            df = filter_locs(df).copy()

        if df.shape[0] == 0:
            print('')
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

        ax4 = fig.add_subplot(gs[0, 3])
        sns.histplot(data=df, x='z [nm]', ax=ax4, stat='density', legend=False, bins=40)


        # Fit KDE to z vals
        kde = gaussian_kde(df['z [nm]'].to_numpy())
        kde.set_bandwidth(bw_method='silverman')
        kde.set_bandwidth(kde.factor * args['kde_factor'])

        zvals = np.linspace(df['z [nm]'].min()-25, df['z [nm]'].max()+25, 5000)
        score = kde(zvals)
        zvals = zvals.squeeze()

        ax4.plot(zvals, score, label='KDE')

        peak_idx, _ = find_peaks(score)
        peak_z = zvals[peak_idx]
        peak_score = score[peak_idx]

        peak_scores_sorted = np.argsort(peak_score)
        if len(peak_scores_sorted) > 2:
            peak_scores_sorted = peak_scores_sorted[-2:]

        z_peaks = peak_z[peak_scores_sorted]

        orig_df = df.copy()
        if len(peak_scores_sorted) == 2:
            sep = abs(np.diff(z_peaks)[0])
            z_between_peaks = np.linspace(min(z_peaks), max(z_peaks), 5000)
            scores = kde(z_between_peaks)
            density_cutoff = min(scores) * 1.05
            df['density'] = kde(df['z [nm]'].to_numpy())

            x = [df['z [nm]'].min(), df['z [nm]'].max()]
            y = [density_cutoff, density_cutoff]
            ax4.plot(x, y, 'r--', label='min density')
            ax4.set_title(f'Sep: {round(sep, 2)}')

            df = df[df['density']>=density_cutoff]
        else:
            sep = 0

        # Plot peaks
        for peak in z_peaks:
            x = [peak, peak]
            y = [0, kde(peak).squeeze()]
            ax4.plot(x, y, 'r--')
            
        if df.shape[0] == 0:
            plt.close()
            continue
        
        ax1 = fig.add_subplot(gs[0, 0])
        render_locs(orig_df, args, (0,0,0), barsize=110, ax=ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        render_locs(df, args, (np.pi/2,0,0), barsize=50, ax=ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        render_locs(df, args, (0, np.pi/2,0), barsize=50, ax=ax3)

        # if color_by_depth:
        #     color_histplot(histplot, cmap_min_z, cmap_max_z)
        # sns.kdeplot(data=df, x='z [nm]', ax=ax4, bw_adjust=0.5, color='black', bw_method='silverman')

        # x = ax4.lines[0].get_xdata()
        # y = ax4.lines[0].get_ydata()
        # peaks, _ = find_peaks(y)

        # sorted_peaks = sorted(peaks, key=lambda peak_index: y[peak_index], reverse=True)
        # peak_vals = y[peaks]
        # if len(peak_vals) == 1:
        #     n_peaks = 1
        # else:
        #     n_peaks = 2
            
        # sorted_peaks = sorted_peaks[:n_peaks]

        # peak_x = x[sorted_peaks]
        # peak_y = y[sorted_peaks]
        # for x, y in zip(peak_x, peak_y):
        #     ax4.vlines(x, 0, y, label=str(round(x)), color='black')

        septxt = 'Sep: '+ str(round(sep))+ 'nm'

        records.append({
            'id': cid,
            'seperation': sep,
        })

        margin=10
        if 50-margin <= sep and sep <= 50+margin:
            cluster_outdir = good_dir
            print(' Good')
            good_nup += 1
        else:
            cluster_outdir = other_dir
            print(' Bad')
            bad_nup += 1
        plt.suptitle(f'Nup ID: {cid}, N points: {df.shape[0]}, {septxt}')
        imname = f'nup_{cid}_{BLUR}.png'
        outpath = os.path.join(cluster_outdir, imname)
        plt.savefig(outpath)
        plt.close()
        if not args['no_wandb']:
            wandb.log({imname: wandb.Image(outpath)})


    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(args['outdir'], 'nup_report.csv'))
    if not args['no_wandb']:
        wandb.log({'mean_nup_sep_mean': np.mean(df['seperation'])})
        wandb.log({'nup_sep_std': np.std(df['seperation'])})
        wandb.log({'nup_good': good_nup})
        wandb.log({'nup_bad': bad_nup})


def prep_dirs(args):

    shutil.rmtree(args['outdir'], ignore_errors=True)
    os.makedirs(args['outdir'], exist_ok=True)

    good_dir = os.path.join(args['outdir'], 'good_results')
    other_dir = os.path.join(args['outdir'], 'other_results')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    return good_dir, other_dir


def load_and_pick_locs(args):
    locs, info = io.load_locs(args['locs'])
    locs = pd.DataFrame.from_records(locs)
    try:
        assert info[1]['Pixelsize'] == args['pixel_size']
    except AssertionError:
        print('Pixel size mismatch', info[1]['Pixelsize'],  args['pixel_size'])
        quit(1)

    if args['picked_locs']:
        picked_locs, old_info = io.load_locs(args['picked_locs'])
        picked_locs = pd.DataFrame.from_records(picked_locs)
        locs = locs.merge(picked_locs, on=['x', 'y', 'photons', 'bg', 'lpx', 'lpy', 'net_gradient', 'iterations', 'frame', 'likelihood', 'sx', 'sy'])
    locs['clusterID'] = locs['group']
    if 'z' not in list(locs):
        locs['z'] = locs['z [nm]'] / args['pixel_size']
    if 'z [nm]' not in list(locs):
        locs['z [nm]'] = locs['z'] * args['pixel_size']
    return locs


def main(args):
    good_dir, other_dir = prep_dirs(args)
    # Save a copy of this script for reproducibility
    shutil.copy(os.path.abspath(__file__), os.path.join(args['outdir'], 'render_nup.py.bak'))

    locs = load_and_pick_locs(args)

    write_nup_plots(locs, args, good_dir, other_dir)
    print('Wrote dirs:')
    print(f'\t- {os.path.abspath(good_dir)}')
    print(f'\t- {os.path.abspath(other_dir)}')

    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--locs', default='./locs_3d.hdf')
    parser.add_argument('-px', '--pixel-size', default=86, type=int)
    parser.add_argument('-p', '--picked-locs')
    parser.add_argument('-o', '--outdir', default='./nup_renders3')
    parser.add_argument('-os', '--oversample', default=30, type=int)
    parser.add_argument('-mb', '--min-blur', default=0.001)
    parser.add_argument('-b', '--blur-method', default='gaussian')
    parser.add_argument('-df', '--disable-filter', action='store_true')
    parser.add_argument('-k', '--kde-factor', default=0.75, type=float)
    parser.add_argument('--no-wandb', action='store_true')
    return vars(parser.parse_args())

def find_matching_runs(hyper_params):
    print(hyper_params.keys())
    runs = wandb.Api().runs('smlm_z3')
    print(f"Matching run with... {hyper_params['norm']},{hyper_params['dataset']}, {hyper_params['aug_gauss']}, {hyper_params['aug_brightness']}, {hyper_params['learning_rate']}, {hyper_params['batch_size']}")
    try:
        while True:

            run = runs.next()
            rc = run.config

            try:
                config_match = [
                    str(rc['norm']) == str(hyper_params['norm']),
                    str(rc['dataset']) == str(hyper_params['dataset']),
                    float(rc['aug_gauss']) == float(hyper_params['aug_gauss']),
                    float(rc['aug_brightness']) == float(hyper_params['aug_brightness']),
                    float(rc['learning_rate']) == float(hyper_params['learning_rate']),
                    float(rc['batch_size']) == float(hyper_params['batch_size']),

                ]
            except KeyError:
                config_match = [False]
            if all(config_match):
                return run.id
    except StopIteration:
        print('Failed to find match...')
        quit(1)
        return None



def init_model_run(args):
    model_dir = os.path.abspath(os.path.join(args['outdir'], os.pardir, os.pardir))
    report_path = os.path.join(model_dir, 'results', 'report.json')

    with open(report_path) as f:
        report_data = json.load(f)

        
    if report_data.get('wandb_run_id'):
        run_id = report_data['wandb_run_id']
    else:
        hyper_params = {k: v for k, v in report_data['args'].items() if k in ['norm', 'dataset', 'aug_gauss', 'aug_ratio', 'batch_size', 'learning_rate', 'aug_brightness']}
        run_id = find_matching_runs(hyper_params)

    if run_id is None:
        wandb.init(project='smlm_z3')
    else:
        wandb.init(project='smlm_z3', id=run_id, resume=True)

    wandb.run.log_code(".")

    try:
        nup_z_hist_path = os.path.abspath(os.path.join(args['outdir'], os.pardir, 'z_histplot.png'))
        wandb.log({'nup_z_hist': wandb.Image(nup_z_hist_path)})
    except Exception as e:
        print(e)
        




if __name__=='__main__':
    args = parse_args()
    if not args['no_wandb']:
        init_model_run(args)

    main(args)