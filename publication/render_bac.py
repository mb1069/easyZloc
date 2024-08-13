import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import h5py
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

from util.util import read_exp_pixel_size

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
        

def crop_z_view(locs, zrange=1000):
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
    mins[:] = min(mins)
    maxs[:] = max(maxs)
    return np.array([mins[1], maxs[1], mins[0], maxs[0]]) * pixel_size


def render_locs(locs, args, ang_xyz=(0,0,0), barsize=None, ax=None, viewport_margin=1):
    locs = locs.copy()

    # locs['lpx'] = 0.01
    # locs['sx'] = 0.01
    # locs['lpy'] = 0.01
    # locs['sy'] = 0.01

    if 'lpz' not in list(locs):
        locs['lpz'] = np.mean(locs[['lpx', 'lpy']].to_numpy()) / 2
    if 'sz' not in list(locs):
        locs['sz'] = np.mean(locs[['sx', 'sy']].to_numpy()) / 3


    # disable_axis_ticks()
    locs['x [nm]'] -= locs['x [nm]'].mean()
    locs['y [nm]'] -= locs['y [nm]'].mean()
    locs['z [nm]'] -= locs['z [nm]'].mean()
    locs['x'] -= locs['x'].mean()
    locs['y'] -= locs['y'].mean()
    locs['z'] -= locs['z'].mean()

    viewport = get_viewport(locs, ('y', 'x'), margin=viewport_margin)

    # print('\n LOCS', locs.shape)
    # print(viewport)
    # for c in ['x', 'y', 'sx', 'sy', 'sz', 'lpx', 'lpy', 'lpz']:
    #     print(c, locs[c].min(), locs[c].max())
    _, img = render(locs.to_records(), blur_method=args['blur_method'], viewport=viewport, min_blur_width=args['min_blur'], ang=ang_xyz, oversampling=args['oversample'])
    if ang_xyz == (0, 0, 0):
        plt.xlabel('x [nm]')
        plt.ylabel('y [nm]')
    elif ang_xyz == (np.pi/2, 0, 0):
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        # img = img.T
        # viewport = np.fliplr(viewport)

    elif ang_xyz == (0, np.pi/2, 0):
        plt.xlabel('y [nm]')
        plt.ylabel('z [nm]')
        img = img.T
        viewport = np.fliplr(viewport)
    else:
        print('Axis labels uncertain due to rotation angle')

    extent = get_extent(viewport, args['pixel_size'])
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # px_vals = img.flatten()
    # per95 = np.percentile(px_vals, 75)
    # print(img.min(), img.max(), per95)
    # np.save('/home/miguel/Projects/smlm_z/publication/tmp.npy', img)
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

from sklearn.linear_model import LinearRegression

def align_x_axis(df):
    df['x'] -= df['x'].mean()
    df['y'] -= df['y'].mean()

    lr = LinearRegression().fit(df[['x']].to_numpy(), df[['y']].to_numpy())
    gradient = lr.coef_.squeeze()
    theta = -np.arctan(gradient)
    
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    # plt.scatter(df['x'], df['y'])
    x = np.linspace(df['x'].min(), df['x'].max())[:, None]
    

    df[['x', 'y']] = (rot_matrix @ df[['x', 'y']].to_numpy().T).T
    # plt.scatter(df['x'], df['y'], alpha=0.2)
    # plt.plot(x, lr.predict(x))
    # plt.savefig(f'/home/miguel/Projects/smlm_z/publication/tmp_{i}.png')
    # plt.close()
    for c in ['x', 'y']:
        df[f'{c} [nm]'] = df[c] * 106
    return df


def write_nup_plots(locs, args, good_dir, other_dir):
    print('Writing nup plots')
    good_nup = 0
    bad_nup = 0
    locs['bimodal_fit'] = 0

    cols = pd.Series(range(locs.shape[1]), index=locs.columns)
    bimodal_fit_col_idx = cols.reindex(['bimodal_fit'])

    for cid in set(locs['clusterID']):
        # if not cid in [0]:
        #     continue
        print('Cluster ID', cid, end='')
        df = locs[locs['clusterID']==cid]
        # df = align_x_axis(df)
        # df = df[df['sx']>(75/106)]
        # df = df[df['sy']>(75/106)]
        # df = df[df['iterations']<1000]
        print(df.shape)
        # df = df[df['net_gradient'] > 3000]
        # if args['filter_locs']:
        #     df = filter_locs(df).copy()
        df = crop_z_view(df)

        if df.shape[0] < 5:
            print('No remaining localisations, continuing...')
            continue

        try:
            del df['index']
        except ValueError:
            pass
        

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

        peak_idx, _ = find_peaks(score, prominence=0.0001)
        peak_z = zvals[peak_idx]
        peak_score = score[peak_idx]

        peak_scores_sorted = np.argsort(peak_score)
        if len(peak_scores_sorted) > 2:
            peak_scores_sorted = peak_scores_sorted[-2:]

        z_peaks = peak_z[peak_scores_sorted]

        orig_df = df.copy(deep=True)
        
        if len(z_peaks) > 1:
            sep = abs(np.diff(z_peaks)[0])
        else:
            sep = 0

        scores = kde(z_peaks)

        density_cutoff = max(scores) * 0

        df['density'] = kde(df['z [nm]'].to_numpy())
        x = [df['z [nm]'].min(), df['z [nm]'].max()]
        y = [density_cutoff, density_cutoff]
        ax4.plot(x, y, 'r--', label='min density')
        ax4.set_title(f'Sep: {round(sep, 2)}')
        df = df[df['density']>=density_cutoff]
        print(df.shape)

        mean_peak = np.mean(kde(z_peaks))
        prominence = any(scores < 0.9 * mean_peak)

        peak_vals = kde(z_peaks)
        ratio = max(peak_vals) / min(peak_vals)
        equal_ratio = ratio <= 1.5

        # Plot peaks
        for peak in z_peaks:
            x = [peak, peak]
            y = [0, kde(peak).squeeze()]
            ax4.plot(x, y, 'r--')
            
        if df.shape[0] == 0:
            plt.close()
            bad_nup += 1
            continue
        ax1 = fig.add_subplot(gs[0, 0])
        render_locs(orig_df.copy(deep=True), args, (0,0,0), barsize=1000, ax=ax1, viewport_margin=1)
        ax2 = fig.add_subplot(gs[0, 1])
        render_locs(df.copy(deep=True), args, (np.pi/2,0,0), barsize=500, ax=ax2, viewport_margin=1)

        ax3 = fig.add_subplot(gs[0, 2])
        render_locs(df.copy(deep=True), args, (0, np.pi/2,0), barsize=500, ax=ax3, viewport_margin=1)
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

        margin=100
        expected_sep = 500
        nup_good = expected_sep-margin <= sep and sep <= expected_sep+margin and prominence and equal_ratio
        if nup_good:
            cluster_outdir = good_dir
            print(' Good')
            good_nup += 1
            locs.loc[locs['clusterID']==cid, bimodal_fit_col_idx] = 1
            # import h5py
            # outpath = os.path.join('/home/miguel/Projects/smlm_z/publication/VIT_zeiss_green_beads2/out2_1/out_bac_3/nup_renders3/good_results', f'{cid}.hdf5')
            # if 'index' in orig_df:
            #     del orig_df['index']
            # with h5py.File(outpath, 'w') as locs_file:
            #     locs_file.create_dataset("locs", data=orig_df.to_records())
        else:
            reasons = [' Bad']
            if not (50-margin <= sep and sep <= 50+margin):
                reasons.append('sep')
            if not prominence:
                reasons.append('prom.')
            if not equal_ratio:
                reasons.append('eq ratio')
            reasons = ', '.join(reasons)
            cluster_outdir = other_dir
            print(reasons)
            bad_nup += 1

        plt.suptitle(f'Bac ID: {cid}, N points: {orig_df.shape[0]}, {septxt}')
        imname = f'bac_{cid}_{BLUR}.png'
        outpath = os.path.join(cluster_outdir, imname)
        if nup_good:
            print(outpath)
        plt.savefig(outpath)
        plt.close()
        if not args['no_wandb'] and nup_good:
            wandb.log({imname: wandb.Image(outpath)})
        # if cid > 10:
        #     break


    locs['group'] = locs['clusterID']
    del locs['clusterID']
    out_locs_path = os.path.join(args['outdir'], 'nup.hdf5')
    del locs['index']
    with h5py.File(out_locs_path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs.to_records())

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(args['outdir'], 'nup_report.csv'))
    if not args['no_wandb']:
        wandb.log({'mean_nup_sep_mean': np.mean(df['seperation'])})
        wandb.log({'nup_sep_std': np.std(df['seperation'])})
        wandb.log({'nup_good': good_nup})
        wandb.log({'nup_bad': bad_nup})


def prep_dirs(args):

    # shutil.rmtree(args['outdir'], ignore_errors=True)
    os.makedirs(args['outdir'], exist_ok=True)

    good_dir = os.path.join(args['outdir'], 'good_results')
    other_dir = os.path.join(args['outdir'], 'other_results')
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    return good_dir, other_dir


def load_and_pick_locs(args):
    locs, info = io.load_locs(args['locs'])
    locs = pd.DataFrame.from_records(locs)
    if info[1]['Pixelsize'] != args['pixel_size']:
        msg = f"Pixel size mismatch {info[1]['Pixelsize']} {args['pixel_size']}"
        raise RuntimeError(msg)

    if args['picked_locs']:
        picked_locs, old_info = io.load_locs(args['picked_locs'])
        picked_locs = pd.DataFrame.from_records(picked_locs)

        locs = locs.merge(picked_locs, on=['x', 'y', 'photons', 'bg', 'lpx', 'lpy', 'net_gradient', 'iterations', 'frame', 'likelihood', 'sx', 'sy'])
    locs['clusterID'] = locs['group']
    locs['z'] = locs['z [nm]'] / args['pixel_size']
    # if args['picked_locs']:
    #     locs.to_hdf(args['locs'].replace('.hdf5', '_merge_picked.hdf5'), key='locs')
    return locs

def gen_z_histplot(locs, args):
    sns.histplot(locs['z [nm]'])
    outpath = os.path.join(args['outdir'], 'z_histplot2.png')
    plt.savefig(outpath)
    plt.close()
    if not args['no_wandb']:
        wandb.log({'z_histplot': wandb.Image(outpath)})

def main(args):
    good_dir, other_dir = prep_dirs(args)
    # Save a copy of this script for reproducibility
    shutil.copy(os.path.abspath(__file__), os.path.join(args['outdir'], 'render_nup.py.bak'))

    locs = load_and_pick_locs(args)
    
    gen_z_histplot(locs, args)

    write_nup_plots(locs, args, good_dir, other_dir)
    print('Wrote dirs:')
    print(f'\t- {os.path.abspath(good_dir)}')
    print(f'\t- {os.path.abspath(other_dir)}')

    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--locs', default='./locs_3d.hdf')
    # parser.add_argument('-px', '--pixel-size', default=86, type=int)
    parser.add_argument('-p', '--picked-locs')
    parser.add_argument('-o', '--outdir', default='./nup_renders3')
    parser.add_argument('-os', '--oversample', default=10, type=int)
    parser.add_argument('-mb', '--min-blur', default=0.001, type=float)
    parser.add_argument('-b', '--blur-method', default='gaussian')
    parser.add_argument('--filter-locs', action='store_true')
    parser.add_argument('-k', '--kde-factor', default=0.5, type=float)
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
        wandb.init(project='autofocus')
    else:
        wandb.init(project='autofocus', id=run_id, resume=True)

    wandb.run.log_code(".")

    try:
        nup_z_hist_path = os.path.abspath(os.path.join(args['outdir'], os.pardir, 'z_histplot.png'))
        wandb.log({'nup_z_hist': wandb.Image(nup_z_hist_path)})
    except Exception as e:
        print(e)
        

if __name__=='__main__':
    args = parse_args()
    args['pixel_size'] = read_exp_pixel_size(args)
    if not args['no_wandb']:
        init_model_run(args)

    main(args)
