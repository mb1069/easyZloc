import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import numpy as np
from picasso.render import render
from scipy.signal import find_peaks
import shutil
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=14)
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

def is_circular(points, threshold=0.8):
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    # Check if the explained variance ratio is roughly equal for both components
    return np.min(pca.explained_variance_ratio_) > threshold
    

def fit_circle(points, radius):
    def objective(center):
        return np.mean((np.sqrt(np.sum((points - center)**2, axis=1)) - radius)**2)
    
    # Initial guess: mean of the points
    initial_center = np.mean(points, axis=0)
    
    # Optimize to find the best center
    result = minimize(objective, initial_center, method='nelder-mead')
    
    return result.x, objective(result.x)


def check_2d_fit(df):
    if df.shape[0] < 20:
        return False
    radius_px = (110/PIXEL_SIZE)/2
    points = df[['x', 'y']].to_numpy()

    if not is_circular(points, threshold=0.2):
        return False
    
    center, error = fit_circle(points, radius_px)
    print(set(df['group']), error)

    return error < 0.05


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


def render_locs(locs, args, ang_xyz=(0,0,0), barsize=None, ax=None, ylabel=True):
    locs = locs.copy()

    # locs['lpx'] = 0.01
    # locs['sx'] = 0.01
    # locs['lpy'] = 0.01
    # locs['sy'] = 0.01

    if 'lpz' not in list(locs):
        locs['lpz'] = np.mean(locs[['lpx', 'lpy']].to_numpy()) / 2
    if 'sz' not in list(locs):
        locs['sz'] = np.mean(locs[['sx', 'sy']].to_numpy()) / 3

    # locs['sx'] = 0.15
    # locs['sy'] = 0.15
    # locs['sz'] = 0.15
    # locs['lpx'] = 0.05
    # locs['lpy'] = 0.05
    # locs['lpz'] = 0.05
    # print(locs['sx'].min(), locs['sx'].max())
    # print(locs['sy'].min(), locs['sy'].max())
    # print(locs['sz'].min(), locs['sz'].max())
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
        ax.set_xlabel('x [nm]')
        if ylabel:
            ax.set_ylabel('y [nm]')
    elif ang_xyz == (np.pi/2, 0, 0):
        ax.set_xlabel('z [nm]')
        if ylabel:
            ax.set_ylabel('x [nm]')
        img = img.T
        viewport = np.fliplr(viewport)

    elif ang_xyz == (0, np.pi/2, 0):
        ax.set_xlabel('z [nm]')
        if ylabel:
            ax.set_ylabel('y [nm]')
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
    vmax = np.percentile(img.flatten(), 99.9)
    img_plot = ax.imshow(img, extent=extent, vmax=vmax)
    
    # plt.colorbar(img_plot)

    if barsize is not None:
        scalebar = AnchoredSizeBar(ax.transData,
                            barsize, f'{barsize} nm', 'lower center', 
                            pad=0.1,
                            borderpad=0.2,
                            color='white',
                            frameon=False,
                            size_vertical=1,
                            fontproperties=fontprops)
        ax.add_artist(scalebar)

def is_good_reconstruction(kde, peaks, sep):
    # Check seperation between 40 and 60 nm
    if not (40 <= sep and sep <= 60):
        return False
    
    # # Check contrast
    # zs = np.linspace(peaks.min(), peaks.max(), 1000)[:, None]
    # bottom = np.exp(kde.score_samples(zs).min())
    # peak_heights = np.exp(kde.score_samples(peaks[:, None]))
    # if (bottom / np.min(peak_heights)) > 0.8 :
    #     return False
    
    # # Check peaks are approximately equal
    # if ((max(peak_heights) - min(peak_heights)) / np.mean(peak_heights)) > 0.25:
    #     return False
    return True
    

class ResultDir:
    def __init__(self, df_path, label, color):
        self.df = pd.read_hdf(df_path, key='locs')
        if 'index' in list(self.df):
            del self.df['index']
        self.label = label
        self.color = color
        self.group_results = []
        self.xz_render_density = {}

    def scatter_xy(self, group, ax):
        sns.scatterplot(data=self.df[self.df['group']==group], x='x', y='y', label=self.label, ax=ax)

    def plot_hist_z(self, group, ax, plot=True, bandwidth=20):
        sub_df = self.df[self.df['group']==group]
        bins = np.arange(-1000, 1000, 25)
        
        zs = sub_df['z [nm]'].to_numpy()[:, None]

        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(zs)

        # kde = gaussian_kde(bw_method='silverman')
        # kde.set_bandwidth(kde.factor * 0.2)

        zvals = np.linspace(sub_df['z [nm]'].min()-25, sub_df['z [nm]'].max()+25, 5000)[:, None]
        score = np.exp(kde.score_samples(zvals))
        zvals = zvals.squeeze()

        peak = zvals[np.argmax(score)]
        zrange = 400

        peak_idx, _ = find_peaks(score, prominence=0.0001)
        peak_z = zvals[peak_idx]
        peak_score = score[peak_idx]

        peak_scores_sorted = np.argsort(peak_score)

        good_reconstruction = False

        if plot:
            sns.histplot(data=sub_df, 
                        x='z [nm]', 
                        stat='density', 
                        bins=bins, 
                        ax=ax, 
                        alpha=0.25, 
                        color=self.color
                        )
            ax.set_xlim((peak-zrange, peak+zrange))
            ax.plot(zvals, score, label='KDE', color=self.color)

        try:
            if len(peak_scores_sorted) > 2:
                peak_scores_sorted = peak_scores_sorted[-2:]
    
            xs = peak_z[peak_scores_sorted]
            ys = peak_score[peak_scores_sorted]
            sep = abs(xs[0]-xs[1])
            
            good_reconstruction = is_good_reconstruction(kde, xs, sep)
            zs = np.linspace(xs.min(), xs.max(), 1000)[:, None]
            scores_between_peaks = np.exp(kde.score_samples(zs))
            mean_peak = np.min(ys)
            threshold = np.mean([min(scores_between_peaks), mean_peak])
            self.xz_render_density[group] = {'kde': kde, 'threshold': threshold}

            if plot:
                ax.scatter(xs, ys, label=f'sep={str(round(sep, 1))}', marker='x', color='red')
                ax.plot(xs, [threshold, threshold], linestyle='dashed', color=self.color)
                for i in range(len(xs)):
                    ax.plot([xs[i], xs[i]], [threshold, ys[i]], linestyle='dashed', color=self.color)
        except IndexError as e:
            tqdm.write(f'Could not resolve a bimodal structure: {self.label}, {group}')
            pass
        self.group_results.append({
            'group': group,
            'quality': good_reconstruction,
            'method': self.label
        })
        if plot:
            ax.set_title(f'{self.label}, N pts: {sub_df.shape[0]}')
            ax.legend()

    def render_xz(self, group, ax, ylabel):
        sub_df = self.df[self.df['group']==group].copy(deep=True)
        if group in self.xz_render_density:
            threshold = self.xz_render_density[group]['threshold']
            kde = self.xz_render_density[group]['kde']
            
            sub_df['density'] = np.exp(kde.score_samples(sub_df['z [nm]'].to_numpy()[:, None]))

            zs = np.linspace(sub_df['z [nm]'].min(), sub_df['z [nm]'].max(), 1000)
            scores = np.exp(kde.score_samples(zs[:, None]))
            peak = zs[np.argmax(scores)]
            z_range = 100
            sub_df = sub_df[sub_df['z [nm]'] <= peak + z_range]
            sub_df = sub_df[sub_df['z [nm]'] >= peak - z_range]

            sub_df = sub_df[sub_df['density']>=threshold]
        if sub_df.shape[0] == 0:
            return
    
        render_locs(sub_df, args, (np.pi/2,0,0), barsize=50, ax=ax, ylabel=ylabel)

    def render_xy(self, group, ax, ylabel):
        sub_df = self.df[self.df['group']==group].copy(deep=True)
        if sub_df.shape[0] == 0:
            return
        render_locs(sub_df, args, (0,0,0), barsize=110, ax=ax, ylabel=ylabel)

    def compile_df(self):
        return pd.DataFrame.from_records(self.group_results)



BANDWIDTH = 15
PIXEL_SIZE = 106
args = {
    'blur_method': 'gaussian',
    'min_blur': 0,
    'oversample': 30,
    'pixel_size': PIXEL_SIZE
}


if __name__=='__main__':
    
    our_res = ResultDir(
        './easyZloc/nup.hdf5',
        'Ours',
        'blue',
    )

    decode_res = ResultDir(
        './decode/emitter_remapped_undrift_picked_matched.hdf5', 
        'DECODE',
        'green',
    )


    fd_deeploc_res = ResultDir(
        './fd-loco/fd_deeploc_results/fov1_locs_remapped_undrift_picked_matched.hdf5',
        'FD-Deeploc',
        'red',
    )

    ress = [
        our_res,
        decode_res,
        fd_deeploc_res
    ]

    groups = sorted(list(set().union(*[set(res.df['group']) for res in ress])))
    # results_bak = pd.read_csv('./results_bak.csv')
    # results_bak = results_bak[results_bak['circular_fit']]
    # groups = results_bak['group']
    outdir = os.path.abspath('./comparison_plots')

    os.makedirs(outdir, exist_ok=True)
    subdirs = ['good', 'other']
    for sd in subdirs:
        path = os.path.join(outdir, sd)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)



    plot = True

    def gen_plot(g, ress, outdir):
        
        # if not our_res.im_quality[i]:
        #     continue
        if plot:
            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot(3, 3, 1)
            ax2 = fig.add_subplot(3, 3, 2)
            ax3 = fig.add_subplot(3, 3, 3)
            
            ax4 = fig.add_subplot(3, 3, 4)
            ax5 = fig.add_subplot(3, 3, 5, sharey=ax4)
            ax6 = fig.add_subplot(3, 3, 6, sharey=ax4)

            ax7 = fig.add_subplot(3, 3, 7, sharex=ax4)
            ax8 = fig.add_subplot(3, 3, 8, sharey=ax7, sharex=ax5)
            ax9 = fig.add_subplot(3, 3, 9, sharey=ax7, sharex=ax6)
            axs = np.array([
                [ax1, ax2, ax3],
                [ax4, ax5, ax6],
                [ax7, ax8, ax9]
            ])
            # fig, axs = plt.subplots(3, 3, figsize=(10,10), sharey='row')
            # plt.tight_layout()
        else:
            axs = np.zeros((2, 3))
        for res, ax  in zip(ress, axs[0]):
            try:
                res.plot_hist_z(g, ax, plot=plot, bandwidth=BANDWIDTH)
            except ValueError:
                pass

        if plot:
            for i, (res, ax)  in enumerate(zip(ress, axs[1])):
                ylabel = i == 0
                res.render_xz(g, ax, ylabel)
        
        if plot:
            for i, (res, ax)  in enumerate(zip(ress, axs[2])):
                ylabel = i == 0
                res.render_xy(g, ax, ylabel)
        
        dir_idx = int(not our_res.group_results[-1]['quality'])

        fname = f'{g}.png'
        outpath = os.path.join(outdir, subdirs[dir_idx], fname)

        if plot:
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
            plt.savefig(outpath)
            plt.close()
            tqdm.write(outpath)
        


    for i in tqdm(groups):
        gen_plot(i, ress, outdir)

    circular_fit = [check_2d_fit(our_res.df[our_res.df['group']==g]) for g in tqdm(groups)]

    df = pd.concat([res.compile_df() for res in ress])
    df = pd.pivot(df, index='group', columns='method', values='quality')
    df['all_good'] = df['DECODE'] & df['FD-Deeploc'] & (df['Our method'])
    df.fillna(False, inplace=True)
    df['circular_fit'] = circular_fit
    df.to_csv('./results.csv')

    df = pd.read_csv('./results.csv')
    df.reset_index(inplace=True)
    for row in df.to_dict(orient='records'):
        if row['Our method']:
            print(f"comparison_plots/good/{row['group']}.png")

    for col in [r.label for r in ress]:
        print(col, df[col].sum(), df[col].sum() / df.shape[0])

        # fig, ax = plt.subplots(figsize=(14,3))
        # ax.axis('off')
        # ax.imshow(our_res.get_img(i)[0])
        # plt.show()
        #     ax.axis('off')
        #     try:
        #         img, label = res.get_img(i)
        #         ax.imshow(img)
        #         ax.set_title(label)
        #     except KeyError:
        #         pass
        # plt.show()
            
