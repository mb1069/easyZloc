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
fontprops = fm.FontProperties(size=16)
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from matplotlib.patches import ConnectionPatch

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
    locs = locs

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
    offset = locs['z [nm]'].mean()
    locs['x [nm]'] -= locs['x [nm]'].mean()
    locs['y [nm]'] -= locs['y [nm]'].mean()
    locs['z [nm]'] -= locs['z [nm]'].mean()
    locs['x'] -= locs['x'].mean()
    locs['y'] -= locs['y'].mean()
    locs['z'] -= locs['z'].mean()

    viewport = get_viewport(locs, ('y', 'x'))
    
    _, img = render(locs.to_records(), blur_method=args['blur_method'], viewport=viewport, min_blur_width=args['min_blur'], ang=ang_xyz, oversampling=args['oversample'])

    if ang_xyz == (0, 0, 0):
        ax.set_xlabel('x')
        if ylabel:
            ax.set_ylabel('y')
    elif ang_xyz == (np.pi/2, 0, 0):
        ax.set_xlabel('z')
        if ylabel:
            ax.set_ylabel('x')
        img = img.T
        viewport = np.fliplr(viewport)
    elif ang_xyz == (np.pi/2, 0, np.pi/2):
        ax.set_xlabel('x')
        if ylabel:
            ax.set_ylabel('z')
    elif ang_xyz == (0, np.pi/2, 0):
        ax.set_xlabel('z')
        if ylabel:
            ax.set_ylabel('y')
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
    extent[0] += offset
    extent[1] += offset
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

    def plot_hist_z(self, group, ax, plot=True, bandwidth=20, col=0):
        sub_df = self.df[self.df['group']==group]
        print(self.label, sub_df.shape)
        bins = np.arange(-1000, 1000, 25)
        
        zs = sub_df['z [nm]'].to_numpy()[:, None]

        kde = KernelDensity(bandwidth=bandwidth)
        try:
            kde.fit(zs)
        except ValueError:
            self.group_results.append({
                'group': group,
                'quality': False,
                'method': self.label
            })
            return

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
            ax.set_xlabel('z [nm]')
            if col == 0:
                ax.set_ylabel('density (locs/nm)')
            # sns.histplot(data=sub_df, 
            #             x='z [nm]', 
            #             stat='density', 
            #             bins=bins, 
            #             ax=ax, 
            #             alpha=0.25, 
            #             color=self.color
            #             )
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
                ax.plot([peak-zrange, peak+zrange], [threshold, threshold], linestyle='dashed', color=self.color, label='Dens. thresh.')
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
            ax.legend(loc='lower left')

    def render_xz(self, group, ax, ylabel):
        sub_df = self.df[self.df['group']==group]
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
    
        # render_locs(sub_df, args, (np.pi/2, 0, np.pi/2), barsize=50, ax=ax, ylabel=ylabel)
        render_locs(sub_df, args, (0, np.pi/2, 0), barsize=50, ax=ax, ylabel=ylabel)

    def render_xy(self, group, ax, ylabel):
        sub_df = self.df[self.df['group']==group]
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


if __name__ == '__main__':

    our_res = ResultDir(
        './easyZloc/nup.hdf5',
        'easyZloc',
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
    # groups = list(range(200))
    groups = [g for g in groups if g != 373][338+163:]
    # groups = [4, 38, 62, 89, 52, 65, 153, 140, 220, 399, 452, 592, 667, 779]
    outdir = os.path.abspath('./comparison_plots2')
    # g2 = [10, 1003, 1004, 1006, 1011, 1015, 1019, 1020, 1025, 1026, 1033, 105, 1056, 1059, 1060, 1063, 1064, 1067, 1071, 1077, 1085, 1089, 1094, 110, 1100, 1109, 1112, 112, 1132, 1142, 115, 1151, 1153, 1160, 1167, 1176, 1187, 1196, 1198, 12, 129, 136, 141, 145, 15, 160, 169, 182, 19, 192, 196, 197, 198, 2, 211, 213, 214, 225, 23, 231, 25, 251, 255, 260, 276, 285, 290, 293, 300, 310, 311, 312, 316, 320, 333, 340, 343, 345, 354, 366, 368, 371, 378, 379, 38, 39, 390, 394, 398, 407, 416, 431, 445, 451, 452, 460, 461, 466, 469, 470, 478, 491, 497, 504, 510, 517, 543, 555, 557, 561, 567, 581, 586, 591, 592, 594, 598, 609, 62, 620, 622, 625, 643, 645, 649, 681, 69, 7, 705, 711, 727, 759, 775, 782, 783, 789, 797, 81, 810, 813, 815, 825, 830, 840, 843, 863, 865, 869, 874, 880, 897, 913, 922, 929, 930, 938, 942, 943, 951, 955, 961, 966, 97, 988, 990, 991, 996, 997, 998]

    # groups = [g for g in groups if g not in g2]
    # groups = [373]
    # groups = [7]

    os.makedirs(outdir, exist_ok=True)
    subdirs = [os.path.join(outdir, r.label) for r in ress] + [os.path.join(outdir, 'other')] + [os.path.join(outdir, 'debug')]
    for sd in subdirs:
        path = os.path.join(outdir, sd)
        # shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)



    plot = True

    def gen_plot(g, ress, outdir):
        
        # if not our_res.im_quality[i]:
        #     continue
        fig = plt.figure(figsize=(10, 10))
        for r in ress:
            sns.scatterplot(data=r.df[r.df['group']==g], x='x [nm]', y='y [nm]')
        plt.savefig(f'./comparison_plots2/debug/{g}.png')
        plt.close()

        if plot:
            fig = plt.figure(figsize=(10, 10))
            plt.tight_layout()
            ax1 = fig.add_subplot(3, 3, 1)
            ax2 = fig.add_subplot(3, 3, 2)
            ax3 = fig.add_subplot(3, 3, 3)
            
            ax4 = fig.add_subplot(3, 3, 4)
            ax5 = fig.add_subplot(3, 3, 5)
            ax6 = fig.add_subplot(3, 3, 6)
            # ax4 = fig.add_subplot(3, 3, 4, sharex=ax1)
            # ax5 = fig.add_subplot(3, 3, 5, sharex=ax2)
            # ax6 = fig.add_subplot(3, 3, 6, sharex=ax3)

            ax7 = fig.add_subplot(3, 3, 7)
            ax8 = fig.add_subplot(3, 3, 8)
            ax9 = fig.add_subplot(3, 3, 9)

            axs = np.array([
                [ax1, ax2, ax3],
                [ax4, ax5, ax6],
                [ax7, ax8, ax9]
            ])

            # for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
            #     ax.get_xaxis().set_ticks([])
            #     ax.get_yaxis().set_ticks([])
            # fig, axs = plt.subplots(3, 3, figsize=(10,10), sharey='row')
            # plt.tight_layout()
        else:
            axs = np.zeros((2, 3))
        for i, (res, ax) in enumerate(zip(ress, axs[0])):
            res.plot_hist_z(g, ax, plot=plot, bandwidth=BANDWIDTH, col=i)

        if plot:
            for i, (res, ax)  in enumerate(zip(ress, axs[1])):
                ylabel = i == 0
                res.render_xz(g, ax, ylabel)
        

            for ax_o, ax_t in [(ax1, ax4), (ax2, ax5), (ax3, ax6)]:
                con = ConnectionPatch(xyA=(ax_t.get_xlim()[0], ax_o.get_ylim()[0]), xyB=(ax_t.get_xlim()[0], ax_t.get_ylim()[1]), 
                      coordsA="data", coordsB="data",
                      axesA=ax_o, axesB=ax_t, arrowstyle="-", linestyle='dashed')
                ax_t.add_artist(con)
                con = ConnectionPatch(xyA=(ax_t.get_xlim()[1], ax_o.get_ylim()[0]), xyB=(ax_t.get_xlim()[1], ax_t.get_ylim()[1]), 
                      coordsA="data", coordsB="data",
                      axesA=ax_o, axesB=ax_t, arrowstyle="-", linestyle='dashed')
                ax_t.add_artist(con)
            for i, (res, ax)  in enumerate(zip(ress, axs[2])):
                ylabel = i == 0
                res.render_xy(g, ax, ylabel)
        
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)

            fname = f'{g}.png'
            saved = False
            for r in ress:
                if r.group_results[-1]['quality']:
                    outpath = os.path.join(outdir, r.label, fname)
                    plt.savefig(outpath)
                    tqdm.write(outpath)
                    saved = True
            if not saved:
                outpath = os.path.join(outdir, 'other', fname)
                plt.savefig(outpath)
                tqdm.write(outpath)
            plt.close()
            
        


    for i in tqdm(groups):
        gen_plot(i, ress, outdir)

    circular_fit = [check_2d_fit(our_res.df[our_res.df['group']==g]) for g in tqdm(groups)]

    df = pd.concat([res.compile_df() for res in ress])
    df = pd.pivot(df, index='group', columns='method', values='quality')
    df['all_good'] = df['DECODE'] & df['FD-Deeploc'] & (df['easyZloc'])
    df.fillna(False, inplace=True)
    df['circular_fit'] = circular_fit
    df.to_csv('./results.csv')

    df = pd.read_csv('./results.csv')
    df.reset_index(inplace=True)

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
            
