

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
from scipy.signal import find_peaks

from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=14)
from tqdm import tqdm
from gen_comparison_plots import is_good_reconstruction
from sklearn.neighbors import KernelDensity

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

    def calc_hist_z(self, group, bandwidth):
        sub_df = self.df[self.df['group']==group]
        if sub_df.shape[0] == 0:
            return {
                'group': group,
                'quality': 0,
                'method': self.label,
                'bandwidth': bandwidth
            }
        kde = KernelDensity(bandwidth=bandwidth).fit(sub_df['z [nm]'].to_numpy()[:, None])

        # kde = gaussian_kde(sub_df['z [nm]'].to_numpy(), bw_method='silverman')
        # kde.set_bandwidth(kde.factor * kde_factor)

        zvals = np.linspace(sub_df['z [nm]'].min()-25, sub_df['z [nm]'].max()+25, 5000)[:, None]
        score = np.exp(kde.score_samples(zvals))
        zvals = zvals.squeeze()

        peak_idx, _ = find_peaks(score, prominence=0.0001)
        peak_z = zvals[peak_idx]
        peak_score = score[peak_idx]

        peak_scores_sorted = np.argsort(peak_score)

        good_reconstruction = False
        try:
            if len(peak_scores_sorted) > 2:
                peak_scores_sorted = peak_scores_sorted[-2:]
    
            xs = peak_z[peak_scores_sorted]
            ys = peak_score[peak_scores_sorted]
            sep = abs(xs[0]-xs[1])
            good_reconstruction = is_good_reconstruction(kde, xs, sep)

            
            scores_between_peaks = np.exp(kde.score_samples(np.linspace(xs.min(), xs.max(), 1000)[:, None]))
            mean_peak = np.min(ys)
            threshold = np.mean([min(scores_between_peaks), mean_peak])
            self.xz_render_density[group] = {'kde': kde, 'threshold': threshold}

        except IndexError:
            pass

        res = {
            'group': group,
            'quality': int(good_reconstruction),
            'method': self.label,
            'bandwidth': bandwidth
        }
        return res
    def compile_df(self):
        return pd.DataFrame.from_records(self.group_results)


PIXEL_SIZE = 106

dfs = []


# results_bak = pd.read_csv('./results_bak.csv')
# results_bak = results_bak[results_bak['circular_fit']]
# groups = results_bak['group']

# for kde in tqdm([0.25]):
our_res = ResultDir(
    '/home/miguel/Projects/smlm_z/publication/models/zeiss_red_beads/out_24_nvidia6_bak/out_nup_alt_2_pic_updated/nup_renders3/nup.hdf5',
    'Ours',
    'blue',
)

decode_res = ResultDir(
    '/home/miguel/Projects/smlm_z/publication/comparisons/decode/emitter_remapped_undrift_picked_matched.hdf5', 
    'DECODE',
    'green',
)


fd_deeploc_res = ResultDir(
    '/home/miguel/Projects/smlm_z/publication/comparisons/fd-loco/fd_deeploc_results/fov1_locs_remapped_undrift_picked_matched.hdf5',
    'FD-Deeploc',
    'red',
)

ress = [
    our_res,
    decode_res,
    fd_deeploc_res
]

groups = sorted(list(set().union(*[set(res.df['group']) for res in ress])))
# groups = [77, 152, 166]

# kdes = np.linspace(0.1, 0.7, 10)
bandwidths = list(map(int, np.linspace(5, 30, 20)))

def test_kde(args):
    res, kde_factor = args
    return [res.calc_hist_z(g, kde_factor) for g in groups]


from itertools import product
from multiprocessing import Pool

arg_combis = list(product(ress, bandwidths))
with Pool(8) as p:
    r = list(tqdm(p.imap(test_kde, arg_combis), total=len(arg_combis)))

all_r = []
for _r in r:
    all_r += _r
df = pd.DataFrame.from_records(all_r)
# for res in ress:
#     circular_fit = {g: check_2d_fit(res.df[res.df['group']==g]) for g in tqdm(groups)}
#     df[f'{res.label}_is_circular'] = df['group'].map(lambda g: circular_fit[g])



df.to_csv('./kde_results.csv')

df = pd.read_csv('./kde_results.csv')
# df_circular = df[df['is_circular']]
# sns.lineplot(data=df_circular, x='bandwidth', y='quality', hue='method')
# plt.plot([20, 20], [0, df_circular['quality'].max()], '--', label='Selected factor')
# plt.legend()
# plt.ylabel('Valid reconstructions')
# plt.xlabel('KDE Bandwidth (nm)')
# plt.savefig('../figures/kde_circular.png')
# plt.close()

# del df['is_circular']

df = df.groupby(['bandwidth', 'method']).sum()

plt.plot([15, 15], [0, df['quality'].max()], '--', label='Selected bandwidth')
sns.lineplot(data=df.groupby(['bandwidth', 'method']).sum(), x='bandwidth', y='quality', hue='method')
plt.ylabel('Good reconstructions (seperation between [40,60] nm)')
plt.xlabel('KDE Bandwidth')
# plt.title('W/o filtering of groups for circular structure in X/Y ')
plt.savefig('../figures/kde_all.png')
plt.close()