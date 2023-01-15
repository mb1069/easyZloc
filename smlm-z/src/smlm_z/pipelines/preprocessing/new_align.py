import tensorflow as tf
import numpy as np
import networkx as nx

from .align_psfs import UPSCALE_RATIO
from keras.losses import MeanSquaredError
from skimage.filters import gaussian
from skimage.exposure import match_histograms
from itertools import product
from functools import partial
from scipy.interpolate import UnivariateSpline
from tqdm import trange

UPSCALE_RATIO = 10
def norm_zero_one(img):
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max - img_min)


def pad_and_fit_spline(coords, psf, z, z_ups):
    x, y = coords
    zs = psf[:, x, y]
    cs = UnivariateSpline(z, zs, k=1, s=1e-4)
    if False:
        plt.scatter(z, zs, label='raw')
        plt.plot(z_ups, cs(z_ups), label='smooth')
        plt.legend()
        plt.show()
    return x, y, cs(z_ups)
    
def upsample_psf(psf, ratio=UPSCALE_RATIO):
    pad_width = 0
    z = np.arange(-pad_width, psf.shape[0] + pad_width)
    z_ups = np.arange(0, psf.shape[0], 1/ratio)
    upsampled_psf = np.zeros((z_ups.shape[0], *psf.shape[1:]))
    
    psf = np.pad(psf, ((pad_width, pad_width), (0, 0), (0, 0)), mode='median')
    xys = list(product(np.arange(psf.shape[1]), np.arange(psf.shape[2])))
    func = partial(pad_and_fit_spline, psf=psf, z=z, z_ups=z_ups)
    res = list(map(func, xys))
    # with Pool(8) as p:
    #     res = list(p.imap(func, xys))
    for x, y, z_col in res:
        upsampled_psf[:, x, y] = z_col

    return upsampled_psf


def plot_correction(target, img, psf_corrected, errors):
    if False:
        plt.plot(target.max(axis=(1,2)), label='target')
        plt.plot(img.max(axis=(1,2)),  label='original')
        plt.plot(psf_corrected.max(axis=(1,2)), label='corrected', )
        plt.legend()
        plt.show()

        
mse = MeanSquaredError(reduction='sum')

def loss_func(true_m, pred_m):
    m = tf.math.abs(true_m-pred_m)
    m = tf.math.square(m*pred_m)
    return tf.math.reduce_mean(m)

def tf_find_optimal_roll(target, img, upscale_ratio=UPSCALE_RATIO):
    ref_tf = tf.convert_to_tensor(target)
    img_tf = tf.convert_to_tensor(img)
    errors = []

    for i in range(img.shape[0]):
        error = loss_func(ref_tf, img_tf)
#         error = mse(ref_tf, img_tf)
        errors.append(error)
        img_tf = tf.roll(img_tf, 1, axis=0)

    best_i = tf.argmin(errors).numpy()
    # Prefer small backwards roll to large forwards roll
    if abs(best_i - img.shape[0]) < best_i:
        best_i = best_i - img.shape[0]

    psf_corrected = np.roll(img, int(best_i), axis=0)
    plot_correction(target, img, psf_corrected, errors)

    return best_i/upscale_ratio


def prepare_psf(psf):
    psf = psf**2
    psf = gaussian(psf, sigma=1)
    psf = norm_zero_one(psf.copy())
    psf = upsample_psf(psf)
    # psf = mask_img_stack(psf, 12)
    return psf

def align_psfs(psf, psf2):
    psf = prepare_psf(psf)
    psf2 = prepare_psf(psf2)
    psf = match_histograms(psf, psf2)
    offset = tf_find_optimal_roll(psf, psf2)
    return offset


from sklearn.metrics.pairwise import euclidean_distances

def get_center_point(df):
    center_point = [[0, 0]]
    df['dists'] = euclidean_distances(df[['x', 'y']], center_point)
    idx = df['dists'].idxmin()
    return df.iloc[idx]

def get_graph(df):
    dists = euclidean_distances(df[['x', 'y']])

    G = nx.from_numpy_matrix(dists)
    G = nx.minimum_spanning_tree(G)
    return G

def get_offsets(df, psfs):
    offsets = np.zeros((df.shape[0], df.shape[0]))
    offsets[:] = None

    center_point = get_center_point(df)
    G = get_graph(df)
    target_node = center_point.name
    all_offsets = []
    def get_path_offset(G, src_node, target_node):
        spath = nx.shortest_path(G, src_node, target_node)
        if not np.isnan(offsets[src_node, target_node]):
            cumul = offsets[src_node, target_node]
        else:
            cumul = 0
            for i in range(0, len(spath)-1):
                a, b = spath[i], spath[i+1]
                if not np.isnan(offsets[a, b]):
                    offset = offsets[a, b]
                else:
                    offset = align_psfs(psfs[a], psfs[b])
                    offsets[a, b] = -offset
                    offsets[b, a] = offset
                cumul += offset
            offsets[src_node, target_node] = -cumul
            offsets[target_node, src_node] = cumul
        all_offsets.append(cumul)
        
    for i in trange(0, df.shape[0]):
        if i == target_node:
            all_offsets.append(0)
            continue
        get_path_offset(G, i, target_node)

    all_offsets = np.array(all_offsets)

    import sys
    sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
    from data.estimate_offset import get_peak_sharpness
    ref_0 = get_peak_sharpness(psfs[target_node])
    return all_offsets  - ref_0
