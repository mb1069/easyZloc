import networkx as nx
from scipy.spatial import Delaunay
from functools import lru_cache
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from skimage.exposure import match_histograms
from sklearn.metrics.pairwise import euclidean_distances
from itertools import product
from functools import partial
from keras.losses import MeanSquaredError
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import trange
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
from data.estimate_offset import get_peak_sharpness
from data.visualise import grid_psfs, show_psf_axial
from skimage.filters import gaussian

import numpy as np

import itertools

class GraphPSFAligner:
    def __init__(self, df, psfs, n_paths=5):
        self.df = df
        self.coords = df[['x', 'y']].to_numpy()
        self.seed_idx = self.find_seed_psf(self.coords)
        print(f'Seed PSF: {self.seed_idx}')
        self.psfs = psfs.copy()
        
        self.dists = euclidean_distances(self.coords)
        self.n_paths = n_paths
        
        self.G = self.delaunay(self.coords)

    def delaunay(self, coords):
        dt = Delaunay(points=coords)
        G = nx.Graph()
        for path in dt.simplices:
            nx.add_path(G, path)

        for src, target in G.edges:
            G[src][target]['weight'] = self.dists[src][target]
        pos = nx.circular_layout(G)
        n_points = self.coords.shape[0]
        for i in range(n_points):
            pos[i] = coords[i]
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen')
        return G

    def find_seed_psf(self, coords):
        # Seed PSF - most centered PSF in FOV
        center = coords.mean(axis=0)
        dists = euclidean_distances([center], coords).squeeze()
        first_point = np.argmin(dists)
        return first_point

    def find_offset(self, target_idx, debug=False):
        paths = list(itertools.islice(nx.shortest_simple_paths(self.G, target_idx, self.seed_idx, weight='weight'), self.n_paths))
        offsets = []
        for path in paths:
            path_dist = []
            for i in range(len(path)-1):
                path_dist.append(self.pairwise_offset(path[i], path[i+1]))
            offsets.append(sum(path_dist))
        
        med, var = np.median(offsets), np.var(offsets)
        if debug:
            true_val = self.df['roll'][target_idx] - self.df['roll'][self.seed_idx]
            print('Pred: ', offsets)
            print('Median, var: ', med, var)
            print('Mean, var: ', np.mean(offsets))
            print('True: ', true_val)
        
        return med, var
    
    def pairwise_offset(self, i, i2):
        raise NotImplementedError

    def align_all(self):
        res = []
        for i in trange(self.df.shape[0]):
            res.append(self.find_offset(i)[0])
        return np.array(res)

mse = MeanSquaredError()

DEBUG = False
UPSCALE_RATIO = 10


def norm_zero_one(s):
    max_s = s.max()
    min_s = s.min()
    return (s - min_s) / (max_s - min_s)

def pad_and_fit_spline(coords, psf, z, z_ups):
    x, y = coords
    zs = psf[:, x, y]
    cs = UnivariateSpline(z, zs, k=3, s=1e-1)
    # if max(cs(z_ups)) > 2:
    #     plt.plot(z, zs, label='raw')
    #     plt.plot(z_ups, cs(z_ups), '.', label='smooth')
    #     plt.legend()
    #     plt.show()
    return x, y, cs(z_ups)
    
def upsample_psf(psf, ratio=UPSCALE_RATIO):
    pad_width = 10
    z = np.arange(-pad_width, psf.shape[0] + pad_width)
    z_ups = np.arange(0, psf.shape[0], 1/ratio)
    upsampled_psf = np.zeros((z_ups.shape[0], *psf.shape[1:]))
    
    psf = np.pad(psf, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
    xys = list(product(np.arange(psf.shape[1]), np.arange(psf.shape[2])))
    func = partial(pad_and_fit_spline, psf=psf, z=z, z_ups=z_ups)
    res = list(map(func, xys))
    # with Pool(8) as p:
    #     res = list(p.imap(func, xys))
    for x, y, z_col in res:
        upsampled_psf[:, x, y] = z_col

    return upsampled_psf

def pad_psf(psf):
    return np.pad(psf, ((20, 20), (0, 0), (0, 0)), mode='edge')

def plot_correction(target, img, best_i, errors):

    psf_corrected = np.roll(img, int(best_i), axis=0)
    plt.plot(target.max(axis=(1,2)), label='target')
    plt.plot(img.max(axis=(1,2)),  label='original')
    plt.plot(psf_corrected.max(axis=(1,2)), label='corrected', )

    plt.legend()
    plt.show()

def tf_find_optimal_roll(target, img, upscale_ratio=UPSCALE_RATIO):
    ref_tf = tf.convert_to_tensor(target)
    img_tf = tf.convert_to_tensor(img)
    errors = []

    for i in range(img.shape[0]):
        error = tf.reduce_mean(mse(ref_tf, img_tf))
        errors.append(error)
        img_tf = tf.roll(img_tf, 1, axis=0)

    best_i = tf.argmin(errors).numpy()
    # Prefer small backwards roll to large forwards roll
    if abs(best_i - img.shape[0]) < best_i:
        best_i = best_i - img.shape[0]

    if DEBUG:
        plot_correction(target, img, best_i, errors)

    return best_i/upscale_ratio


def find_seed_psf(df):
    # Seed PSF - most centered PSF in FOV
    center = df[['x', 'y']].mean(axis=0).to_numpy()
    coords = df[['x', 'y']].to_numpy()
    dists = euclidean_distances([center], coords).squeeze()
    first_point = np.argmin(dists)
    return first_point

import numpy as np
import scipy.optimize as opt
import skimage.filters as filters
from tqdm import trange


def measure_psf_fwhm(psf):
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))
    # Normalize the PSF to range [0, 1]
    psf_norm = (psf - np.min(psf)) / (np.max(psf) - np.min(psf))
    
    # Find the center of the PSF using the maximum intensity
    center = np.unravel_index(np.argmax(psf_norm), psf_norm.shape)
    # Extract a 1D slice of the PSF along the z-axis passing through the center
    z_slice = psf_norm[:, center[0]]
    
    # Estimate the initial parameters of the Gaussian fit
    amplitude = np.max(z_slice) - np.min(z_slice)
    mean = center[0]
    stddev = 2
    
    # Fit the Gaussian to the 1D slice using least squares optimization
    popt, _ = opt.curve_fit(gaussian, np.arange(z_slice.size), z_slice, p0=[amplitude, mean, stddev])
    
    # Compute the FWHM of the Gaussian fit
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
    
    return fwhm

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2).astype(int)

    mask = dist_from_center <= radius
    return mask

def mask_img_stack(stack, radius):
    mask = create_circular_mask(stack.shape[1], stack.shape[2], radius=radius)
    for i in range(stack.shape[0]):
        stack[i][~mask] = 0
    return stack

def prepare_psf(psf):
    psf = gaussian(psf, sigma=1)
    psf = norm_zero_one(psf.copy())
    psf = pad_psf(psf)
    psf = upsample_psf(psf)
    psf = mask_img_stack(psf, 12)
    return psf


from skimage.exposure import match_histograms


class ClassicAligner(GraphPSFAligner):
    
    @lru_cache(maxsize=None)
    def pairwise_offset(self, i, i2):
        psf1 = prepare_psf(self.psfs[i])
        psf2 = prepare_psf(self.psfs[i2])
        psf1 = match_histograms(psf1, psf2)
        return -tf_find_optimal_roll(psf2, psf1)