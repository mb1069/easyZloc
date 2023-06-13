import numpy as np
from skimage.exposure import match_histograms
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from multiprocessing import Pool
from itertools import product
from functools import partial
import tqdm
from keras.losses import MeanSquaredError
from multiprocessing.spawn import prepare
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import trange
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
from data.estimate_offset import get_peak_sharpness
from data.visualise import grid_psfs, show_psf_axial


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

def plot_correction(target, img, psf_corrected, errors):
    if DEBUG:
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

    psf_corrected = np.roll(img, int(best_i), axis=0)
    plot_correction(target, img, psf_corrected, errors)

    return best_i/upscale_ratio

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

from skimage.filters import gaussian

def prepare_psf(psf):
    psf = gaussian(psf, sigma=1)
    psf = norm_zero_one(psf.copy())
    psf = pad_psf(psf)
    psf = upsample_psf(psf)
    # psf = mask_img_stack(psf, 12)
    return psf

def find_seed_psf(df):
    # Seed PSF - most centered PSF in FOV
    center = df[['x', 'y']].mean(axis=0).to_numpy()
    coords = df[['x', 'y']].to_numpy()
    dists = euclidean_distances([center], coords).squeeze()
    first_point = np.argmin(dists)
    return first_point

def get_or_prepare_psf(prepped_psfs, raw_psfs, idx):
    if idx not in prepped_psfs:
        prepped_psfs[idx] = prepare_psf(raw_psfs[idx])
    return prepped_psfs[idx]

import numpy as np
import scipy.optimize as opt
import skimage.filters as filters



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
    try:
        popt, _ = opt.curve_fit(gaussian, np.arange(z_slice.size), z_slice, p0=[amplitude, mean, stddev])
    except RuntimeError:
        return np.inf
    # Compute the FWHM of the Gaussian fit
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
    
    return fwhm

def determine_best_focus_slice(psf):
    # Measure the FWHM of the PSF for each z-slice
    fwhm_values = []
    for i in range(psf.shape[0]):
        fwhm = measure_psf_fwhm(psf[i])
        fwhm_values.append(fwhm)
    
    # Find the index of the z-slice with the minimum FWHM value
    best_slice_idx = np.argmin(fwhm_values)
    return best_slice_idx


def fwhm_offsets(psfs):
    idxs = np.array([determine_best_focus_slice(psf) for psf in psfs]).astype(float)
    idxs -= np.mean(idxs)
    return idxs


def classic_align_psfs(psfs, df):
    seed_psf = find_seed_psf(df)
    ref_psf = prepare_psf(psfs[seed_psf])
    offsets = np.zeros((psfs.shape[0]))

    ref_0 = get_peak_sharpness(psfs[seed_psf])

    for i in trange(1, psfs.shape[0]):
        psf = psfs[i]
        psf = prepare_psf(psf)
        psf = match_histograms(psf, ref_psf)
        offsets[i] = -tf_find_optimal_roll(ref_psf, psf)
        if DEBUG:
            offset_psf = np.roll(psf, shift=-int(offsets[i]), axis=0)
            imgs = np.concatenate((ref_psf, offset_psf), axis=2)
            show_psf_axial(imgs, subsample_n=30)

    offsets -= ref_0

    return offsets