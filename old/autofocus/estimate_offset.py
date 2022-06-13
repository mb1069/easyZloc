import math

import cv2

import jax.numpy as jnp
from jax.ops import index, index_add, index_update
from jax.scipy.special import erf
from jax.scipy.stats import norm

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from src.config.optics import voxel_sizes, fwhm
from src.data.evaluate import mse
import os
import matplotlib.pyplot as plt

from src.wavelets.wavelet_data.util import min_max_norm
from src.data.visualise import show_psf_axial

est_sigma = (fwhm[0] / voxel_sizes[0]) * 1.5


def resize_img(img):
    height = 256
    width = 256
    # width = int((height / img.shape[1]) * img.shape[2])
    new_img = jnp.zeros((img.shape[0], height, width))
    for i in range(len(img)):
        img_z = img[i]
        val = cv2.resize(img_z, (width, height), interpolation=cv2.INTER_CUBIC)
        new_img = index_update(new_img, index[i, :, :], val)
    return new_img



def pdf(x):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-x ** 2 / 2)


def cdf(x):
    return (1 + erf(x / math.sqrt(2))) / 2


def skew(x, s, e=0, w=1, a=0, c=0):
    t = (x - e) / w
    return (s * norm.pdf(t) * norm.cdf(a * t)) + c


def fit_gaussian(x, y):
    mean = jnp.argmax(y)
    peak = jnp.max(y)
    min_val = jnp.min(y)
    bounds = [
        (0, peak * 10),
        (mean * 0.75, mean * 1.25),
        (0, len(y)),
        (-10, 10),
        (0, jnp.max(y))
    ]
    low_bounds, high_bounds = zip(*bounds)
    popt, pcov = curve_fit(skew, x, y, p0=[peak, mean, est_sigma, 0, min_val], bounds=(low_bounds, high_bounds),
                           maxfev=10000)
    return lambda x: skew(x, *popt)


def estimate_offset(img, voxel_sizes=voxel_sizes, row_avg=True):
    if img.nbytes / (10 ** 9) > 5:
        img = resize_img(img)
    unit_norm = jnp.linalg.norm(img, axis=(1, 2))
    img = jnp.stack([i / n for i, n in zip(img, unit_norm)])

    # background subtraction
    background = jnp.amin(img, axis=0)
    img = img - background

    # Average over rows
    if row_avg:
        axis = 2
    else:
        axis = 1
    img_rows = jnp.mean(img, axis=axis)

    # Apply smoothing over profiles and subtract background again
    smoothed_img_rows = savgol_filter(img_rows, 21, 3, axis=1)
    smoothed_bg = jnp.min(smoothed_img_rows, axis=0)
    smoothed_img_rows -= smoothed_bg

    # FFT and reorder by frequency
    fft_rows = jnp.fft.fft(smoothed_img_rows, axis=1)
    fft_rows = jnp.abs(fft_rows) ** 2
    freqs = jnp.fft.fftfreq(fft_rows.shape[1])
    freq_idx = jnp.argsort(freqs)
    fft_rows = fft_rows[:, freq_idx]

    fft_rows = jnp.std(fft_rows, axis=1)

    # Re-smooth
    fft_rows = savgol_filter(fft_rows, 51, 3)

    x = jnp.linspace(0, fft_rows.shape[0] - 1, num=fft_rows.shape[0])
    defocus = (x.squeeze() - fft_rows.argmin()) * voxel_sizes[0]

    # Uncomment for comparison to Jonathan's code
    # from src.autofocus.jonny_code.CNN_subFunctions import centerStd2
    # _, ref = centerStd2(img, [], img.shape[0])
    # plt.plot(defocus, fft_rows, label='mine')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    # fft_rows = fft_rows / fft_rows.max()
    # ref = ref / ref.max()
    # plt.plot(x, fft_rows, label='mine')
    # plt.plot(x, ref, label='jonny')
    # plt.show()

    return defocus
