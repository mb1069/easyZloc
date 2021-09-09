import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import erf
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from final_project.smlm_3d.config.optics import fwhm, voxel_sizes
from final_project.smlm_3d.data.visualise import show_psf_axial

est_sigma = (fwhm[0] / voxel_sizes[0]) * 1.5

DEBUG = False


def pdf(x):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-x ** 2 / 2)


def cdf(x):
    return (1 + erf(x / math.sqrt(2))) / 2


def skew(x, s, e=0, w=1, a=0, c=0):
    t = (x - e) / w
    return (s * norm.pdf(t) * norm.cdf(a * t)) + c


def fit_gaussian(x, y):
    mean = np.argmax(y)
    peak = np.max(y)
    min_val = np.min(y)
    bounds = [
        (0, peak * 10),
        (0, len(y)),
        (0, len(y)),
        (-10, 10),
        (0, np.max(y))
    ]
    low_bounds, high_bounds = zip(*bounds)

    p0 = [peak, mean, np.mean(bounds[2]), 0, min_val]
    popt, _ = curve_fit(skew, x, y, p0=p0, bounds=(low_bounds, high_bounds),
                        maxfev=10000)
    return lambda x: skew(x, *popt)


def estimate_offset(psf, voxel_sizes=voxel_sizes):
    target_psf_shape = psf.shape
    _psf = psf.copy()
    # psf = psf.sum(axis=1)

    axial_max = psf.max(axis=(1, 2))
    # axial_max = savgol_filter(axial_max, 7, 3)

    # # Normalise along axial_max
    min_val = axial_max.min()
    max_val = axial_max.max()
    axial_max = (axial_max - min_val) / (max_val - min_val)

    x = np.linspace(0, target_psf_shape[0] - 1, num=target_psf_shape[0])

    fit_failed = False
    gauss_func = fit_gaussian(x, axial_max)
    # Reject points out of range
    if np.argmax(axial_max) == 0:
        fit_failed = True

    if fit_failed or mean_squared_error(axial_max, gauss_func(x)) > 5e-3:
        if DEBUG:
            plt.plot(x, axial_max, label='max')
            plt.plot(x, gauss_func(x), label='fit')
            plt.legend()
            plt.title(f'Failed: {float(mean_squared_error(axial_max, gauss_func(x)))}')
            plt.figure()
            show_psf_axial(_psf / _psf.max(), title=str(mean_squared_error(axial_max, gauss_func(x))))
        raise RuntimeError('Failed to fit gaussian to emitter')
    else:
        # Cover range longer than stack in bead position not in measured range
        high_res_x = np.linspace(0, len(axial_max), 10000, endpoint=True)
        gauss_fit = gauss_func(high_res_x)
        peak = high_res_x[np.argmax(gauss_fit)]

        if DEBUG:
            plt.plot(x, axial_max, label='max')
            plt.plot(high_res_x, gauss_func(high_res_x), label='fit')
            plt.legend()
            plt.title(f'{float(mean_squared_error(axial_max, gauss_func(x)))}')
            plt.show()
            # show_psf_axial((_psf / _psf.max()), title=str(peak))
    z_pos = (x.squeeze() - peak) * voxel_sizes[0]
    return z_pos
