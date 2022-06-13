from dis import dis
import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import erf
from scipy.stats import norm
from csaps import csaps
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.ndimage as ndi


from final_project.smlm_3d.config.optics import fwhm, voxel_sizes
from final_project.smlm_3d.data.visualise import show_psf_axial, concat_psf_axial, grid_psfs

est_sigma = (fwhm[0] / voxel_sizes[0]) * 1.5

DEBUG = False

if DEBUG:
    print('DEBUG enabled in estimate_offset')


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

S = 0.2

from scipy.interpolate import UnivariateSpline

# Image processing methods
from skimage.filters import butterworth
import numpy as np

def norm_zero_one(s):
    max_s = s.max()
    min_s = s.min()
    return (s - min_s) / (max_s - min_s)

def remove_bg(img):
    img = norm_zero_one(img)
    if img.ndim == 3:
        bg_level = img.max(axis=(1,2)).min()
    else:
        bg_level = img.min()
    mult = 1.2
    img = img - (bg_level * mult)
    img[img<0] = 0
    img = norm_zero_one(img)
    return img

def butter_psf(psf):
    psf = norm_zero_one(psf)
    psf = remove_bg(psf)
    psf = np.stack([butterworth(img, 0.05, False) for img in psf])
    psf = norm_zero_one(psf)
    return psf

def get_img_gradient(img):
    gy, gx = np.gradient(img)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def fit_cubic_spline(y, x, sub_x, s=0.05):
    cs = UnivariateSpline(x, y, k=3, s=s)

    # TODO potentially use this to exclude beads?
    # evaluate sum of squared second derivative as a measurement of curve smoothness
    smoothness = sum(np.power(cs(sub_x, 2), 2))
    if smoothness > 1:
        raise EnvironmentError('Failed to fit bead accurately.')

    return norm_zero_one(cs(sub_x))

def get_sharpness(psf):
    return norm_zero_one(np.array([get_img_gradient(img) for img in psf]))

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2).astype(int)

    mask = dist_from_center <= radius
    return mask


def get_peak_sharpness(_psf, s=0.15):
    x = np.arange(0, _psf.shape[0])
    sub_x = np.linspace(0, _psf.shape[0], 1000, endpoint=True)
#     _psf = np.power(_psf, 2)
    psf = remove_bg(_psf)
#     psf = butter_psf(_psf)
    sharpness = get_sharpness(psf)

    cubic_sharpness = fit_cubic_spline(sharpness, x, sub_x, s=s)


    sharpness_peak = sub_x[np.argmax(cubic_sharpness)]

    if DEBUG:
        plt.rcParams['figure.figsize'] = [30, 10] 
        fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
        ax0.plot(x, norm_zero_one(psf.max(axis=(1,2))), label='max')
        ax0.plot(x, norm_zero_one(sharpness), label='sharpness')
        ax0.plot(sub_x, norm_zero_one(cubic_sharpness), label='sharpness_peak')
        plt.xlabel('z slice')
        ax0.legend()
        peak_frame = round(sharpness_peak)
        diff = 5
        imgs = [psf[i] for i in [peak_frame-diff*2, peak_frame-diff, peak_frame, peak_frame+diff, peak_frame+diff*2] if 0 < i and i < _psf.shape[0]]
        imgs = np.concatenate(imgs, axis=0)
        ax1.imshow(imgs)
        plt.title(str(round(sharpness_peak, 3)))
        plt.show()

    return sharpness_peak

# def estimate_offset(psf, voxel_sizes=voxel_sizes, disable_boundary_check=False):
#     peak = get_peak_sharpness(psf.copy(), 0.4)
#     x = np.linspace(0, psf.shape[0]-1, num=psf.shape[0])
#     z_pos = (x.squeeze() -peak) * voxel_sizes[0]
#     return z_pos



def estimate_offset(_psf, voxel_sizes=voxel_sizes, disable_boundary_check=False):
    psf = _psf.copy()
    target_psf_shape = psf.shape
    # psf = psf.sum(axis=2)
    # if psf.dtype != 'uint16':
    #     psf /= psf.max()
    #     psf = psf.astype('uint16')
    #     psf *= 65535
    
    # psf = remove_bg(psf, multiplier=1.5)

    # psf = psf.astype(float) / psf.max()
    axial_max = psf.max(axis=(1, 2))

    # axial_max = savgol_filter(axial_max, 11, 3)

    # # Normalise along axial_max
    min_val = axial_max.min()
    max_val = axial_max.max()
    axial_max = (axial_max - min_val) / (max_val - min_val)

    x = np.linspace(0, target_psf_shape[0]-1, num=target_psf_shape[0])

    sub_x = np.linspace(x.min(), x.max(), 100000, endpoint=True)

    cs = UnivariateSpline(x, axial_max, k=3, s=S)

    fit = cs(x)

    
    diff = fit.max() / fit.mean()
    
    center_img = [d/2 for d in _psf.shape[1:]]
    center_mass = ndi.center_of_mass(_psf[np.argmax(fit)])
    offset = np.hypot(*[center_img[i] - center_mass[i] for i in range(2)])
    # if diff < 1.5 or offset > 1.5:
    #     if DEBUG:
    #         plt.scatter(x, axial_max)
    #         plt.plot(sub_x, cs(sub_x))
    #         plt.show()
    #         print(diff, offset)
    #         show_psf_axial(psf, '', psf.shape[0]//5)
    #     raise RuntimeError(f'Diff is {diff} and offset is {offset}')

    peak = sub_x[np.argmax(cs(sub_x))]
    
    if not disable_boundary_check and not (len(x) * 0.3 < peak and peak < len(x) * 0.7):
        raise RuntimeError
    z_pos = (x.squeeze() -peak) * voxel_sizes[0]

    # if DEBUG:
    #     from matplotlib.pyplot import Slider
    #     fig = plt.figure(constrained_layout=True)
    #     subfigs = fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
    #     axs0 = subfigs[0].subplots(1,1)

    #     axs1 = subfigs[1].subplots(2,1, gridspec_kw={'height_ratios': [10, 1]})
    #     slider = Slider(ax=axs1[1], valmin=0, valmax=2, valinit=S, label='smoothing')

    #     sub_psf = grid_psfs(_psf)
    #     axs0.imshow(sub_psf)
    #     axs0.set_title(str(np.argmin(abs(z_pos))))
    #     axs1[0].scatter(x, axial_max, label='data', marker='x')
    #     line = axs1[0].plot(sub_x, cs(sub_x), linestyle='-', label='fit', color='orange')[0]
    #     axs1[0].set(xlabel='Frame in image stack (Z)', ylabel='Max pixel intensity')
    #     axs1[0].legend()

    #     def update(val):
    #         global S
    #         cs = UnivariateSpline(x, axial_max, k=3, s=val)
    #         line.set_ydata(cs(sub_x))
    #         peak = sub_x[np.argmax((cs(sub_x)))]
    #         axs0.set_title(str(peak))
    #         fig.canvas.draw_idle()
    #         S = val
    #     slider.on_changed(update)

    #     plt.show()
        
    # plt.plot(x, axial_max)
    # plt.plot(x, z_pos / z_pos.max())
    # plt.axvline(x=peak)
    # plt.show()
    return z_pos