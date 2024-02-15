from argparse import ArgumentParser
import os
from glob import glob
from natsort import natsorted
import subprocess
import h5py
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
from sklearn.metrics import euclidean_distances
from tifffile import imread, imwrite
from skimage.feature import match_template
from skimage.filters import butterworth
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
import json
import shutil
from util.util import grid_psfs

def norm_zero_one(s):
    max_s = s.max()
    min_s = s.min()
    return (s - min_s) / (max_s - min_s)

def validate_args(args):
    args['bead_stacks'] = [b for b in args['bead_stacks'] if 'ignored' not in b]
    n_stacks = len(args['bead_stacks'])
    print(f"Found {n_stacks} bead stacks")
    if n_stacks == 0:
        quit(1)
    for f in natsorted(args['bead_stacks']):
        print(f'\t - {f}')


def test_picasso_exec():
    res = subprocess.run(['picasso', '-h'], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        print('\n')
        raise EnvironmentError('Picasso not found/working (see above)')

def transform_args(args):
    fnames = glob(f"{args['bead_stacks']}/**/*.tif", recursive=True)

    args['outpath'] = args['bead_stacks']
    fnames = [os.path.abspath(f) for f in fnames if '_slice.ome.tif' not in f and os.path.basename(f) != 'stacks.ome.tif']
    args['bead_stacks'] = fnames

    args['gaussian_blur'] = list(map(int, args['gaussian_blur'].split(',')))
    return args

def get_or_create_slice(bead_stack, slice_path):
    if not os.path.exists(slice_path):
        im_slice = bead_stack[bead_stack.shape[0]//2]
        # plt.imshow(im_slice)
        # plt.show()
        imwrite(slice_path, im_slice.astype(np.uint16))
    return slice_path

def get_or_create_locs(slice_path, args):
    spots_path = slice_path.replace('.ome.tif', '.ome_spots.hdf5')
    locs_path = slice_path.replace('.ome.tif', '.ome_locs.hdf5')

    if not os.path.exists(spots_path) or not os.path.exists(locs_path) or args['regen']:
        cmd = ['picasso', 'localize', slice_path, '-b', args['box_size_length'], '-g', args['gradient'], '-px', args['pixel_size']]
        print(f'Running {" ".join(list(map(str, cmd)))}')
        for extra_arg in ['qe', 'sensitivity', 'gain', 'baseline', 'fit-method']:
            if extra_arg in args and args[extra_arg]:
                cmd.extend([f'-{extra_arg}', args[extra_arg]])
        cmd = ' '.join(list(map(str, cmd)))
        tqdm.write('Running picasso...', end='')
        res = subprocess.run(cmd, capture_output=True, shell=True, text=True)
        if res.returncode != 0:
            print('Picasso error occured')
            print(res.stdout)
            print(res.stderr)
            return
        tqdm.write('finished!')

    with h5py.File(spots_path) as f:
        spots = np.array(f['spots'])

    locs = pd.read_hdf(locs_path, key='locs')
    locs['fname'] = '___'.join(slice_path.split('/')[-3:])
    print(f'Found {locs.shape[0]} beads')
    return locs, spots

def remove_colocal_locs(locs, spots, args):
    tqdm.write('Removing overlapping beads...')
    coords = locs[['x', 'y']].to_numpy()
    dists = euclidean_distances(coords, coords)
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)
    
    error_margin = 0.8
    min_seperation = (np.sqrt(2)  * args['box_size_length']) * error_margin
    idx = np.argwhere(min_dists > min_seperation).squeeze()
    locs = locs.iloc[idx]
    spots = spots[idx]

    return locs, spots


def extract_training_stacks(spots, bead_stack, args) -> np.array:
    spot_size = args['box_size_length']
    frame_idx = bead_stack.shape[0]//2
    frame = bead_stack[frame_idx]
    stacks = []
    for spot in spots:
        res = match_template(frame, spot)
        i, j = np.unravel_index(np.argmax(res), res.shape)
        stack = bead_stack[:, i:i+spot_size, j:j+spot_size]
        stacks.append(stack)
    return np.array(stacks)

def snr(psf):
    return psf.max() / np.median(psf)


def has_fwhm(psf, args):
    psf = butterworth(psf, cutoff_frequency_ratio=0.2, high_pass=False)
    y = np.max(gaussian(psf), axis=(1,2))
    max_val = np.max(y)
    min_val = np.min(y)
    half_max = min_val + ((max_val-min_val) / 2)
    crossCount = np.sum((y[:-1]>half_max) != (y[1:]>half_max))
    # if args['debug'] and crossCount < 2:
    #     plt.plot(y, label='raw')
    #     plt.plot([0, len(y)], [half_max, half_max])
    #     plt.show()
    # if not (crossCount >= 2):
    #     fig = plt.figure(layout="constrained", figsize=(20, 15), dpi=64)
    #     gs = plt.GridSpec(1, 2, figure=fig)
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     ax2 = fig.add_subplot(gs[0, 1])

    #     ax1.imshow(grid_psfs(psf, cols=20))
    #     ax2.plot(psf)
    #     plt.show()
    return crossCount >= 2


def filter_mse_zprofile(psf, args, i):
    z_step = args['zstep']

    # Define the skewed Gaussian function
    def skewed_gaussian(x, A, x0, sigma, alpha, offset):
        """
        A: Amplitude
        x0: Center
        sigma: Standard Deviation
        alpha: Skewness parameter
        offset: Vertical offset
        """
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) * (1 + erf(alpha * (x - x0))) + offset


    # Fit the skewed Gaussian to the data
    x_data = np.arange(psf.shape[0]) * z_step
    y_data = psf.max(axis=(1,2))
    y_data = norm_zero_one(y_data)
    initial_guess = [1, psf.shape[0] * z_step / 2, psf.shape[0] * z_step/4, 0.0, np.median(y_data)]

    bounds = [
        (0.6, 1.2),
        (psf.shape[0] * z_step/8, psf.shape[0] * z_step),
        (psf.shape[0] * z_step/20, psf.shape[0] * z_step/4),
        (-np.inf, np.inf),
        (y_data.min(), y_data.max())
    ]
    try:
        params, _ = curve_fit(skewed_gaussian, x_data, y_data, p0=initial_guess, bounds=list(zip(*bounds)))
    except RuntimeError:
        print('Failed to find fit')
        params = initial_guess

    y_fit = skewed_gaussian(x_data, *params)

    mse = (y_fit - y_data) ** 2
    avg_mse = np.mean(mse)
    max_mse = np.max(mse)

    permitted_avg_mse = 0.02
    permitted_max_mse = 0.1
    # if (avg_mse < permitted_avg_mse and max_mse < permitted_max_mse):
    #     fig = plt.figure(layout="constrained", figsize=(10, 8), dpi=64)
    #     gs = plt.GridSpec(1, 2, figure=fig)
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     ax2 = fig.add_subplot(gs[0, 1])

    #     ax1.imshow(grid_psfs(psf, cols=20))
    #     print(avg_mse, max_mse)
    #     ax2.plot(x_data, y_data)
    #     ax2.plot(x_data, y_fit)
    #     plt.show()
    return avg_mse < permitted_avg_mse and max_mse < permitted_max_mse
        

def get_sharpness(array):
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness


def reduce_img(psf):
    return np.stack([get_sharpness(x) for x in psf])


def est_bead_offsets(psfs, locs, args):
    UPSCALE_RATIO = 10

    def denoise(img):
        
        sigmas = np.array(args['gaussian_blur'])
        return gaussian_filter(img.copy(), sigma=sigmas)

    def find_peak(psf):
        if psf.ndim == 4:
            psf = psf.mean(axis=-1)
        x = np.arange(psf.shape[0]) * args['zstep']
        psf = denoise(psf)
        
        inten = norm_zero_one(reduce_img(psf))

        cs = UnivariateSpline(x, inten, k=3, s=0.2)

        x_ups = np.linspace(0, psf.shape[0], len(x) * UPSCALE_RATIO) * args['zstep']

        peak_xups = x_ups[np.argmax(cs(x_ups))] 

        return peak_xups
    offsets = np.array(map(find_peak, psfs))

    locs['offset'] = offsets


def filter_mse_xy(stack, max_mse, args):

    # Define a 2D Gaussian function
    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
        return g.ravel()

    sharp = reduce_img(stack)
    idx = np.argmax(sharp)
    image = stack[idx]


    # Load and preprocess the image (e.g., convert to grayscale)
    # For simplicity, let's generate a simple image for demonstration
    image_size = image.shape[1]
    x = np.linspace(0, image_size - 1, image_size)
    y = np.linspace(0, image_size - 1, image_size)
    x, y = np.meshgrid(x, y)
    
    # Fit the Gaussian to the image data
    p0 = [1, image_size / 2, image_size / 2, 5, 5, 0, 0]  # Initial guess for parameters
    bounds = [
        (-np.inf, np.inf),
        (image_size * (2/5), image_size * (3/5)),
        (image_size * (2/5), image_size * (3/5)),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (0, np.inf),
    ]

    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=p0, bounds=list(zip(*bounds)))
    except RuntimeError:
        print('fit failed')
        popt = p0
    render = gaussian_2d((x, y), *popt).reshape(image.shape)

    error = mean_squared_error(render, image)

    # if error > max_mse:
    #     # Visualize the original image and the fitted Gaussian
    #     plt.plot(sharp)
    #     plt.show()
    #     plt.figure(figsize=(5, 3))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image)
    #     plt.title('Original Image')
        
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(render)
    #     plt.title('Fitted Gaussian')
        
    #     plt.tight_layout()
    #     plt.show()
    #     print(error, max_mse, error <= max_mse)

    return error <= max_mse


def filter_beads(spots, locs, stacks, args, rejected_outpath):
    print('Removing poorly imaged beads...', end='')
    # Filter by SNR threshold

    snrs = np.array([snr(psf) > args['min_snr'] for psf in stacks])
    fwhms = np.array([has_fwhm(psf, args) for psf in stacks])
    mse_z = np.array([filter_mse_zprofile(psf, args, i) for i, psf in enumerate(stacks)])
    mse_xy = np.array([filter_mse_xy(psf, 5000, args) for i, psf in enumerate(stacks)])

    # snrs[:] = True
    # mse_filters[:] = True

    idx = np.argwhere(snrs & fwhms & mse_z & mse_xy).squeeze()
    reasons = [''] * len(snrs)
    for i in range(spots.shape[0]):
        if not fwhms[i]:
            reasons[i] += 'fwhm'
        if not snrs[i]:
            reasons[i] += f',snrs({round(snr(stacks[i]), 3)})' 
        if not mse_z[i]:
            reasons[i] += ',mse_z'
        if not mse_xy[i]:
            reasons[i] += ',mse_xy'

    locs['rejected'] = reasons


    est_bead_offsets(stacks, locs, args)

    if args['debug']:
        rejected_idx = np.argwhere(np.invert(snrs & fwhms & mse_z & mse_xy))[:, 0]
        print('\n', 'Rejected: ', rejected_idx)

        if len(rejected_idx):
            print('Writing rejected figures...')

            write_stack_figures(stacks[rejected_idx], locs.iloc[rejected_idx], rejected_outpath)

    spots = spots[idx]
    locs = locs.iloc[idx]
    stacks = stacks[idx]

    print('finished!')

    return spots, locs, stacks


def write_combined_data(stacks, locs, args):

    outpath = os.path.join(args['outpath'], 'combined')
    os.makedirs(outpath, exist_ok=True)

    locs_outpath = os.path.join(outpath, 'locs.hdf')
    stacks_outpath = os.path.join(outpath, 'stacks.ome.tif')

    imwrite(stacks_outpath, stacks)
    locs.to_hdf(locs_outpath, key='locs')

    stacks_config = {
        'zstep': args['zstep'],
        'gen_args': args
    }
    
    stacks_config_outpath = os.path.join(outpath, 'stacks_config.json')
    with open(stacks_config_outpath, 'w') as fp:
        json_dumps_str = json.dumps(stacks_config, indent=4)
        print(json_dumps_str, file=fp)

    print('Saved results to:')
    print(f'\t{locs_outpath}')
    print(f'\t{stacks_outpath}')
    print(f'\t{stacks_config_outpath}')
    print(f'Total beads: {locs.shape[0]}')


def write_stack_figure(i, stacks, locs, outpath, fname):
    stack = stacks[i]
    loc = locs.iloc[i].to_dict()

    fig = plt.figure(layout="constrained", figsize=(20, 15), dpi=64)
    gs = plt.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, 1:])


    fig.suptitle(f'Bead: {i}')

    ax1.imshow(grid_psfs(stack, cols=20))
    ax1.set_title('Ordered by frame')

    intensity = stack.max(axis=(1,2))
    min_val = min(intensity)
    max_val = max(intensity)
    frame_zpos = (np.arange(len(intensity)) * args['zstep']) - loc['offset']
    ax2.plot(frame_zpos, intensity)
    ax2.vlines(0, min_val, max_val, colors='orange')
    ax2.set_title('Max normalised pixel intensity over z')
    ax2.set_xlabel('z (nm)')
    ax2.set_ylabel('pixel intensity')    


    for k, v in loc.items():
        if isinstance(v, float):
            loc[k] = round(v, 5)
    text = json.dumps(loc, indent=4)
    ax3.axis((0, 10, 0, 10))
    ax3.text(0,0, text, fontsize=18, wrap=True)
    outfpath = os.path.join(outpath, f'{fname}_bead_{i}.png')
    plt.savefig(outfpath)
    plt.close()
    print(f'Wrote {outfpath}')


from multiprocessing import Pool
from itertools import repeat


def write_stack_figures(stacks, locs, outpath):
    fname = set(locs['fname']).pop().replace('.ome.tif', '')
    os.makedirs(outpath, exist_ok=True)

    idx = np.arange(stacks.shape[0])
    with Pool(8) as pool:
        res = pool.starmap(write_stack_figure, zip(idx, repeat(stacks), repeat(locs), repeat(outpath), repeat(fname)))

# def filter_by_tmp_locs(locs, spots):
#     original_locs = pd.read_hdf('/home/miguel/Projects/smlm_z/publication/original_locs.hdf', key='locs')
#     x_coords = set(original_locs['x'])
#     idx = np.argwhere([x in x_coords for x in locs['x']]).squeeze()
#     locs = locs.iloc[idx]
#     spots = spots[idx]
#     print(locs.shape)
#     return locs, spots

def main(args):
    all_stacks = []
    all_spots = []
    all_locs = []

    found_beads = 0
    retained_beads = 0

    rejected_outpath = os.path.join(args['outpath'], 'combined', 'rejected')
    shutil.rmtree(rejected_outpath, ignore_errors=True)

    if args['debug']:
        os.makedirs(rejected_outpath, exist_ok=True)

    for bead_stack_path in tqdm(natsorted(args['bead_stacks'])):
        # if 'stack_3_' not in bead_stack_path:
        #     continue
        tqdm.write(f'Preparing {os.path.basename(bead_stack_path)}')

        bead_stack = imread(bead_stack_path)
        slice_path = bead_stack_path.replace('.ome', '_slice.ome')
        slice_path = get_or_create_slice(bead_stack, slice_path)

        raw_locs, spots = get_or_create_locs(slice_path, args)
        
        # raw_locs, spots = filter_by_tmp_locs(raw_locs, spots)
        found_beads += raw_locs.shape[0]
        locs, spots = remove_colocal_locs(raw_locs, spots, args)
        perc_removed = round(100*(1-(locs.shape[0]/raw_locs.shape[0])), 2)
        print(f'Removed {perc_removed}% due to co-location')

        stacks = extract_training_stacks(spots, bead_stack, args)

        spots, locs, stacks = filter_beads(spots, locs, stacks, args, rejected_outpath)
        retained_beads += locs.shape[0]
        tqdm.write(f'Retained {stacks.shape[0]} beads')
        all_stacks.append(stacks)
        all_spots.append(spots)
        all_locs.append(locs)

    print(f'Found {found_beads} total beads')
    min_stack_length = min(list(map(lambda s: s.shape[1], all_stacks)))
    stacks = [s[:, :min_stack_length] for s in all_stacks]
    locs = pd.concat(all_locs)
    stacks = np.concatenate(stacks)

    # original_locs = pd.read_hdf('/home/miguel/Projects/smlm_z/publication/original_locs.hdf', key='locs')
    # x_coords = set(original_locs['x'])
    # print(len(set(locs['x'])), len(set(original_locs['x'])))
    # print(len(set(locs['x']).intersection(set(original_locs['x']))))
    # locs = pd.concat((locs, locs))
    # stacks = np.concatenate((stacks, stacks))

    print(locs.shape)
    print(stacks.shape)
    print(f'Kept {locs.shape[0]} total beads')

    write_combined_data(stacks, locs, args)

    if args['debug']:
        outpath = os.path.join(args['outpath'], 'combined', 'debug')
        write_stack_figures(stacks, locs, outpath)
        



def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('bead_stacks', help='Path to TIFF bead stacks / directory containing bead stacks.')
    parser.add_argument('-z', '--zstep', help='Pixel size (nm)', default=10, type=int)
    parser.add_argument('-px', '--pixel_size', help='Pixel size (nm)', default=86, type=int)
    parser.add_argument('-g', '--gradient', help='Min. net gradient', default=1000, type=int)
    parser.add_argument('-b', '--box-size-length', help='Box size', default=15, type=int)
    parser.add_argument('-qe', '--qe', help='Quantum efficiency', type=float)
    parser.add_argument('-s', '--sensitivity', help='Sensitivity', type=float)
    parser.add_argument('-ga', '--gain', help='Gain', type=float)
    parser.add_argument('-bl', '--baseline', help='Baseline', type=int)
    parser.add_argument('-a', '--fit-method', help='Fit method', choices=['mle', 'lq', 'avg'])
    parser.add_argument('--regen', action='store_true')
    parser.add_argument('-snr', '--min-snr', type=float, default=2.0)
    parser.add_argument('-gb', '--gaussian-blur', default='3,2,2', help='Gaussian pixel-blur in Z/Y/X for bead offset estimation')
    parser.add_argument('--debug', action='store_true')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    test_picasso_exec()
    args = transform_args(args)
    validate_args(args)
    main(args)
