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
    print(f"Found {len(args['bead_stacks'])} bead stacks")
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
    fnames = glob(f"{args['bead_stacks']}/**/*.tif")
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
    locs.loc[:, 'fname'] = os.path.basename(slice_path)
    print(f'Found {locs.shape[0]} beads')
    return locs, spots

def remove_colocal_locs(locs, spots, args):
    tqdm.write('Removing overlapping beads...')
    coords = locs[['x', 'y']].to_numpy()
    dists = euclidean_distances(coords, coords)
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)
    
    error_margin = 0.5
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
    return crossCount >= 2


def filter_mse(psf, args):
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

    permitted_avg_mse = 0.2
    permitted_max_mse = 0.15
    return avg_mse < permitted_avg_mse and max_mse < permitted_max_mse
        


def est_bead_offsets(psfs, locs, args):
    UPSCALE_RATIO = 10

    keep_psf = []

    def denoise(img):
        
        sigmas = np.array(args['gaussian_blur'])
        return gaussian_filter(img.copy(), sigma=sigmas)

    def get_sharpness(array):
        gy, gx = np.gradient(array)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        return sharpness

    def reduce_img(psf):
        return np.stack([get_sharpness(x) for x in psf])

    def find_peak(psf):
        if psf.ndim == 4:
            psf = psf.mean(axis=-1)
        x = np.arange(psf.shape[0]) * args['zstep']
        psf = denoise(psf)
        
        inten = norm_zero_one(reduce_img(psf))

        cs = UnivariateSpline(x, inten, k=3, s=0.2)

        x_ups = np.linspace(0, psf.shape[0], len(x) * UPSCALE_RATIO) * args['zstep']

        peak_xups = x_ups[np.argmax(cs(x_ups))] 

        fit = cs(x_ups)
        
        peak = max(fit)
        low = min(fit)
        half_max = (peak - low) / 2
        
        half_max_crossings = np.where(np.diff(np.sign(fit-half_max)))[0]
        keep_psf.append(len(half_max_crossings)>=2)
        return peak_xups

    offsets = np.array([find_peak(psf) for psf in psfs])

    locs['offset'] = offsets
    return np.array(keep_psf)

def filter_beads(spots, locs, stacks, args):
    print('Removing poorly imaged beads...')
    # Filter by SNR threshold
    snrs = np.array([snr(psf) > args['min_snr'] for psf in stacks])
    fwhms = np.array([has_fwhm(psf, args) for psf in stacks])
    mse_filters = np.array([filter_mse(psf, args) for psf in stacks])

    snrs[:] = True
    # fwhms[:] = True
    # mse_filters[:] = True
    peak_filters = est_bead_offsets(stacks, locs, args)


    idx = np.argwhere(snrs & fwhms & mse_filters & peak_filters).squeeze()

    if args['debug']:
        print('Writing rejected figures...')
        rejected_idx = np.argwhere(np.invert(snrs & fwhms & mse_filters & peak_filters)).squeeze()
        outpath = os.path.join(args['outpath'], 'combined', 'rejected')
        write_stack_figures(stacks[rejected_idx], locs.iloc[rejected_idx], outpath, args)
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

def write_stack_figures(stacks, locs, outpath, args):
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=False)

    for i in trange(stacks.shape[0]):
        stack = stacks[i]
        loc = locs.iloc[i].to_dict()

        fig = plt.figure(layout="constrained", figsize=(20, 15), dpi=80)
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
        ax2.set_xlabel('z (frame)')
        ax2.set_ylabel('pixel intensity')    


        for k, v in loc.items():
            if isinstance(v, float):
                loc[k] = round(v, 5)
        text = json.dumps(loc, indent=4)
        ax3.axis((0, 10, 0, 10))
        ax3.text(0,0, text, fontsize=18)
    
        plt.savefig(os.path.join(outpath, f'bead_{i}.png'))
        plt.close()


def main(args):
    all_stacks = []
    all_spots = []
    all_locs = []

    stacks = imread('./stacks.ome.tif')
    locs = pd.read_hdf('./locs.hdf', key='locs')

    spots = np.zeros(locs.shape)

    # n = 200
    # stacks = stacks[0:n]
    # locs = locs[0:n]

    # locs, stacks = remove_colocal_locs(raw_locs, stacks, args)
    # perc_removed = round(100*(1-(locs.shape[0]/raw_locs.shape[0])), 2)
    # print(f'Removed {perc_removed}% due to co-location')

        # stacks = extract_training_stacks(spots, bead_stack, args)

    spots, locs, stacks = filter_beads(spots, locs, stacks, args)
        # retained_beads += locs.shape[0]
        # tqdm.write(f'Retained {stacks.shape[0]} beads')
        # all_stacks.append(stacks)
        # all_spots.append(spots)
        # all_locs.append(locs)

    # min_stack_length = min(list(map(lambda s: s.shape[1], all_stacks)))
    # stacks = [s[:, :min_stack_length] for s in all_stacks]

    print(f'Kept {locs.shape[0]} total beads')

    write_combined_data(stacks, locs, args)

    if args['debug']:
        outpath = os.path.join(args['outpath'], 'combined', 'debug')
        write_stack_figures(stacks, locs, outpath, args)
        



def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('bead_stacks', help='Path to TIFF bead stacks / directory containing bead stacks.')
    parser.add_argument('-z', '--zstep', help='Pixel size (nm)', default=20, type=int)
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
    args['outpath'] = '.'
    return args


if __name__ == '__main__':
    main(transform_args(parse_args()))
