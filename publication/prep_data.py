import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from itertools import repeat
from multiprocessing import Pool
from publication.util.util import grid_psfs, norm_zero_one
from publication.generate_spots import main as main_gen_spots
import seaborn as sns
import shutil
import json
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from keras.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as skl_mean_squared_error
from scipy.special import erf
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.filters import butterworth
from tifffile import imwrite, TiffFile
from sklearn.metrics import euclidean_distances
import numpy as np
import pandas as pd
import h5py
import subprocess
from natsort import natsorted
from glob import glob
from argparse import ArgumentParser
pd.options.mode.chained_assignment = None
import tensorflow as tf


def test_picasso_exec():
    # Checks Picasso can be run (i.e installation issues, library compatibility, etc)
    res = subprocess.run(['picasso', '-h'], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        print('\n')
        raise EnvironmentError('Picasso not found/working (see above)')


def transform_args(args):
    # Finds all tif files in path
    fnames = glob(f"{args['bead_stacks']}/**/*.tif", recursive=True)

    args['outpath'] = args['bead_stacks']

    # Filter out intermediate / output tif files
    fnames = [os.path.abspath(f) for f in fnames if '_slice.ome.tif' not in f and os.path.basename(f) != 'stacks.ome.tif']
    fnames = [b for b in fnames if ('ignored' not in b and 'combined' not in b)]
    args['bead_stacks'] = fnames

    n_stacks = len(args['bead_stacks'])
    print(f"Found {n_stacks} bead stacks")
    if n_stacks == 0:
        quit(1)
    for f in natsorted(args['bead_stacks']):
        print(f'\t - {f}')

    args['gaussian_blur'] = list(map(int, args['gaussian_blur'].split(',')))
    return args


def get_or_create_slice(bead_stack, slice_path):
    # Extract the brightest (i.e in-focus) slice from an image stack
    if not os.path.exists(slice_path):
        peak_idx = np.argmax(bead_stack.max(axis=(1, 2))).squeeze()
        im_slice = bead_stack[peak_idx]
        imwrite(slice_path, im_slice.astype(np.uint16))
    return slice_path


def get_or_create_locs(slice_path, pixel_size, args):
    # Load locs+spots if they exist, or generate them using Picasso

    spots_path = slice_path.replace('.ome.tif', '.ome_spots.hdf5')
    locs_path = slice_path.replace('.ome.tif', '.ome_locs.hdf5')

    if not os.path.exists(spots_path) or not os.path.exists(locs_path) or args['regen']:
        cmd = ['picasso', 'localize', slice_path, '-b', args['box_size_length'], '-g', args['gradient'], '-px', pixel_size]
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

    if not os.path.exists(spots_path):
        main_gen_spots({'locs': locs_path, 'img': slice_path})

    with h5py.File(spots_path) as f:
        spots = np.array(f['spots'])
    locs = pd.read_hdf(locs_path, key='locs')
    locs['fname'] = '___'.join(slice_path.split('/')[-3:])
    print(f'Found {locs.shape[0]} beads')
    return locs, spots


def remove_edge_spots(locs, spots, xsize, ysize, args):
    # Removes spots on edge of image as a full-size ROI cannot be extracted
    margin = args['box_size_length'] * (2 / 3)
    xlim = xsize
    ylim = ysize

    crit = (locs['x'] - margin >= 0)
    crit &= (locs['x'] + margin <= xlim)
    crit &= (locs['y'] - margin >= 0)
    crit &= (locs['y'] + margin <= ylim)
    valid_idx = np.argwhere(crit)[:, 0]
    return locs.iloc[valid_idx], spots[valid_idx]


def remove_colocal_beads(locs, spots, args):
    tqdm.write('Removing overlapping beads...')
    coords = locs[['x', 'y']].to_numpy()
    dists = euclidean_distances(coords, coords)
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)

    error_margin = 0.8
    min_seperation = (np.sqrt(2) * args['box_size_length']) * error_margin
    idx = np.argwhere(min_dists > min_seperation).squeeze()
    locs = locs.iloc[idx]
    spots = spots[idx]

    return locs, spots


def extract_training_stacks(locs, bead_stack, args):
    # Extracts stacks of beads from full FOV 
    spot_size = int(args['box_size_length'] / 2)
    stacks = []
    for loc in locs.to_dict(orient="records"):
        x = int(round(loc['x']))
        y = int(round(loc['y']))
        stack = bead_stack[:, y - spot_size:y + spot_size + 1, x - spot_size:x + spot_size + 1]
        stacks.append(stack)
    return np.array(stacks)


def snr(psf):
    return psf.max() / np.median(psf)


def has_fwhm_z(psf):
    psf = butterworth(psf, cutoff_frequency_ratio=0.2, high_pass=False)
    y = np.max(gaussian(psf), axis=(1, 2))
    max_val = np.max(y)
    min_val = np.min(y)
    half_max = min_val + ((max_val - min_val) / 2)
    crossCount = np.sum((y[:-1] > half_max) != (y[1:] > half_max))
    return crossCount >= 2


def filter_mse_zprofile(psf, z_step):

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
    y_data = psf.max(axis=(1, 2))
    y_data = norm_zero_one(y_data)
    initial_guess = [1, psf.shape[0] * z_step / 2, psf.shape[0] * z_step / 4, 0.0, np.median(y_data)]

    bounds = [
        (0.6, 1.2),
        (psf.shape[0] * z_step / 8, psf.shape[0] * z_step),
        (psf.shape[0] * z_step / 20, psf.shape[0] * z_step / 4),
        (-np.inf, np.inf),
        (y_data.min(), y_data.max())
    ]
    try:
        params, _ = curve_fit(skewed_gaussian, x_data, y_data, p0=initial_guess, bounds=list(zip(*bounds)))
    except RuntimeError:
        print('Failed to find Z fit')
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

    # Check for saturated img
    peak = np.max(y_data)
    n_peak_vals = np.argwhere(y_data == peak).shape[0]
    acceptable_saturation = (n_peak_vals * z_step) < 250
    # if not acceptable_saturation:
    #     plt.plot(x_data, y_data)
    #     plt.show()

    return avg_mse < permitted_avg_mse and max_mse < permitted_max_mse and acceptable_saturation


def get_sharpness(array):
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness


def reduce_img(psf):
    return np.stack([get_sharpness(x) for x in psf])


def est_bead_offsets(psfs, locs, z_step, args):
    UPSCALE_RATIO = 10

    def denoise(img):

        sigmas = np.array(args['gaussian_blur'])
        return gaussian_filter(img.copy(), sigma=sigmas)

    def find_peak(psf):
        if psf.ndim == 4:
            psf = psf.mean(axis=-1)
        x = np.arange(psf.shape[0]) * z_step
        psf = denoise(psf)

        inten = norm_zero_one(reduce_img(psf))

        cs = UnivariateSpline(x, inten, k=3, s=0.2)

        x_ups = np.linspace(0, psf.shape[0], len(x) * UPSCALE_RATIO) * z_step

        peak_xups = x_ups[np.argmax(cs(x_ups))]

        return peak_xups
    offsets = np.array(map(find_peak, psfs))

    locs['offset'] = offsets


def tf_eval_roll(ref_psf, psf, roll):
    return tf.reduce_mean(mean_squared_error(ref_psf, tf.roll(psf, roll, axis=0)))


def tf_find_optimal_roll(ref_tf, img):
    img_tf = tf.convert_to_tensor(img)

    roll_range = ref_tf.shape[0] // 4
    rolls = np.arange(-roll_range, roll_range).astype(int)
    errors = tf.map_fn(lambda roll: tf_eval_roll(ref_tf, img_tf, roll), rolls, dtype=tf.float64)
    # idx = 0
    # for roll in tqdm(rolls):
    #     error = tf.eval_roll(ref_tf, img_tf, roll)
    #     print(i, error)
    #     errors[idx] = error
    #     idx += 1

    best_roll = rolls[tf.argmin(errors).numpy()]
    # Prefer small backwards roll to large forwards roll
    if abs(best_roll - img.shape[0]) < best_roll:
        best_roll = best_roll - img.shape[0]

    return best_roll


def get_central_bead_idx(df):
    df['dist'] = euclidean_distances(df[['x', 'y']].to_numpy(), [[df['x'].max() / 2, df['y'].max() / 2]])
    ref_idx = np.argsort(df['dist'].to_numpy())[1]
    return ref_idx


def realign_beads(psfs, df, z_step):
    # Find relative offset in Z of beads relative to most central bead in FOV
    ref_idx = get_central_bead_idx(df)
    ref_offset = df.iloc[ref_idx]['offset']

    ref_psf = norm_zero_one(psfs[ref_idx])
    ref_tf = tf.convert_to_tensor(ref_psf)
    rolls = []
    for idx in trange(df.shape[0]):
        if idx == ref_idx:
            roll = 0
        else:
            psf2 = norm_zero_one(psfs[idx])
            roll = -tf_find_optimal_roll(ref_tf, psf2)
        rolls.append(roll)
    rolls = np.array(rolls)
    df['offset'] = rolls * z_step
    df['offset'] += ref_offset
    return psfs, df


def filter_2d_gaussian(image, max_mse):
    # Check 2D image of bead is approximately gaussian in shape
    # Filters out images with co-local beads where one bead was not localised, or
    # other statining/illumination issues

    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
        return g.ravel()

    # Load and preprocess the image (e.g., convert to grayscale)
    # For simplicity, let's generate a simple image for demonstration
    image_size = image.shape[1]
    x = np.linspace(0, image_size - 1, image_size)
    y = np.linspace(0, image_size - 1, image_size)
    x, y = np.meshgrid(x, y)

    image = image / image.max()

    # Fit the Gaussian to the image data
    p0 = [1, image_size / 2, image_size / 2, 2, 2, 0, 0]  # Initial guess for parameters
    bounds = [
        (0, np.inf),
        (image_size * (1 / 5), image_size * (4 / 5)),
        (image_size * (1 / 5), image_size * (4 / 5)),
        (0, image_size / 3),
        (0, image_size / 3),
        (-np.inf, np.inf),
        (0, np.inf),
    ]

    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=p0, bounds=list(zip(*bounds)))
    except RuntimeError:
        print('XY fit failed')
        popt = p0
    render = gaussian_2d((x, y), *popt).reshape(image.shape)

    error = skl_mean_squared_error(render, image)
    return error <= max_mse


def filter_mse_xy(stack, max_mse):
    sharp = reduce_img(stack)
    idx = np.argmax(sharp)
    image = stack[idx]
    return filter_2d_gaussian(image, max_mse)


def filter_beads(spots, locs, stacks, z_step, args, rejected_outpath):
    print('Removing poorly imaged beads...', end='')

    # Eval all filters on all bead stacks
    mse_xy = np.array([filter_mse_xy(psf, 5000) for psf in stacks])
    snrs = np.array([snr(psf) > args['min_snr'] for psf in stacks])
    fwhms = np.array([has_fwhm_z(psf) for psf in stacks])
    mse_z = np.array([filter_mse_zprofile(psf, z_step) for psf in stacks])

    # Keep only beads which pass all filters
    idx = np.argwhere(snrs & fwhms & mse_z & mse_xy)[:, 0]
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

    est_bead_offsets(stacks, locs, z_step, args)

    if args['debug']:
        rejected_idx = np.argwhere(np.invert(snrs & fwhms & mse_z & mse_xy))[:, 0]
        print('\n', 'Rejected: ', rejected_idx)

        if len(rejected_idx):
            print('Writing rejected figures...')
            print('Test1', 'offset' in list(locs))
            write_stack_figures(stacks[rejected_idx], locs.iloc[rejected_idx], rejected_outpath, z_step)

    spots = spots[idx]
    locs = locs.iloc[idx]
    stacks = stacks[idx]

    print('finished!')

    return spots, locs, stacks


def write_combined_data(stacks, locs, z_step, px_size, xsize, ysize, args):
    # Write training data (and optionally debug info about each bead)
    outpath = os.path.join(args['outpath'], 'combined')
    os.makedirs(outpath, exist_ok=True)

    locs_outpath = os.path.join(outpath, 'locs.hdf')
    stacks_outpath = os.path.join(outpath, 'stacks.ome.tif')

    imwrite(stacks_outpath, stacks)
    locs.to_hdf(locs_outpath, key='locs')

    stacks_config = {
        'zstep': z_step,
        'pixel_size': px_size,
        'gen_args': args,
        'xsize': xsize,
        'ysize': ysize
    }

    stacks_config_outpath = os.path.join(outpath, 'stacks_config.json')
    with open(stacks_config_outpath, 'w') as fp:
        json_dumps_str = json.dumps(stacks_config, indent=4)
        print(json_dumps_str, file=fp)

    figpath = os.path.join(outpath, 'offsets.png')
    sns.scatterplot(data=locs, x='x', y='y', hue='offset')
    plt.savefig(figpath)
    plt.close()

    print('Saved results to:')
    print(f'\t{locs_outpath}')
    print(f'\t{stacks_outpath}')
    print(f'\t{stacks_config_outpath}')
    print(f'Total beads: {locs.shape[0]}')


def write_stack_figure(i, central_bead_idx, stacks, locs, outpath, fname, z_step):
    # Write debug plot for an individual bead
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

    # plot main bead
    intensity = stack.max(axis=(1, 2))
    intensity = norm_zero_one(intensity)
    min_val = min(intensity)
    max_val = max(intensity)
    frame_zpos = (np.arange(len(intensity)) * z_step) - loc['offset']
    ax2.plot(frame_zpos, intensity, label='This bead')
    ax2.vlines(0, min_val, max_val, colors='orange')

    # Plot alignment ref bead
    intensity = stacks[central_bead_idx].max(axis=(1, 2))
    intensity = norm_zero_one(intensity)
    min_val = min(intensity)
    max_val = max(intensity)
    frame_zpos = (np.arange(len(intensity)) * z_step) - locs.iloc[central_bead_idx].to_dict()['offset']
    ax2.plot(frame_zpos, intensity, label='Alignment reference bead')

    ax2.set_title('Max normalised pixel intensity over z')
    ax2.set_xlabel('z (nm)')
    ax2.set_ylabel('pixel intensity')
    ax2.legend()

    for k, v in loc.items():
        if isinstance(v, float):
            loc[k] = round(v, 5)

    text = json.dumps(loc, indent=4)
    ax3.axis((0, 10, 0, 10))
    ax3.text(0, 0, text, fontsize=18, wrap=True)
    outfpath = os.path.join(outpath, f'{fname}_bead_{i}.png')
    plt.savefig(outfpath)
    plt.close()


def write_stack_figures(stacks, locs, outpath, z_step):
    fname = set(locs['fname']).pop().replace('.ome.tif', '')
    os.makedirs(outpath, exist_ok=True)

    idx = np.arange(stacks.shape[0])
    central_bead_idx = get_central_bead_idx(locs)
    with Pool(8) as pool:
        _ = pool.starmap(write_stack_figure, zip(idx, repeat(central_bead_idx), repeat(stacks), repeat(locs), repeat(outpath), repeat(fname), repeat(z_step)))



def get_pixel_size(tags):
    # Read metadata from tiff ImageJ tags
    num_pixels, units = tags['XResolution'].value
    _num_pixels, _units = tags['YResolution'].value
    val = units / num_pixels
    val2 = _units / _num_pixels
    if val != val2:
        raise NotImplementedError('Non-square pixel sizes not supported.')

    unit = tags['ResolutionUnit'].value
    if unit == 3:
        mult = 1e7
    else:
        raise NotImplementedError('Unknown resolution unit in tiff file')
    val = int(round(val * mult))
    return val


def main(args):
    # Extracts beads from each image stack and compiles them into output files
    all_stacks = []
    all_spots = []
    all_locs = []

    n_total_beads = 0
    retained_beads = 0

    rejected_outpath = os.path.join(args['outpath'], 'combined', 'rejected')
    shutil.rmtree(rejected_outpath, ignore_errors=True)

    if args['debug']:
        os.makedirs(rejected_outpath, exist_ok=True)

    z_step = args.get('z_step')
    px_size = args.get('pixel_size')

    xsize = None
    ysize = None
    for bead_stack_path in tqdm(natsorted(args['bead_stacks'])):
        tqdm.write(f'Preparing {os.path.basename(bead_stack_path)}')
        img_handle = TiffFile(bead_stack_path)
        metadata = img_handle.imagej_metadata
        tags = img_handle.pages[0].tags
        bead_stack = img_handle.asarray()
        img_handle.close()

        if 'spacing' in metadata:
            direction = metadata['spacing'] > 0
            if not direction:
                # Handle image sacks where piezo stage moved in opposite direction
                bead_stack = bead_stack[::-1]
                metadata['spacing'] *= -1

        if not args.get('z_step'):
            _z_step = round(abs(metadata['spacing']) * 1000)
            if z_step is None:
                z_step = _z_step
                print(f'Setting zstep to {z_step}nm.')
            else:
                if z_step != _z_step:
                    raise RuntimeError('Varying z_step not supported')

        if not args.get('pixel_size'):
            _px_size = round(get_pixel_size(tags))
            if px_size is None:
                px_size = _px_size
                print(f'Setting pixel size to {px_size}nm.')

            else:
                if px_size != _px_size:
                    raise RuntimeError('Varying pixel size not supported')

        if xsize is None:
            xsize = bead_stack.shape[2]
            ysize = bead_stack.shape[1]
        else:
            assert xsize == bead_stack.shape[2]
            assert ysize == bead_stack.shape[1]
        if bead_stack is None:
            print(f'Failed to read {bead_stack_path}')
            continue

        slice_path = bead_stack_path.replace('.ome', '_slice.ome')
        slice_path = get_or_create_slice(bead_stack, slice_path)

        raw_locs, spots = get_or_create_locs(slice_path, px_size, args)

        n_total_beads += raw_locs.shape[0]
        locs, spots = remove_colocal_beads(raw_locs, spots, args)
        locs, spots = remove_edge_spots(locs, spots, xsize, ysize, args)
        perc_removed = round(100 * (1 - (locs.shape[0] / raw_locs.shape[0])), 2)
        print(f'Removed {perc_removed}% due to co-location')

        stacks = extract_training_stacks(locs, bead_stack, args)
        spots, locs, stacks = filter_beads(spots, locs, stacks, z_step, args, rejected_outpath)
        retained_beads += locs.shape[0]
        tqdm.write(f'Retained {stacks.shape[0]} beads')
        all_stacks.append(stacks)
        all_spots.append(spots)
        all_locs.append(locs)

    print(f'Found {n_total_beads} total beads')
    min_stack_length = min(list(map(lambda s: s.shape[1], all_stacks)))
    stacks = [s[:, :min_stack_length] for s in all_stacks]
    locs = pd.concat(all_locs)
    stacks = np.concatenate(stacks)[:, :, :, :, np.newaxis]

    stacks, locs = realign_beads(stacks, locs, z_step)

    print(f'Kept {locs.shape[0]} total beads')

    write_combined_data(stacks, locs, z_step, px_size, xsize, ysize, args)

    if args['debug']:
        outpath = os.path.join(args['outpath'], 'combined', 'debug')
        write_stack_figures(stacks.mean(axis=-1), locs, outpath, z_step)


def check_path(args):
    dirpath = args['bead_stacks']
    if os.path.exists(dirpath):
        print('Bead directory exists')
        return args
    elif '/' in dirpath:
        raise ValueError(f'{dirpath} not found, check Windows vs UNIX pathing')
    else:
        raise ValueError(f'{dirpath} not found')


def parse_args():
    parser = ArgumentParser(description='Tool to extract, filter, pre-process and compile bead data')
    parser.add_argument('bead_stacks', help='Path to TIFF bead stacks / directory containing bead stacks.')
    parser.add_argument('-z', '--z_step', help='Overwrite metadata z step size (nm)', type=int)
    parser.add_argument('--zrange', help='Maximum z-range (+- value) over which to extract bead data (nm)', type=int, default=1000)
    parser.add_argument('-px', '--pixel_size', help='Overwrite metadata Pixel size (nm)', type=int)
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


def run_tool():
    args = parse_args()
    args = check_path(args)
    print(args)
    test_picasso_exec()
    args = transform_args(args)
    main(args)


if __name__ == '__main__':
    run_tool()
