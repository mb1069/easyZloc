
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from scipy import optimize as opt
from tifffile import imread
import wandb
from util.util import preprocess_img_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
import argparse
import shutil
import json
import joblib
import sys
import os
cwd = os.path.dirname(__file__)
sys.path.append(cwd)


# # TODO remove this
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.use('Agg')


N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))


VERSION = '0.1'


def norm_whole_image(psfs):
    psfs = psfs - psfs.min()
    psfs = psfs / psfs.max()
    return psfs


def norm_psf_stack(psfs):
    for i in range(psfs.shape[0]):
        psf_min = psfs[i].min()
        psfs[i] = psfs[i] - psf_min
        psf_max = psfs[i].max()
        psfs[i] = psfs[i] / psf_max

    psfs[psfs < 0] = 0
    return psfs


def norm_psf_frame(psfs):
    for i in range(psfs.shape[0]):
        psf_min = psfs[i].min(keepdims=True)
        psfs[i] = psfs[i] - psf_min
        psf_max = psfs[i].max(keepdims=True)
        psfs[i] = psfs[i] / psf_max

    psfs[psfs < 0] = 0
    return psfs


def norm_frame_sum(psfs):
    for i in range(psfs.shape[0]):
        psf_min = psfs[i].min(keepdims=True)
        psfs[i] = psfs[i] - psf_min
        psf_sum = psfs[i].sum()
        psfs[i] = psfs[i] / psf_sum

    psfs[psfs < 0] = 0
    return psfs


norm_funcs = {
    'frame': norm_psf_frame,
    'stack': norm_psf_frame,
    'image': norm_whole_image,
    'sum': norm_frame_sum
}


def apply_coords_normalisation(coords, args):
    print('Applying pre-processing')
    scaler = joblib.load(args['coords_scaler'])

    coords = scaler.transform(coords)

    return coords


def pred_z(model, spots, coords):

    spots = spots.astype(np.float32)
    print('Predicting z locs')

    exp_spots = tf.data.Dataset.from_tensor_slices(spots)
    exp_coords = tf.data.Dataset.from_tensor_slices(coords)

    exp_X = tf.data.Dataset.zip((exp_spots, exp_coords))

    fake_z = np.zeros((coords.shape[0],))
    exp_z = tf.data.Dataset.from_tensor_slices(fake_z)

    exp_data = tf.data.Dataset.zip((exp_X, exp_z))

    BATCH_SIZE = 2048
    exp_data = exp_data.batch(BATCH_SIZE)
    exp_data = preprocess_img_dataset(exp_data, args)

    pred_z = model.predict(exp_data, batch_size=BATCH_SIZE, workers=4)

    return pred_z


def _find_matching_runs(hyper_params):
    runs = wandb.Api().runs('smlm_z3')
    print(
        f"Matching run with... {hyper_params['dataset']}, {hyper_params['aug_gauss']}, {hyper_params['aug_brightness']}, {hyper_params['learning_rate']}, {hyper_params['batch_size']}")
    try:
        while True:
            run = runs.next()

            rc = run.config
            try:
                config_match = [
                    str(rc['dataset']) == str(hyper_params['dataset']),
                    float(rc['aug_gauss']) == float(hyper_params['aug_gauss']),
                    float(rc['aug_brightness']) == float(
                        hyper_params['aug_brightness']),

                    float(rc['learning_rate']) == float(
                        hyper_params['learning_rate']),
                    float(rc['batch_size']) == float(
                        hyper_params['batch_size']),

                ]
            except KeyError:
                config_match = [False]
            if all(config_match):
                print(
                    f"Found run with... {rc.get('dataset')}, {rc.get('aug_gauss')}, {rc.get('learning_rate')}, {rc.get('batch_size')}")
                return run.id
    except StopIteration:
        quit()
        return None


def find_matching_run(args):
    report_path = os.path.join(args['model_dir'], 'results', 'report.json')

    with open(report_path) as f:
        report_data = json.load(f)

    if report_data.get('wandb_run_id'):
        run_id = report_data['wandb_run_id']
    else:
        hyper_params = {k: v for k, v in report_data['args'].items() if k in [
            'norm', 'dataset', 'aug_gauss', 'batch_size', 'learning_rate', 'aug_brightness']}
        run_id = _find_matching_runs(hyper_params)
    wandb.init(project='smlm_z3', id=run_id, resume=True)


def bestfit_error(z_true, z_pred):
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = mean_absolute_error(y_fit, y)
    return error, popt[0], y_fit, abs(y_fit - y)


def eval_dataset_without_const_bias(coords, zs, z_pred, dname):
    coords_str = np.array(['_'.join(x.astype(str))
                          for x in coords.astype(str)])
    coords_groups = {c: np.argwhere(coords_str == c).squeeze()
                     for c in set(coords_str)}

    errors = []

    vals = []
    preds = []
    fig = plt.figure()
    for c, idx in coords_groups.items():
        z_vals = zs[idx]
        z_preds = z_pred[idx]

        error = bestfit_error(z_vals, z_preds)[3]
        errors.append(error)
        z_preds -= z_preds.mean()
        z_vals -= z_vals.mean()
        plt.scatter(z_vals, z_preds)
        vals.append(z_vals)
        preds.append(z_preds)

    vals = np.concatenate(vals)
    preds = np.concatenate(preds)
    correl = pearsonr(vals, preds)[0]

    errors = np.concatenate(errors)

    mae = np.mean(errors)
    fpath = f'./{dname}.png'
    plt.savefig(fpath)
    plt.close()
    wandb.log({
        f'{dname}_p3': wandb.Image(fpath),
        f'{dname}_error': mae,
        f'{dname}_correl': correl
    })


def get_dataset_zstep(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config['zstep']


def main(args):

    find_matching_run(args)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(args['model'])

    for locs_path, psfs_path in args['datasets']:
        dname = os.path.basename(os.path.abspath(
            os.path.join(os.path.dirname(locs_path), os.pardir)))
        stacks_config = os.path.join(
            os.path.dirname(locs_path), 'stacks_config.json')

        locs = pd.read_hdf(locs_path, key='locs')
        psfs = imread(psfs_path)
        z_step = float(get_dataset_zstep(stacks_config))
        zs = []

        xy_coords = []

        for offset in locs['offset']:
            z = ((np.arange(psfs.shape[1])) * z_step) - offset
            zs.append(z)

        for xy in locs[['x', 'y']].to_numpy():
            xy_coords.append(
                np.repeat(xy[np.newaxis, :], repeats=psfs.shape[1], axis=0))

        xy_coords = np.array(xy_coords)
        zs = np.array(zs)

        zs = np.concatenate(zs)
        spots = np.concatenate(psfs)[:, :, :, np.newaxis].astype(np.float32)
        coords = np.concatenate(xy_coords)

        idx = np.argwhere(abs(zs) < 1000).squeeze()
        coords = coords[idx]
        spots = spots[idx]
        zs = zs[idx]

        coords = apply_coords_normalisation(coords, args)

        z_pred = pred_z(model, spots, coords).squeeze()

        eval_dataset_without_const_bias(coords, zs, z_pred, dname)

    # locs, info = io.load_locs(args['locs'])
    # locs = pd.DataFrame.from_records(locs)

    # with h5py.File(args['spots'], 'r') as f:
    #     spots = np.array(f['spots']).astype(np.uint16)

    # spots = (spots * GAIN / SENSITIVITY) + BASELINE

    # # TODO remove temp subset of locs
    # if args['picked_locs']:
    #     locs, spots = tmp_filter_locs(locs, spots, args)

    # assert locs.shape[0] == spots.shape[0]
    # print(locs.shape)
    # if XLIM or YLIM:
    #     spots, locs = extract_fov(spots, locs)

    # gen_2d_plot(locs, args['outdir'])
    # gen_example_spots(spots, args['outdir'])
    # coords, spots = apply_normalisation(locs, spots, args)

    # z_coords = pred_z(model, spots, coords, args['outdir'])

    # write_locs(locs, z_coords, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', '--model-dir',
                        help='Path to output dir from train_model')
    parser.add_argument(
        '-m', '--model', help='Path to trained 3d localisation model')
    parser.add_argument(
        '-d', '--datagen', help='Path to fitted image standardisation tool (datagen.gz)')
    parser.add_argument('-c', '--coords-scaler', help='2D coordinate rescaler')
    parser.add_argument('--datasets', required=True, nargs='+')
    parser.add_argument('--norm', required=True,
                        choices=['frame-min', 'frame-mean'])
    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['model_dir']:
        print('Using model dir from parameter -mo/--model-dir')
        dirname = os.path.abspath(args['model_dir'])
        args['model'] = os.path.join(dirname, 'model')
        args['datagen'] = os.path.join(dirname, 'datagen.gz')
        args['coords_scaler'] = os.path.join(dirname, 'scaler.save')

    for i in range(len(args['datasets'])):
        stacks = os.path.join(args['datasets'][i],
                              'combined', 'stacks.ome.tif')
        locs = os.path.join(args['datasets'][i], 'combined', 'locs.hdf')
        args['datasets'][i] = (locs, stacks)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


# # Johnny mitochondria data
# dirname = '/home/mdb119/data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/'
# locs = 'storm_1_MMStack_Default.ome_locs_undrift.hdf5'
# spots = 'storm_1_MMStack_Default.ome_spots.hdf5'
