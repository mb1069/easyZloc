
import sys, os
cwd = os.path.dirname(__file__)
sys.path.append(cwd)


# # TODO remove this
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

import joblib
import json
import shutil
import argparse
import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Resizing, Lambda
from tensorflow.keras import Sequential
import tensorflow as tf
from picasso import io

from util.util import grid_psfs



N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))


VERSION = '0.1'



# Picasso localisation parameters
BASELINE = 100
SENSITIVITY = 0.45
GAIN = 1


DEFAULT_LOCS = None
DEFAULT_SPOTS = None
DEFAULT_PIXEL_SIZE = None
PICKED = None
XLIM, YLIM = None, None

# NUP FD-LOCO
# DEFAULT_LOCS = '/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_locs.hdf5'
# DEFAULT_SPOTS = '/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_spots.hdf5'
# PICKED = '/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_locs_picked.hdf5'
# DEFAULT_PIXEL_SIZE = 110
# XLIM, YLIM = None, None



# NUP OPENFRAME
# DEFAULT_LOCS = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5'
# DEFAULT_SPOTS = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5'
# PICKED = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted_picked_4.hdf5'
# DEFAULT_PIXEL_SIZE = 86
# XLIM, YLIM = None, None



# Zeiss
# DEFAULT_LOCS = '/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5'
# DEFAULT_SPOTS = '/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5'
# PICKED = '/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_picked.hdf5'
# DEFAULT_PIXEL_SIZE = 106
# XLIM, YLIM = None, None


# Unused below

# # Mitochondria (older)
# DEFAULT_LOCS = '/home/miguel/Projects/data/20231205_miguel_mitochondria/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5'
# DEFAULT_SPOTS = '/home/miguel/Projects/data/20231205_miguel_mitochondria/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5'
# DEFAULT_PIXEL_SIZE = 86
# XLIM, YLIM = None, None

# Mitochondria (newer) (still not clearly working)
# DEFAULT_LOCS = '/media/Data/smlm_z_data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5'
# DEFAULT_SPOTS = '/media/Data/smlm_z_data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5'
# PICKED = None
# DEFAULT_PIXEL_SIZE = 86
# XLIM = 400, 600
# YLIM = 700, 1000



# Tubulin
# DEFAULT_LOCS = '/media/Data/smlm_z_data/20231212_miguel_openframe/tubulin/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5'
# DEFAULT_SPOTS = '/media/Data/smlm_z_data/20231212_miguel_openframe/tubulin/FOV1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5'
# PICKED = None
# DEFAULT_PIXEL_SIZE = 86
# XLIM = 200, 800
# YLIM = 500, 1000

def write_arg_log(args):
    outfile = os.path.join(args['outdir'], 'config.json')
    with open(outfile, 'w') as fp:
        json_dumps_str = json.dumps(args, indent=4)
        print(json_dumps_str, file=fp)


def save_copy_script(outdir):
    outpath = os.path.join(outdir, 'localise_exp_sample.py.bak')
    shutil.copy(os.path.abspath(__file__), outpath)


def gen_2d_plot(locs, outdir):
    print('Gen 2d plot')
    sns.scatterplot(data=locs, x='x', y='y', marker='.', alpha=0.1)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(outdir, '2d_scatterplot.png'))
    plt.close()


def gen_example_spots(spots, outdir):
    print('Gen example splots')
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.imshow(grid_psfs(spots[0:100]))
    plt.savefig(os.path.join(outdir, 'example_spots.png'))
    plt.close()


def apply_normalisation(locs, spots, args):
    print('Applying pre-processing')
    scaler = joblib.load(args['coords_scaler'])
    datagen = joblib.load(args['datagen'])

    coords = scaler.transform(locs[['x', 'y']].to_numpy())
    spots = datagen.standardize(spots.astype(np.float32))[:, :, :, np.newaxis]

    return coords, spots



def pred_z(model, spots, coords, outdir):

    spots = spots.astype(np.float32)
    print('Predicting z locs')

    # exp_spots = tf.data.Dataset.from_generator(
    #     generator=lambda: iter(spots),
    #     output_signature=tf.TensorSpec(shape=spots.shape[1:], dtype=tf.float32)
    # )
    exp_spots = tf.data.Dataset.from_tensor_slices(spots)
    exp_coords = tf.data.Dataset.from_tensor_slices(coords)

    exp_X = tf.data.Dataset.zip((exp_spots, exp_coords))

    fake_z = np.zeros((coords.shape[0],))
    exp_z = tf.data.Dataset.from_tensor_slices(fake_z)

    exp_data = tf.data.Dataset.zip((exp_X, exp_z))

    image_size = 64
    imshape = (image_size, image_size)
    img_preprocessing = Sequential([
        Resizing(*imshape),
        Lambda(tf.image.grayscale_to_rgb)
    ])

    def apply_rescaling(x, y):
        x = [x[0], x[1]]
        x[0] = img_preprocessing(x[0])
        return tuple(x), y

    BATCH_SIZE = 2048
    exp_data = exp_data.map(apply_rescaling, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

    pred_z = model.predict(exp_data, batch_size=BATCH_SIZE, workers=4)

    

    sns.histplot(pred_z)
    plt.savefig(os.path.join(outdir, 'z_histplot.png'))
    plt.close()
    return pred_z

def write_locs(locs, z_coords, args):
    locs['z [nm]'] = z_coords
    locs['z'] = locs['z [nm]']
    # locs['z'] = z_coords / args['pixel_size']
    locs['x [nm]'] = locs['x'] * args['pixel_size']
    locs['y [nm]'] = locs['y'] * args['pixel_size']

    locs_path = os.path.join(args['outdir'], 'locs_3d.hdf5')
    with h5py.File(locs_path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs.to_records())

    yaml_file = args['locs'].replace('.hdf5', '.yaml')
    if os.path.exists(yaml_file):
        dest_yaml = locs_path.replace('.hdf5', '.yaml')
        shutil.copy(yaml_file, dest_yaml)
    else:
        dest_yaml = None
        print('Could not write yaml file (original from 2D localisation not found)')
    print('Wrote results to:')
    print(f'\t- {os.path.abspath(locs_path)}')
    if dest_yaml:
        print(f'\t- {os.path.abspath(dest_yaml)}')


def write_report_data(args):
    report_data = {
        'code_version': VERSION
    }
    report_data.update(args)
    with open(os.path.join(args['outdir'], 'report.json'), 'w') as fp:
        json_dumps_str = json.dumps(report_data, indent=4)
        print(json_dumps_str, file=fp)


def extract_fov(spots, locs):
    print(locs.shape)
    idx = np.argwhere((XLIM[0]<locs['x']) & (XLIM[1]>locs['x']) & (YLIM[0]<locs['y']) & (YLIM[1]>locs['y'])).squeeze()
    spots = spots[idx]
    locs = locs.iloc[idx]
    return spots, locs

def tmp_filter_locs(new_locs, spots, args):
    old_locs = pd.read_hdf(args['picked_locs'], key='locs')

    idx = np.argwhere(new_locs['x'].isin(old_locs['x'])).squeeze()
    new_locs = new_locs.iloc[idx]
    spots = spots[idx]
    return new_locs, spots

def norm_ds_zero_one(imgs):
    mins = imgs.min(axis=(1,2,3), keepdims=True)
    maxs = imgs.max(axis=(1,2,3), keepdims=True)
    return (imgs-mins) / (maxs-mins)

def main(args):
    write_report_data(args)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(args['model'])
    
    locs, info = io.load_locs(args['locs'])
    locs = pd.DataFrame.from_records(locs)

    with h5py.File(args['spots'], 'r') as f:
        spots = np.array(f['spots']).astype(np.uint16)

    spots = (spots * GAIN / SENSITIVITY) + BASELINE
    spots = norm_ds_zero_one(spots)


    # TODO remove temp subset of locs
    if args['picked_locs']:
        locs, spots = tmp_filter_locs(locs, spots, args)

    assert locs.shape[0] == spots.shape[0]
    print(locs.shape)
    if XLIM or YLIM:
        spots, locs = extract_fov(spots, locs)

    gen_2d_plot(locs, args['outdir'])
    gen_example_spots(spots, args['outdir'])
    coords, spots = apply_normalisation(locs, spots, args)

    z_coords = pred_z(model, spots, coords, args['outdir'])

    write_locs(locs, z_coords, args)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', '--model-dir', help='Path to output dir from train_model')
    parser.add_argument('-m', '--model', help='Path to trained 3d localisation model')
    parser.add_argument('-d', '--datagen', help='Path to fitted image standardisation tool (datagen.gz)')
    parser.add_argument('-c', '--coords-scaler', help='2D coordinate rescaler')
    parser.add_argument('-l', '--locs', help='2D localisation file from Picasso', required=True)
    parser.add_argument('-s', '--spots', help='Spots file from Picasso', required=True)
    parser.add_argument('-p', '--picked-locs', help='Localisations picked from Locs file using Picasso')

    parser.add_argument('-px', '--pixel_size', help='Pixel size (nm)', type=int, required=True)
    parser.add_argument('-o', '--outdir', help='Output dir', default='./out')
    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['model_dir']:
        print('Using model dir from parameter -mo/--model-dir')
        dirname = os.path.abspath(args['model_dir'])
        args['model'] = os.path.join(dirname, 'latest_vit_model3')
        args['datagen'] = os.path.join(dirname, 'datagen.gz')
        args['coords_scaler'] = os.path.join(dirname, 'scaler.save')

    args['locs'] = os.path.abspath(args['locs'])
    args['spots'] = os.path.abspath(args['spots'])

    os.makedirs(args['outdir'], exist_ok=True)

    write_arg_log(args)
    save_copy_script(args['outdir'])
    

    for k, v in args.items():
        if v is None:
            continue
        if k in ['pixel_size']:
            continue
        try:
            assert os.path.exists(v)
        except Exception:
            print(f'{k}:{v} not found')
            quit()
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)


# # Johnny mitochondria data
# dirname = '/home/mdb119/data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/'
# locs = 'storm_1_MMStack_Default.ome_locs_undrift.hdf5'
# spots = 'storm_1_MMStack_Default.ome_spots.hdf5'
