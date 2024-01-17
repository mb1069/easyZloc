
import sys, os
cwd = os.path.dirname(__file__)
sys.path.append(cwd)

import joblib
import json
import shutil
import keras
import argparse
import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Resizing, Lambda
from tensorflow.keras import Sequential
import tensorflow as tf

from util.util import grid_psfs


N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))


VERSION = '0.1'

# TODO remove this
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', '--model-dir', help='Path to output dir from train_model')
    parser.add_argument('-m', '--model', help='Path to trained 3d localisation model')
    parser.add_argument('-d', '--datagen', help='Path to fitted image standardisation tool (datagen.gz)')
    parser.add_argument('-c', '--coords-scaler', help='2D coordinate rescaler')
    parser.add_argument('-l', '--locs', help='2D localisation file from Picasso')
    parser.add_argument('-s', '--spots', help='Spots file from Picasso')
    parser.add_argument('-px', '--pixel_size', help='Pixel size (nm)', default=86, type=int)
    parser.add_argument('-o', '--outdir', help='Output dir', default='./out')
    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['model_dir']:
        print('Using model dir from parameter -mo/--model-dir')
        dirname = os.path.abspath(args['model_dir'])
        args['model'] = os.path.join(dirname, 'latest_vit_model3')
        args['datagen'] = os.path.join(dirname, 'datagen.gz')
        args['coords_scaler'] = os.path.join(dirname, 'scaler.save')

    os.makedirs(args['outdir'], exist_ok=True)

    for k, v in args.items():
        if k in ['pixel_size']:
            continue
        try:
            assert os.path.exists(v)
        except AssertionError:
            print(f'{v} not found')
            quit()
    return args


def gen_2d_plot(locs, outdir):
    sns.scatterplot(data=locs, x='x', y='y', marker='.', alpha=0.01)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(outdir, '2d_scatterplot.png'))
    plt.close()


def gen_example_spots(spots, outdir):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.imshow(grid_psfs(spots[0:100]))
    plt.savefig(os.path.join(outdir, 'example_spots.png'))
    plt.close()


def apply_preprocessing(locs, spots, args):
    scaler = joblib.load(args['coords_scaler'])
    datagen = joblib.load(args['datagen'])

    coords = scaler.transform(locs[['x', 'y']].to_numpy())
    spots = datagen.standardize(spots.astype(float))[:, :, :, np.newaxis]

    return coords, spots


def pred_z(model, spots, coords, outdir):

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


    exp_data = exp_data.map(apply_rescaling, num_parallel_calls=tf.data.AUTOTUNE).batch(2048*3 * N_GPUS)

    pred_z = model.predict(exp_data)

    sns.histplot(pred_z)
    plt.savefig(os.path.join(outdir, 'z_histplot.png'))
    plt.close()
    return pred_z

def write_locs(locs, z_coords, args):
    locs['z [nm]'] = z_coords
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
    print(f'\t- {locs_path}')
    if dest_yaml:
        print(f'\t- {dest_yaml}')


def write_report_data(args):
    report_data = {
        'code_version': VERSION
    }
    report_data.update(args)
    with open(os.path.join(args['outdir'], 'report.json'), 'w') as fp:
        json_dumps_str = json.dumps(report_data, indent=4)
        print(json_dumps_str, file=fp)


def main():
    args = parse_args()
    write_report_data(args)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(args['model'])
    
    locs = pd.read_hdf(args['locs'], key='locs')
    with h5py.File(args['spots'], 'r') as f:
        spots = np.array(f['spots']).astype(np.uint16)

    gen_2d_plot(locs, args['outdir'])
    gen_example_spots(spots, args['outdir'])

    coords, spots = apply_preprocessing(locs, spots, args)

    z_coords = pred_z(model, spots, coords, args['outdir'])

    write_locs(locs, z_coords, args)

if __name__=='__main__':
    main()


# Johnny mitochondria data
dirname = '/home/mdb119/data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/'
locs = 'storm_1_MMStack_Default.ome_locs_undrift.hdf5'
spots = 'storm_1_MMStack_Default.ome_spots.hdf5'
