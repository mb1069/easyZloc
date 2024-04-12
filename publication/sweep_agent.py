#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys, os
import json
import shutil

cwd = os.path.dirname(__file__)
sys.path.append(cwd)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

VERSION = '0.16'
CHANGE_NOTES = 'Fixing validation system'

import argparse
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util.util import grid_psfs, preprocess_img_dataset, image_size
from tifffile import imread
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.integration.keras import WandbMetricsLogger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, layers
from vit_keras import vit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error

import joblib
import gc

N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))

import tensorflow as tf

UPSCALE_RATIO = 1


def norm_frames(psfs):
    for i in range(psfs.shape[0]):
        psf_min = psfs[i].min(axis=(1,2), keepdims=True)
        psfs[i] -= psf_min
        psf_max = psfs[i].max(axis=(1,2), keepdims=True)
        psfs[i] /= psf_max
    psfs[psfs<0] = 0
    return psfs


def load_test_dataset(args):
    stacks = os.path.join(args['test_dataset'], 'combined', 'stacks.ome.tif')
    locs = os.path.join(args['test_dataset'], 'combined', 'locs.hdf')
    args = {
        'stacks': stacks,
        'locs': locs,
        'debug': False,
        'zstep': 10
    }
    psfs, locs, zs = load_data(args)

    xy_coords = []
    for xy in locs[['x', 'y']].to_numpy():
        xy_coords.append(np.repeat(xy[np.newaxis, :], repeats=psfs.shape[1], axis=0))

    xy_coords = np.array(xy_coords)
    zs = np.array(zs)

    zs = np.concatenate(zs)[:, np.newaxis]
    spots = np.concatenate(psfs)
    coords = np.concatenate(xy_coords)
    return spots, coords, zs


def load_data(args):
    psfs = imread(args['stacks'])[:, :, :, :, np.newaxis]
    locs = pd.read_hdf(args['locs'], key='locs')
    locs['idx'] = np.arange(locs.shape[0])

    if args['debug']:
        idx = np.arange(psfs.shape[0])
        np.random.seed(42)
        idx = np.random.choice(idx, 2000)
        idx = idx[0:100]
        psfs = psfs[idx]
        locs = locs.iloc[idx]

    ys = []
    for offset in locs['offset']:
        zs = ((np.arange(psfs.shape[1])) * args['zstep']) - offset
        ys.append(zs)

    ys = np.array(ys)

    psfs = psfs.astype(float)
    psfs = norm_frames(psfs)

    return psfs, locs, ys


def stratify_data(locs, args):
    def cart2pol(xy):
        x, y = xy
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    center = locs[['x', 'y']].mean().to_numpy()
    coords = locs[['x', 'y']].to_numpy() - center

    polar_coords = np.stack([cart2pol(xy) for xy in coords])

    discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal')
    groups = discretizer.fit_transform(polar_coords[:, 1:2]).astype(str)

    center_radius = 50
    idx = np.argwhere(polar_coords[:, 0] <= center_radius).squeeze()
    groups[idx] = -1

    if args['debug']:
        groups[:] = 0
    locs['group'] = groups

    return locs


# Withold some PSFs for evaluation

def split_train_val_test(psfs, locs, ys):

    def get_sub_ds(psfs, xy_coords, ys, idx):
        psfs_idx = psfs[idx]
        xy_coords_idx = xy_coords[idx].squeeze()
        xy_coords_idx = np.repeat(xy_coords_idx[:, :, np.newaxis], psfs_idx.shape[1], axis=0).squeeze()

        psfs_idx = np.concatenate(psfs_idx)
        ys_idx = np.concatenate(ys[idx])[:, np.newaxis]
        return psfs_idx, xy_coords_idx, ys_idx

    idx = np.arange(psfs.shape[0])

    # removed stratification
    train_idx, test_idx = train_test_split(idx, train_size=0.9, random_state=args['seed'])


    train_idx, val_idx = train_test_split(train_idx, train_size=0.9, random_state=args['seed'])

    xy_coords = locs[['x', 'y']].to_numpy()
    train_psfs, train_coords, train_ys = get_sub_ds(psfs, xy_coords, ys, train_idx)
    val_psfs, val_coords, val_ys = get_sub_ds(psfs, xy_coords, ys, val_idx)
    test_psfs, test_coords, test_ys = get_sub_ds(psfs, xy_coords, ys, test_idx)

    ds_cls = np.zeros((psfs.shape[0]), dtype=object)
    ds_cls[train_idx] = 'train'
    ds_cls[val_idx] = 'val'
    ds_cls[test_idx] = 'test'
    locs['ds'] = ds_cls
    plt.rcParams['figure.figsize'] = [5, 5]

    print(train_psfs.shape, train_coords.shape, train_ys.shape)
    print(val_psfs.shape, val_coords.shape, val_ys.shape)
    print(test_psfs.shape, test_coords.shape, test_ys.shape)

    return (train_psfs, train_coords, train_ys), (val_psfs, val_coords, val_ys), (test_psfs, test_coords, test_ys)


def filter_zranges(train_data, val_data, test_data, args):
    train_psfs, train_coords, train_ys = train_data
    val_psfs, val_coords, val_ys = val_data
    test_psfs, test_coords, test_ys = test_data

    def filter_zrange(X, zs):
        psfs, groups = X
        valid_ids = np.argwhere(abs(zs.squeeze()) < args['zrange']).squeeze()
        return [psfs[valid_ids], groups[valid_ids]], zs[valid_ids]
    
    
    X_train, y_train = filter_zrange((train_psfs, train_coords), train_ys)
    X_val, y_val = filter_zrange((val_psfs, val_coords), val_ys)
    X_test, y_test = filter_zrange((test_psfs, test_coords), test_ys)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def norm_xy_coords(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train[1] = scaler.fit_transform(X_train[1])
    X_val[1] = scaler.transform(X_val[1])
    X_test[1] = scaler.transform(X_test[1])

    outpath = os.path.join(args['outdir'], 'scaler.save')
    joblib.dump(scaler, outpath) 

    return X_train, X_val, X_test


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


def get_dataset(X, y, args, shuffle=False):
    images, coords = X
    img_ds = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
    coords_ds = tf.data.Dataset.from_tensor_slices(coords.astype(np.float32))
    labels_ds = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))

    x_ds = tf.data.Dataset.zip(img_ds, coords_ds)
    ds = tf.data.Dataset.zip(x_ds, labels_ds)
    ds = ds.batch(args['batch_size'])
    if shuffle:
        ds = ds.shuffle(buffer_size=int(args['batch_size']*1.5), seed=args['seed'])
    print('Created dataset')
    ds = ds.with_options(options)
    return ds


def prep_tf_dataset(dataset):
    return dataset.cache().prefetch(tf.data.AUTOTUNE)


# Assuming your input images have size (image_size, image_size, num_channels)
num_channels = 3
num_classes = 1  # Regression task, predicting a single continuous value


class RandomPoissonNoise(layers.Layer):
    def __init__(self, shape, lam_min, lam_max, rescale=65336, seed=42):
        super(RandomPoissonNoise, self).__init__()
        tf.random.set_seed(seed)

        self.shape = shape
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.rescale = rescale

    def call(self, input, training=False):
        if training==False:
            return input
        lam = tf.random.uniform((1,), self.lam_min, self.lam_max)[0]
        noise = tf.random.poisson(self.shape, lam, dtype=tf.float32) / self.rescale
        return input + noise


def get_model(args):
   # Create the Vision Transformer model using the vit_keras library
   imshape = (image_size, image_size, num_channels)
   img_input = Input(shape=imshape)
   extra_aug = Sequential([
        layers.GaussianNoise(stddev=args['gauss'], seed=args['seed']),
        # layers.RandomTranslation(1/imshape[0], 1/imshape[0], seed=args['seed']),
        layers.RandomBrightness(args['brightness'], value_range=[0, 1], seed=args['seed']),
        # RandomPoissonNoise(imshape, 1, args['poisson_lam'], seed=args['seed'])
    ], name='extra_aug')
   

   img_aug_out = extra_aug(img_input)
   
   coords_input = layers.Input((2,))
   x_coords = layers.Dense(64)(coords_input)
   
   x_coords = layers.Dense(64)(x_coords)
   
   model_version = {
       'vit_b16': vit.vit_b16,
       'vit_b32': vit.vit_b32,
       'vit_l16': vit.vit_l16,
       'vit_l32': vit.vit_l32,
   }[args['architecture']]

   vit_model = model_version(image_size=image_size, 
                           activation='sigmoid',
                           pretrained=True,
                           include_top=False,
                           pretrained_top=False)
   

   x = vit_model(img_aug_out)
   # Add additional layers for regression prediction
   x = Flatten()(x)
   x = tf.concat([x, x_coords], axis=-1)
   x = Dense(args['dense1'], activation='gelu')(x)
   x = Dropout(0.5)(x)
   x = Dense(args['dense2'], activation='gelu')(x)
   x = Dropout(0.5)(x)
   regression_output = Dense(num_classes, activation='linear')(x)  # Linear activation for regression
   model = Model(inputs=[img_input, coords_input], outputs=regression_output)
   
   aug_model = Model(inputs=img_input, outputs=img_aug_out)
   return model, aug_model


def bestfit_error(z_true, z_pred):
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = mean_absolute_error(y_fit, y)
    return error, popt[0], y_fit, abs(y_fit-y)


class ValidationCallback(Callback):
    def __init__(self, val_data):
        super(Callback, self).__init__()
        self.val_data = val_data
        images = []
        coords = []
        z = []
        for batch in val_data.as_numpy_iterator():
            images.append(batch[0][0])
            coords.append(batch[0][1])
            z.append(batch[1])

        self.images = np.concatenate(images)
        self.coords = np.concatenate(coords)
        self.z = np.concatenate(z)

        self.coords2 = np.array(['_'.join(x.astype(str)) for x in self.coords.astype(str)])
        self.coords_groups = {c: np.argwhere(self.coords2 == c).squeeze() for c in set(self.coords2)}

    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(self.val_data, batch_size=N_GPUS * 4096, verbose=False).squeeze()
        errors = []

        for group, idx in self.coords_groups.items():
            z_vals = self.z.squeeze()[idx]
            pred_vals = preds[idx].squeeze()

            error = bestfit_error(z_vals, pred_vals)[3]
            errors.append(error)
        
        mean_error = np.concatenate(errors).flatten().mean()
        print(f'\Val error: {round(np.mean(mean_error), 2)}')
        logs['val_mean_absolute_error'] = mean_error
        logs['val_loss'] = mean_error


def gen_example_aug_imgs(aug_model, train_data, args):
    for (imgs, _), _ in train_data.as_numpy_iterator():
        aug_images = aug_model(imgs).numpy().mean(axis=-1)
        break

    plt.figure(figsize=(30, 10), dpi=300)
    plt.imshow(grid_psfs(aug_images, cols=32))
    outpath = f"{args['outdir']}/sample_aug.png"
    plt.savefig(outpath)
    plt.close()
    wandb.log({'aug_example': wandb.Image(outpath)})
    del aug_model
    gc.collect()



def train_model(train_data, val_data, args):
    epochs = 3 if args['debug'] else 5000
    lr = args['learning_rate']
    print(f'N epochs: {epochs}')


    # Transfer learning
    # n_layers = len(model.layers)
    # for i in range(0, len(model.layers)-4):
    #     model.layers[i].trainable = False
    # assert model.trainable == True

    
    # # Print a summary of the model architecture
    # model.summary()

    # # # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # # Open a strategy scope.
    with strategy.scope():

        # Model refining
        # model = keras.models.load_model('./latest_vit_model3/')

        # Traing from scratch
        model, aug_model = get_model(args)

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer=optimizers.AdamW(learning_rate=lr), metrics=['mean_absolute_error'])

    gen_example_aug_imgs(aug_model, train_data, args)


    callbacks = [
        ValidationCallback(val_data),
        WandbMetricsLogger(),
        ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
                        patience=5, verbose=True, mode='min', min_delta=1, min_lr=1e-8,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=10,
                    verbose=True, min_delta=1, restore_best_weights=True),
                    
    ]

    try:
        history = model.fit(train_data, epochs=epochs, callbacks=callbacks, shuffle=True, verbose=True)
    except KeyboardInterrupt:
        print('\nInterrupted training\n')
        history = None

    model.save(os.path.join(args['outdir'], './latest_vit_model3'))

    print('Finished!')

    if history:
        plt.rcParams['figure.figsize'] = [10, 10]
        fig, ax1 = plt.subplots()
        ax1.plot(history.history['mean_absolute_error'], label='mse')
        ax1.plot(history.history['val_mean_absolute_error'], label='val_mse')
        ax1.set_ylim([0, 500])
        ax1.legend(loc=1)
        ax2 = ax1.twinx()
        ax2.plot(history.history['lr'], label='lr', color='red')
        ax2.legend(loc=0)
        fig.savefig(os.path.join(args['outdir'], 'training_curve.png'))
        plt.close()

    return model


def get_test_error(model, test_data):
    z = []
    for batch in test_data.as_numpy_iterator():
        z.append(batch[1])

    z = np.concatenate(z)

    preds = model.predict(test_data, batch_size=N_GPUS * 4096).squeeze()
    return mean_absolute_error(z, preds)



def save_copy_training_script(outdir):
    outpath = os.path.join(outdir, 'train_model.py.bak')
    shutil.copy(os.path.abspath(__file__), outpath)


def run_config(X_train, y_train, X_val, y_val, X_test, y_test, args):
    wandb.init(
        project='smlm_z3',
        config = {
            'dataset': args['dataset'],
            'learning_rate': args['learning_rate'],
            'architecture': args['architecture'],
            'batch_size': args['batch_size'],
            'n_gpus': N_GPUS,
            'aug_ratio': args['aug_ratio'],
            'aug_brightness': args['aug_brightness'],
            'aug_gauss': args['aug_gauss'],
            'aug_poisson_lam': args['aug_poisson_lam'],
            'norm': 'frame'
        }
    )
    args['gauss'] = args['aug_gauss']
    args['brightness'] = args['aug_brightness']

    train_data = get_dataset(X_train, y_train, args, True)
    val_data = get_dataset(X_val, y_val, args)
    test_data = get_dataset(X_test, y_test, args)

    train_data = preprocess_img_dataset(train_data)
    val_data = preprocess_img_dataset(val_data)
    test_data = preprocess_img_dataset(test_data)

    train_data = prep_tf_dataset(train_data)
    val_data = prep_tf_dataset(val_data)
    test_data = prep_tf_dataset(test_data)    


    model = train_model(train_data, val_data, args)
    wandb.log({'ext_test_mae':  get_test_error(model, test_data)})


def main(args):

    stacks, locs, zs = load_data(args)

    test = load_test_dataset(args)

    locs = stratify_data(locs, args)
    train, val, _ = split_train_val_test(stacks, locs, zs)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = filter_zranges(train, val, test, args)

    X_train, X_val, X_test = norm_xy_coords(X_train, X_val, X_test)

    run_config(X_train, y_train, X_val, y_val, X_test, y_test, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stacks', help='TIF file containing stacks in format N*Z*Y*X', default='./stacks.ome.tif')
    parser.add_argument('-l' ,'--locs', help='HDF5 locs file', default='./locs.hdf')
    parser.add_argument('-sc', '--stacks_config', help='JSON config file for stacks file (can be automatically found if in same dir)', default='./stacks_config.json')
    parser.add_argument('-zrange', '--zrange', help='Z to model (+-val) in nm', default=1000, type=int)
    parser.add_argument('-o', '--outdir', help='Output directory', default='./out')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--seed', default=42, type=int, help='Random seed (for consistent results)')
    parser.add_argument('--dense1', required=True, type=int)
    parser.add_argument('--dense2', required=True, type=int)

    parser.add_argument('-b', '--batch_size', required=True, type=int, help='Batch size (per GPU)')
    parser.add_argument('--aug_ratio', type=float, help='Aug ratio', default=2)
    parser.add_argument('--aug_brightness', type=float, help='Brightness', default=0.5)
    parser.add_argument('--aug_gauss', type=float, help='Gaussian', default=0.005)
    parser.add_argument('--aug_poisson_lam', type=float, help='Poisson noise lam', default=1000)
    parser.add_argument('--architecture')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    parser.add_argument('--dataset', help='Dataset type, used for wandb', default='openframe')
    parser.add_argument('--test-dataset', default='/data/mdb119/data/20231128_tubulin_miguel/')


    args = vars(parser.parse_args())

    if not os.path.exists(args['stacks_config']):
        print(f'Could not find stacks-config file, checking in dir of stacks...')
        args['stacks-config'] = os.path.join(os.path.dirname(args['stacks']), 'stacks_config.json')
        if not os.path.exists(args['stacks_config']):
            print(f'Could not find stacks-config file')
            quit(1)

    with open(args['stacks_config']) as f:
        d = json.load(f)
    args.update(d)
    print(args)
    for k, v in args.items():
        print(f'{k}: {v}')

    return args

if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args['outdir'], exist_ok=True)

    tf.keras.utils.set_random_seed(args['seed'])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()

    main(args)
    wandb.finish()