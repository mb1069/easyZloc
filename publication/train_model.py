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
from util.util import grid_psfs, load_dataset, save_dataset, load_model, preprocess_img_dataset, image_size
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


def load_data(args):
    psfs = imread(args['stacks'])[:, :, :, :, np.newaxis].astype(float)
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


def prep_dataset(dataset):
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

    if not (args['aug_gauss'] or args['aug_brightness']):
        img_aug_out = img_input
    else:
        extra_aug = Sequential([], name='extra_aug')

        if args['aug_gauss']:
            extra_aug.add(layers.GaussianNoise(stddev=args['aug_gauss'], seed=args['seed']))

        if args['aug_brightness']:
            extra_aug.add(layers.RandomBrightness(args['aug_brightness'], value_range=[0, 1], seed=args['seed']))
            
        # layers.RandomTranslation(1/imshape[0], 1/imshape[0], seed=args['seed']),
        if args['aug_poisson_lam']:
            extra_aug.add(RandomPoissonNoise(imshape, 1, args['aug_poisson_lam'], seed=args['seed']))
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
    n_batches = 2
    aug_images = []
    for (imgs, _), _ in train_data.as_numpy_iterator():
        aug_images.append(aug_model(imgs).numpy().mean(axis=-1))
        n_batches -= 1
        if n_batches == 0:
            break

    aug_images = np.concatenate(aug_images)
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
        # 
        if args['pretrained_model']:
            model = keras.models.load_model(args['pretrained_model'])
        else:
            # Traing from scratch
            model, aug_model = get_model(args)
            gen_example_aug_imgs(aug_model, train_data, args)

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer=optimizers.AdamW(learning_rate=lr), metrics=['mean_absolute_error'])

    


    callbacks = [
        ValidationCallback(val_data),
        WandbMetricsLogger(),
        ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
                        patience=10, verbose=True, mode='min', min_delta=1, min_lr=1e-8,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=20,
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



def write_report(model, locs, train_data, val_data, test_data, args):

    # Check output on all stacks

    tbl_data = []
    report_data = {
        'code_version': VERSION,
        'change_notes': CHANGE_NOTES,
        'args': args,
        'wandb_run_id': wandb.run.id,
    }

    os.makedirs(os.path.join(args['outdir'], 'results'), exist_ok=True)
    
    sns.scatterplot(data=locs, x='x', y='y', hue='ds')
    plt.savefig(os.path.join(args['outdir'], 'results', 'data_split') + '.png')
    plt.close()



    for dirname, ds in [('train', train_data), ('val', val_data), ('test', test_data)]:
        os.makedirs(os.path.join(args['outdir'], 'results', dirname), exist_ok=True)
        images = []
        coords = []
        z = []
        for batch in ds.as_numpy_iterator():
            images.append(batch[0][0])
            coords.append(batch[0][1])
            z.append(batch[1])

        images = np.concatenate(images)
        coords = np.concatenate(coords)
        z = np.concatenate(z)

        preds = model.predict(ds, batch_size=N_GPUS * 4096).squeeze()
        errors = abs(preds - z.squeeze())

        fname = f'{dirname}_all_preds'
        fig = plt.figure(layout="constrained", figsize=(12, 10), dpi=80)
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        plt.title(fname)
        ax1.scatter(z, preds)
        ds_df = pd.DataFrame.from_dict({'x': coords[:, 0], 'y': coords[:, 1], 'error': errors})
        ds_df = ds_df.groupby(['x', 'y']).mean().reset_index()

        sns.scatterplot(data=ds_df, x='x', y='y', hue='error', ax=ax2)
        sns.scatterplot(data=ds_df, x='x', y='error', ax=ax3)
        sns.scatterplot(data=ds_df, x='y', y='error', ax=ax4)
        img_fname = fname + '.png'
        outpath = os.path.join(args['outdir'], 'results', dirname, img_fname)
        plt.savefig(outpath)
        plt.close()
        wandb.log({img_fname: wandb.Image(outpath)})


        coords2 = ['_'.join(x.astype(str)) for x in coords]
        ds_true_vals = []
        ds_pred_vals = []
        for num, c in tqdm(enumerate(sorted(set(coords2))), total=len(set(coords2))):
            idx = [i for i, val in enumerate(coords2) if val==c]
            idx = sorted(idx, key=lambda i: z.squeeze()[i])
            z_vals = z.squeeze()[idx]
            ds_true_vals.extend(z_vals)
            pred_vals = preds[idx]
            group_images = images[idx]
            error = abs(z_vals-pred_vals)

            mae = str(round(error.mean(), 2))
            
            fig = plt.figure(layout="constrained", figsize=(12, 10), dpi=80)
            gs = plt.GridSpec(4, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 1])
            ax5 = fig.add_subplot(gs[3, 1])


            fig.suptitle(f'Bead: {num}')
            ax1.scatter(z_vals, pred_vals)
            ax1.plot([-1000, 1000], [-1000, 1000], c='orange')
            ax1.set_xlabel('estimated true z (nm)')
            ax1.set_ylabel('Predicted z (nm)')
            ax1.set_title(f'MAE: {mae}nm')

            
            ax2.imshow(grid_psfs(group_images.mean(axis=-1)).T)
            ax2.set_title('Ordered by frame')

            ax3.imshow(grid_psfs(group_images[np.argsort(error)].mean(axis=-1)).T)
            ax3.set_title('Ordered by increasing prediction error')
            ax3.set_xlabel(f'min error: {str(round(min(error), 2))}, max error: {str(round(max(error), 2))}')

            
            ax4.scatter(coords[:, 0], coords[:, 1])
            gcoords = list(map(lambda x: [float(x)], c.split('_')))
            ax4.scatter(gcoords[0], gcoords[1])
            ax4.set_xlabel('x (nm)')
            ax4.set_ylabel('y (nm)')
            ax4.set_title('2d loc within dataset')

            sorted_idx = np.argsort(z_vals)
            ax5.plot(z_vals[sorted_idx], group_images.max(axis=(1,2,3))[sorted_idx])
            ax5.set_title('Max normalised pixel intensity over z')
            ax5.set_xlabel('z (frame)')
            ax5.set_ylabel('pixel intensity')    
                
            
            img_fname = f'{dirname}_bead' + f'_{num}.png'
            outpath = os.path.join(args['outdir'], 'results', dirname, img_fname)
            plt.savefig(outpath)
            
            plt.close()

            # wandb.log({img_fname: wandb.Image(outpath)})

            if dirname != 'train':
                adj_mae, offset, zs, error = bestfit_error(z_vals, pred_vals)
                pred_vals -= offset
                adj_mae = str(round(adj_mae, 2))
                offset_fmt = str(round(offset, 2))
                ax1.set_title(f'MAE: {mae}nm, corrected MAE: {adj_mae}nm, offset: {offset_fmt}nm')
                mae = adj_mae
            else:
                offset = 0

            ds_pred_vals.extend(pred_vals)
            tbl_data.append((dirname, num,  gcoords[0][0], gcoords[1][0], mae, min(error), max(error),offset))
        
        ds_mae = mean_absolute_error(ds_true_vals, ds_pred_vals)
        report_data[dirname+'_mae'] = float(ds_mae)
        print(dirname, report_data[dirname+'_mae'])

        wandb.run.summary[dirname+'_mae'] = float(ds_mae)


    df = pd.DataFrame(tbl_data, columns=['dataset', 'id', 'x', 'y', 'mae', 'min_error', 'max_error', 'offset'])
    df.to_csv(os.path.join(args['outdir'], 'results', 'results.csv'))


    with open(os.path.join(args['outdir'], 'results', 'report.json'), 'w') as fp:
        json_dumps_str = json.dumps(report_data, indent=4)
        print(json_dumps_str, file=fp)


def save_copy_training_script(outdir):
    outpath = os.path.join(outdir, 'train_model.py.bak')
    shutil.copy(os.path.abspath(__file__), outpath)

def prepare_data(args):

    stacks, locs, zs = load_data(args)

    locs = stratify_data(locs, args)
    train, val, test = split_train_val_test(stacks, locs, zs)

    if args['ext_test_dataset']:
        test_imgs, test_xy, y_test = load_ext_test_dataset(args)
        X_test = (test_imgs, test_xy)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = filter_zranges(train, val, test, args)

    # X_train, X_val, X_test = norm_images(X_train, X_val, X_test, args)

    # X_train, y_train = aug_train_data(X_train, y_train, args)

    X_train, X_val, X_test = norm_xy_coords(X_train, X_val, X_test)

    print('Train pixel vals', X_train[0].min(), X_train[0].max())
    print('Val pixel vals', X_val[0].min(), X_val[0].max())
    print('Test pixel vals', X_test[0].min(), X_test[0].max())

    print('Train XY vals', X_train[1].min(), X_train[1].max())
    print('Val XY vals',X_val[1].min(), X_val[1].max())
    print('Test XY vals',X_test[1].min(), X_test[1].max())

    print('Train Z vals', y_train.min(), y_train.max())
    print('Val Z vals', y_val.min(), y_val.max())
    print('Test Z vals', y_test.min(), y_test.max())

    print('Train', X_train[0].shape, X_train[1].shape)
    print('Val', X_val[0].shape, X_val[1].shape)
    print('Test', X_test[0].shape, X_test[1].shape)

    train_data = get_dataset(X_train, y_train, args, True)
    val_data = get_dataset(X_val, y_val, args)
    test_data = get_dataset(X_test, y_test, args)

    train_data = preprocess_img_dataset(train_data)
    val_data = preprocess_img_dataset(val_data)
    test_data = preprocess_img_dataset(test_data)

    train_data = prep_dataset(train_data)
    val_data = prep_dataset(val_data)
    test_data = prep_dataset(test_data)    

    save_dataset(val_data, 'val', args)
    save_dataset(test_data, 'test', args)

    save_dataset(train_data, 'train', args)

    return train_data, val_data, test_data, locs



def load_ext_test_dataset(args):
    stacks = os.path.join(args['ext_test_dataset'], 'combined', 'stacks.ome.tif')
    locs = os.path.join(args['ext_test_dataset'], 'combined', 'locs.hdf')
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


def get_test_error(model, test_data):
    z = []
    for batch in test_data.as_numpy_iterator():
        z.append(batch[1])

    z = np.concatenate(z)

    preds = model.predict(test_data, batch_size=N_GPUS * 4096).squeeze()
    return mean_absolute_error(z, preds)


def main(args):
    if not args['regen_report']:
        train_data, val_data, test_data, locs = prepare_data(args)
        save_copy_training_script(args['outdir'])
        model = train_model(train_data, val_data, args)

    else:
        train_data = load_dataset('train', args)
        val_data = load_dataset('val', args)
        test_data = load_dataset('test', args)
        model = load_model(args)
        stacks, locs, zs = load_data(args)
        # Used to regen train/val/test split in locs file
        split_train_val_test(stacks, locs, zs)

    if args['ext_test_dataset']:
        wandb.log({'ext_test_mae':  get_test_error(model, test_data)})

    write_report(model, locs, train_data, val_data, test_data, args)

    print('Output in:', args['outdir'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='smlm_z3')

    parser.add_argument('-s', '--stacks', help='TIF file containing stacks in format N*Z*Y*X', default='./stacks.ome.tif')
    parser.add_argument('-l' ,'--locs', help='HDF5 locs file', default='./locs.hdf')
    parser.add_argument('-sc', '--stacks_config', help='JSON config file for stacks file (can be automatically found if in same dir)', default='./stacks_config.json')
    # parser.add_argument('-zstep', '--zstep', help='Z step in stacks (in nm)', default=10, type=int)
    parser.add_argument('-zrange', '--zrange', help='Z to model (+-val) in nm', default=1000, type=int)
    # parser.add_argument('-m', '--pretrained-model', help='Start training from existing model (path)')
    parser.add_argument('-o', '--outdir', help='Output directory', default='./out')

    parser.add_argument('--debug', action='store_true', help='Train on subset of data for fewer iterations')
    parser.add_argument('--seed', default=42, type=int, help='Random seed (for consistent results)')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size (per GPU)', default=1024)
    parser.add_argument('--aug-brightness', type=float, help='Brightness', default=0)
    parser.add_argument('--aug-gauss', type=float, help='Gaussian', default=0)
    parser.add_argument('--norm')
    parser.add_argument('--aug-poisson-lam', type=float, help='Poisson noise lam', default=0)

    parser.add_argument('--dense1', type=int, default=128)
    parser.add_argument('--dense2', type=int, default=64)
    parser.add_argument('--architecture', default='vit_b16')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    parser.add_argument('--dataset', help='Dataset type, used for wandb', default='unknown')
    parser.add_argument('--system', help='Optical system', default='unknown')

    parser.add_argument('--regen-report', action='store_true', help='Regen only training report from existing dir')
    parser.add_argument('--ext-test-dataset')
    parser.add_argument('--pretrained-model')

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

    if not args['batch_size']:
        args['batch_size'] = 512 * N_GPUS

    return args


def init_wandb(args):
    code_dir = os.path.dirname(os.path.normpath(__file__))
    wandb.init(
        project=args['project'],
        config = {
            'system': args['system'],
            'dataset': os.path.basename(os.path.normpath(args['dataset'])),
            'learning_rate': args['learning_rate'],
            'architecture': args['architecture'],
            'batch_size': args['batch_size'],
            'n_gpus': N_GPUS,
            'aug_brightness': args['aug_brightness'],
            'aug_gauss': args['aug_gauss'],
            'aug_poisson_lam': args['aug_poisson_lam'],
            'norm': args['norm']
        },
        settings=wandb.Settings(code_dir=code_dir)
    )

if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args['outdir'], exist_ok=True)

    tf.keras.utils.set_random_seed(args['seed'])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()

    init_wandb(args)
    main(args)
    wandb.finish()