#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys, os
import json
import shutil

cwd = os.path.dirname(__file__)
sys.path.append(cwd)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

VERSION = '0.13'
CHANGE_NOTES = 'Reduced brightness range, increased n aug'

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util.util import grid_psfs, norm_zero_one
from tifffile import imread
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Resizing, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, layers
from vit_keras import vit



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error

import joblib
import gc

N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))



# stacks = './stacks.ome.tif'
# locs = './locs.hdf'
# Z_STEP = 10
# zrange = 1000


def load_data(args):

    psfs = imread(args['stacks'])[:, :, :, :, np.newaxis]
    locs = pd.read_hdf(args['locs'], key='locs')
    locs['idx'] = np.arange(locs.shape[0])
    # idx = (xlim[0] < all_locs['x']) & (all_locs['x'] < xlim[1]) & (ylim[0] < all_locs['y']) & (all_locs['y'] < ylim[1])
    # locs = all_locs[idx]
    # psfs = all_psfs[locs['idx']]

    ys = []
    for offset in locs['offset']:
        zs = ((np.arange(psfs.shape[1])) * args['zstep']) - offset
        ys.append(zs)

    ys = np.array(ys)

    if args['debug']:
        idx = np.arange(psfs.shape[0])
        np.random.seed(42)
        idx = np.random.choice(idx, 2000)
        idx = idx[0:100]
        psfs = psfs[idx]
        locs = locs.iloc[idx]
        ys = ys[idx]

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
# In[10]:


# Withold some PSFs for evaluation

def split_train_val_test(psfs, locs, ys):

    idx = np.arange(psfs.shape[0])

    train_idx, test_idx = train_test_split(idx, train_size=0.9, random_state=args['seed'], stratify=locs['group'])

    _train_val_psfs = psfs[train_idx]
    test_psfs = psfs[test_idx]

    _train_val_ys = ys[train_idx]
    test_ys = ys[test_idx]

    # train_fov_groups = locs['group'].to_numpy()[train_idx]

    train_val_coords = locs[['x', 'y']].to_numpy()[train_idx]
    test_coords = locs[['x', 'y']].to_numpy()[test_idx]

    ds_cls = np.zeros((psfs.shape[0]), dtype=object)
    ds_cls[train_idx] = 'train/val'
    ds_cls[test_idx] = 'test'
    locs['ds'] = ds_cls
    plt.rcParams['figure.figsize'] = [5, 5]

    groups = np.repeat(np.arange(len(train_idx))[:, np.newaxis], psfs.shape[1], axis=1).flatten()

    coords = np.repeat(train_val_coords[:, :, np.newaxis], psfs.shape[1], axis=0)

    train_val_psfs = np.concatenate(_train_val_psfs)
    train_val_ys = np.concatenate(_train_val_ys)
    split_idx = np.arange(train_val_psfs.shape[0])

    train_idx, val_idx = train_test_split(split_idx, train_size=0.9, random_state=args['seed'], stratify=groups)

    train_psfs = train_val_psfs[train_idx]
    train_ys = train_val_ys[train_idx][:, np.newaxis]
    train_coords = coords[train_idx].squeeze()

    val_psfs = train_val_psfs[val_idx]
    val_ys = train_val_ys[val_idx][:, np.newaxis]
    val_coords = coords[val_idx].squeeze()

    test_psfs = psfs[test_idx]
    test_ys = ys[test_idx]
    test_groups = np.repeat(np.arange(len(test_psfs))[:, np.newaxis], test_psfs.shape[1], axis=1)
    test_groups = np.concatenate(test_groups)

    test_coords = np.repeat(test_coords[:, :, np.newaxis], test_psfs.shape[1], axis=0).squeeze()

    test_psfs = np.concatenate(test_psfs)
    test_ys = np.concatenate(test_ys)[:, np.newaxis]

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



def aug_train_data(X_train, y_train, args):
    X_train[0] = X_train[0].astype(float)
    AUG_RATIO = 4
    MAX_TRANSLATION_PX = 1
    MAX_GAUSS_NOISE = 0.005
    img_size = X_train[0].shape[1]

    aug_pipeline = Sequential([
        layers.GaussianNoise(stddev=MAX_GAUSS_NOISE*X_train[0].max(), seed=args['seed']),
        layers.RandomTranslation(MAX_TRANSLATION_PX/img_size, MAX_TRANSLATION_PX/img_size, seed=args['seed']),
        layers.RandomBrightness([-0.05, 0.05], [X_train[0].min(), X_train[0].max()], seed=args['seed']),
    ])
    
    idx = np.random.randint(0, X_train[0].shape[0], size=int(AUG_RATIO*X_train[0].shape[0]))

    aug_psfs = aug_pipeline(X_train[0][idx].copy(), training=True).numpy()
    aug_coords = X_train[1][idx]
    aug_z = y_train[idx]

    stdevs = np.std(aug_psfs, axis=(1,2,3))
    idx2 = np.argwhere(stdevs!=0).squeeze()
    aug_psfs = aug_psfs[idx2]
    aug_coords = aug_coords[idx2]
    aug_z = aug_z[idx2]

    train_psfs = np.concatenate([aug_psfs, X_train[0]])
    train_coords = np.concatenate([aug_coords, X_train[1]])
    train_zs = np.concatenate([aug_z, y_train])

    X_train = [train_psfs, train_coords]
    y_train = train_zs
    del aug_pipeline
    X_train[0] = X_train[0].astype(np.uint16)
    return X_train, y_train

def norm_images(X_train, X_val, X_test, args):
    datagen = ImageDataGenerator(
        rescale=1.0/65336.0,
        samplewise_center=False,
        samplewise_std_normalization=False,
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)

    print('Fitting datagen...')
    datagen.fit(X_train[0])
    print('Fitted')

    X_train[0] = datagen.standardize(X_train[0].astype(float))
    X_val[0] = datagen.standardize(X_val[0].astype(float))
    X_test[0] = datagen.standardize(X_test[0].astype(float))

    outpath = os.path.join(args['outdir'], 'datagen.gz')
    joblib.dump(datagen, outpath)

    return X_train, X_val, X_test


def norm_xy_coords(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train[1] = scaler.fit_transform(X_train[1])
    X_val[1] = scaler.transform(X_val[1])
    X_test[1] = scaler.transform(X_test[1])

    outpath = os.path.join(args['outdir'], 'scaler.save')
    joblib.dump(scaler, outpath) 

    return X_train, X_val, X_test


def get_dataset(X, y, batch_size, shuffle=False):
    images, coords = X
    img_ds = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
    coords_ds = tf.data.Dataset.from_tensor_slices(coords.astype(np.float32))
    labels_ds = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))

    x_ds = tf.data.Dataset.zip(img_ds, coords_ds)
    ds = tf.data.Dataset.zip(x_ds, labels_ds)
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=int(batch_size*1.5))
    print('Created dataset')
    return ds


image_size = 64
imshape = (image_size, image_size)
img_preprocessing = Sequential([
    Resizing(*imshape),
    Lambda(tf.image.grayscale_to_rgb)
])


def apply_rescaling(dataset):
    def _apply_rescaling(x, y):
        x = [x[0], x[1]]
        x[0] = img_preprocessing(x[0])
        return tuple(x), y

    return dataset.map(lambda x, y: _apply_rescaling(x, y), num_parallel_calls=tf.data.AUTOTUNE)

def prep_tf_dataset(dataset):
    return dataset.cache().prefetch(tf.data.AUTOTUNE)

def save_dataset(dataset, name, args):
    dataset.save(os.path.join(args['outdir'], name))

# Vision transformer training

# Assuming your input images have size (image_size, image_size, num_channels)
num_channels = 3
num_classes = 1  # Regression task, predicting a single continuous value

def get_model():
   # Create the Vision Transformer model using the vit_keras library
   inputs = Input(shape=(image_size, image_size, num_channels))
   
   coords_input = layers.Input((2,))
   x_coords = layers.Dense(64)(coords_input)
   
   x_coords = layers.Dense(64)(x_coords)
   
   
   vit_model = vit.vit_b16(image_size=image_size, 
                           activation='sigmoid',
                           pretrained=True,
                           include_top=False,
                           pretrained_top=False)
   
   x = vit_model(inputs)
   # Add additional layers for regression prediction
   x = Flatten()(x)
   x = tf.concat([x, x_coords], axis=-1)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.5)(x)
   x = Dense(64, activation='relu')(x)
   x = Dropout(0.5)(x)
   regression_output = Dense(num_classes, activation='linear')(x)  # Linear activation for regression
   model = Model(inputs=[inputs, coords_input], outputs=regression_output)

   return model

def train_model(train_data, val_data, args):
    epochs = 3 if args['debug'] else 5000
    lr = 0.0001
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
        model = get_model()

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer=optimizers.AdamW(learning_rate=lr), metrics=['mean_absolute_error'])



    callbacks = [
        ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
                        patience=25, verbose=True, mode='min', min_delta=1, min_lr=1e-6,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=50,
                    verbose=True, min_delta=1, restore_best_weights=True),
    ]

    try:
        history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks, shuffle=True, verbose=True)
    except KeyboardInterrupt:
        print('Interrupted training')
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

    return model


def write_report(model, train_data, val_data, test_data, args):

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

    # Check output on all stacks

    tbl_data = []
    report_data = {
        'code_version': VERSION,
        'change_notes': CHANGE_NOTES
    }

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

        report_data[dirname+'_mae'] = float(mean_absolute_error(preds.squeeze(), z.squeeze()))

        fname = f'{dirname}_all_preds'
        plt.figure(figsize=(12, 10), dpi=80)
        plt.title(fname)
        plt.scatter(z, preds)
        plt.savefig(os.path.join(args['outdir'], 'results', dirname, fname) + '.png')

        coords2 = ['_'.join(x.astype(str)) for x in coords]

        for num, c in tqdm(enumerate(set(coords2)), total=len(set(coords2))):
            idx = [i for i, val in enumerate(coords2) if val==c]
            idx = sorted(idx, key=lambda i: z.squeeze()[i])
            z_vals = z.squeeze()[idx]
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
            ax1.set_ylabel('estimated pred z (nm)')
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

            ax5.plot(group_images.max(axis=(1,2,3)))
            ax5.set_title('Max normalised pixel intensity over z')
            ax5.set_xlabel('z (frame)')
            ax5.set_ylabel('pixel intensity')    
                
            plt.savefig(os.path.join(args['outdir'], 'results', dirname, f'{dirname}_bead') + f'_{num}.png')
            plt.close()

            if dirname == 'test':
                adj_mae, offset, zs, error = bestfit_error(z_vals, pred_vals)
                adj_mae = str(round(adj_mae, 2))
                offset_fmt = str(round(offset, 2))
                ax1.set_title(f'MAE: {mae}nm, corrected MAE: {adj_mae}nm, offset: {offset_fmt}nm')
                mae = adj_mae
            else:
                offset = 0

            tbl_data.append((dirname, num,  gcoords[0][0], gcoords[1][0], mae, min(error), max(error),offset))

    df = pd.DataFrame(tbl_data, columns=['dataset', 'id', 'x', 'y', 'mae', 'min_error', 'max_error', 'offset'])
    df.to_csv(os.path.join(args['outdir'], 'results', 'results.csv'))



    with open(os.path.join(args['outdir'], 'results', 'report.json'), 'w') as fp:
        json_dumps_str = json.dumps(report_data, indent=4)
        print(json_dumps_str, file=fp)


def save_copy_training_script(outdir):
    outpath = os.path.join(outdir, 'train_model.py.bak')
    shutil.copy(os.path.abspath(__file__), outpath)


def main(args):
    stacks, locs, zs = load_data(args)
    locs = stratify_data(locs, args)
    train, val, test = split_train_val_test(stacks, locs, zs)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = filter_zranges(train, val, test, args)

    X_train, y_train = aug_train_data(X_train, y_train, args)
    X_train, X_val, X_test = norm_images(X_train, X_val, X_test, args)
    X_train, X_val, X_test = norm_xy_coords(X_train, X_val, X_test)


    print(X_train[0].min(), X_train[0].max())
    print(X_val[0].min(), X_val[0].max())
    print(X_test[0].min(), X_test[0].max())

    print(X_train[1].min(), X_train[1].max())
    print(X_val[1].min(), X_val[1].max())
    print(X_test[1].min(), X_test[1].max())

    print(X_train[0].shape, X_train[1].shape)
    print(X_val[0].shape, X_val[1].shape)
    print(X_test[0].shape, X_test[1].shape)


    train_data = get_dataset(X_train, y_train, args['batch_size'], True)
    val_data = get_dataset(X_val, y_val, args['batch_size'])
    test_data = get_dataset(X_test, y_test, args['batch_size'])

    train_data = apply_rescaling(train_data)
    val_data = apply_rescaling(val_data)
    test_data = apply_rescaling(test_data)
    gc.collect()

    train_data = prep_tf_dataset(train_data)
    val_data = prep_tf_dataset(val_data)
    test_data = prep_tf_dataset(test_data)    

    save_dataset(val_data, 'val', args)
    save_dataset(test_data, 'test', args)

    save_dataset(train_data, 'train', args)


    model = train_model(train_data, val_data, args)
    write_report(model, train_data, val_data, test_data, args)

    print('Output in:', args['outdir'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stacks', help='TIF file containing stacks in format N*Z*Y*X', default='./stacks.ome.tif')
    parser.add_argument('-l' ,'--locs', help='HDF5 locs file', default='./locs.hdf')
    parser.add_argument('-sc', '--stacks-config', help='JSON config file for stacks file (can be automatically found if in same dir)', default='./stacks_config.json')
    # parser.add_argument('-zstep', '--zstep', help='Z step in stacks (in nm)', default=10, type=int)
    parser.add_argument('-zrange', '--zrange', help='Z to model (+-val) in nm', default=1000, type=int)
    # parser.add_argument('-m', '--pretrained-model', help='Start training from existing model (path)')
    parser.add_argument('-o', '--outdir', help='Output directory', default='./out')

    parser.add_argument('--debug', action='store_true', help='Train on subset of data for fewer iterations')
    parser.add_argument('--seed', default=42, type=int, help='Random seed (for consistent results)')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size (per GPU)')

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
    return args


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args['outdir'], exist_ok=True)

    save_copy_training_script(args['outdir'])

    if not args['batch_size']:
        args['batch_size'] = 512 * N_GPUS

    main(args)