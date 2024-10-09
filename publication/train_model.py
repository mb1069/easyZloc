import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import scipy.optimize as opt
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger
import wandb
from tqdm import tqdm
import pandas as pd
from tifffile import imread
from util.util import grid_psfs, load_dataset, save_dataset, load_model, preprocess_img_dataset, ScaledMeanAbsoluteError
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import argparse
import sys
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from vit_keras import vit

cwd = os.path.dirname(__file__)
sys.path.append(cwd)
matplotlib.use('Agg')


N_GPUS = max(1, len(tf.config.experimental.list_physical_devices("GPU")))

AVAILABLE_MODELS = {
    'vit_b16': vit.vit_b16,
    'vit_b32': vit.vit_b32,
    'vit_l16': vit.vit_l16,
    'vit_l32': vit.vit_l32,
    'mobilenet': keras.applications.MobileNetV3Small,
    'mobilenet_large': keras.applications.MobileNetV3Large,
    'vgg': keras.applications.VGG19,
    'resnet': keras.applications.ResNet50V2,
    'resnet_large': keras.applications.ResNet101V2,
    'efficientnet_v2s': keras.applications.EfficientNetV2S
}


def load_data(args):
    # Load bead stacks, locs
    psfs = imread(args['stacks'])[:, :, :, :, np.newaxis].astype(np.float32)
    locs = pd.read_hdf(args['locs'], key='locs')
    locs['idx'] = np.arange(locs.shape[0])

    # Sub-sample data for debugging purposes
    if args['debug']:
        idx = np.arange(psfs.shape[0])
        np.random.seed(42)
        idx = np.random.choice(idx, 1000)
        psfs = psfs[idx]
        locs = locs.iloc[idx]

    ys = []
    for offset in locs['offset']:
        zs = ((np.arange(psfs.shape[1])) * args['zstep']) - offset
        ys.append(zs)

    ys = np.array(ys)

    return psfs, locs, ys


def stratify_data(locs, args):
    # Split FOV into sections to avoid mis-matches between train/val/test datasets
    def cart2pol(xy):
        x, y = xy
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

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


def split_train_val_test(psfs, locs, ys, args):
    # Withold some PSFs to stop training early (validation) and test the model
    def get_sub_ds(psfs, xy_coords, ys, idx):
        psfs_idx = psfs[idx]
        xy_coords_idx = xy_coords[idx].squeeze()
        xy_coords_idx = np.repeat(xy_coords_idx[:, :, np.newaxis], psfs_idx.shape[1], axis=0).squeeze()

        psfs_idx = np.concatenate(psfs_idx)
        ys_idx = np.concatenate(ys[idx])[:, np.newaxis]
        return psfs_idx, xy_coords_idx, ys_idx

    idx = np.arange(psfs.shape[0])

    # Split Train/Test first
    train_idx, test_idx = train_test_split(idx, train_size=0.9, random_state=args['seed'])

    # Split Train -> Train/Val
    # ~81% train, 9% val, 10% test
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

    print('Train data', train_psfs.shape, train_coords.shape, train_ys.shape)
    print('Val data', val_psfs.shape, val_coords.shape, val_ys.shape)
    print('Test data', test_psfs.shape, test_coords.shape, test_ys.shape)

    return (train_psfs, train_coords, train_ys), (val_psfs, val_coords, val_ys), (test_psfs, test_coords, test_ys)


def filter_zranges(train_data, val_data, test_data, args):
    # Filter bead data to exclude PSFs outside of z range
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


def norm_xy_coords(X_train, X_val, X_test, args):
    scaler = MinMaxScaler(feature_range=(0, 1))
    full_fov_data = [[0, 0], [args['xsize'], args['ysize']]]
    scaler.fit(full_fov_data)
    X_train[1] = scaler.transform(X_train[1])
    X_val[1] = scaler.transform(X_val[1])
    X_test[1] = scaler.transform(X_test[1])

    for dataset in (X_train[1], X_val[1], X_test[1]):
        try:
            assert dataset.min() >= 0
            assert dataset.max() <= 1
        except AssertionError:
            raise AssertionError('XY coordinates out of range, check training data')
    outpath = os.path.join(args['outdir'], 'scaler.save')
    joblib.dump(scaler, outpath)

    return X_train, X_val, X_test


def norm_z_coords(y_train, y_val, y_test, args):
    # Norm Z coords to [-1, 1]
    rescale = args['zrange']
    y_train = y_train / rescale
    y_val = y_val / rescale
    y_test = y_test / rescale
    return y_train, y_val, y_test


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


def get_dataset(X, y):
    # Convert images,coords and z positions into tf.Data objects for efficient processing / loading
    images, coords = X
    img_ds = tf.data.Dataset.from_tensor_slices(images.astype(np.float32), name='images')
    coords_ds = tf.data.Dataset.from_tensor_slices(coords.astype(np.float32), name='xy_coords')
    labels_ds = tf.data.Dataset.from_tensor_slices(y.astype(np.float32), name='z_coords')

    x_ds = tf.data.Dataset.zip(img_ds, coords_ds)
    ds = tf.data.Dataset.zip(x_ds, labels_ds)
    return ds


def prep_dataset(ds, args, shuffle=False):
    # Pre-training steps on datasets
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=int(args['batch_size'] * 1.5), seed=args['seed'])
    ds = ds.batch(args['batch_size'])
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_model(args):
    # Construct model
    imshape = (args['image_size'], args['image_size'], 1)

    img_input = Input(shape=imshape, name='img_in')
    # RGB used for compatibility with pretrained weights
    img_rgb = tf.image.grayscale_to_rgb(img_input)

    # Optionally add augmentation layers
    if not (args['aug_gauss'] or args['aug_brightness']):
        img_aug_out = img_rgb
    else:
        extra_aug = Sequential([], name='extra_aug')
        if args['aug_gauss']:
            extra_aug.add(layers.GaussianNoise(stddev=args['aug_gauss'], seed=args['seed']))

        if args['aug_brightness']:
            extra_aug.add(layers.RandomBrightness(args['aug_brightness'], value_range=[0, 1], seed=args['seed']))

        img_aug_out = extra_aug(img_rgb)

    # Rescale images from [0, 1] to [-1, 1] for better learning efficiency with imagenet weights
    img_aug_out = layers.Rescaling(scale=2, offset=-1)(img_aug_out)

    # XY coordinate input and pre-processing
    coords_input = layers.Input((2,), name='xy_coords_in')
    x_coords = layers.Dense(64)(coords_input)
    x_coords = layers.Dense(64)(x_coords)

    model_version = AVAILABLE_MODELS[args['architecture']]
    if 'vit_' in args['architecture']:
        feat_model = model_version(image_size=args['image_size'],
                                   activation='sigmoid',
                                   pretrained=True,
                                   include_top=False,
                                   pretrained_top=False)
    elif 'resnet' in args['architecture']:
        rgb_shape = (args['image_size'], args['image_size'], 3)
        feat_model = model_version(input_shape=rgb_shape,
                                   weights='imagenet',
                                   include_top=False)
    else:
        rgb_shape = (args['image_size'], args['image_size'], 3)
        feat_model = model_version(input_shape=rgb_shape,
                                   weights='imagenet',
                                   include_top=False,
                                   include_preprocessing=False)

    x_img = feat_model(img_aug_out)
    x_img = Flatten()(x_img)
    # Merge embeddings from img and coords sections of network
    x = tf.concat([x_img, x_coords], axis=-1)
    x = Dense(args['dense1'], activation='gelu')(x)
    x = Dropout(0.5)(x)
    # Optional second dense layer
    if args['dense2'] != 0:
        x = Dense(args['dense2'], activation='gelu')(x)
        x = Dropout(0.5)(x)
    regression_output = Dense(1, activation='tanh')(x)  # Linear activation for regression
    model = Model(inputs=[img_input, coords_input], outputs=regression_output)

    # Sub-section of model to inspect augmentation behaviour
    aug_model = Model(inputs=img_input, outputs=img_aug_out)

    save_model_plot(model, args['outdir'])
    return model, aug_model


def bestfit_error(z_true, z_pred):
    # Removes a constant bias (i.e alignment in Z error) from a set of data, then evaluate MAE
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = mean_absolute_error(y_fit, y)
    return error, popt[0], y_fit, abs(y_fit - y)


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
    epochs = 3 if args['debug'] else 1000
    lr = args['learning_rate']
    print(f'N epochs: {epochs}')

    # multi-GPU support
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        if args['pretrained_model']:
            model = keras.models.load_model(args['pretrained_model'])
        else:
            model, aug_model = get_model(args)
            gen_example_aug_imgs(aug_model, train_data, args)

        opt = optimizers.AdamW(learning_rate=lr)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ScaledMeanAbsoluteError(scale_nm=args['zrange'])])


    callbacks = [
        WandbMetricsLogger(),
        ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
                          patience=5, verbose=True, mode='min', min_delta=1, min_lr=1e-8, cooldown=5),
        EarlyStopping(monitor='val_mean_absolute_error', patience=20,
                      verbose=True, min_delta=1, restore_best_weights=True),
    ]

    try:
        history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks, shuffle=True, verbose=True)
    except KeyboardInterrupt:
        print('\nInterrupted training\n')
        history = None

    model.save(args['model_path'])

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
        print('Finished plotting history.')

    return model


def write_report(model, locs, train_data, val_data, test_data, args):
    # Writes an evaluation of the model against validation and test data
    # Training data is disabled as this is very slow to run and often un-necessary
    print('Writing report')

    tbl_data = []
    report_data = {
        'args': args,
        'wandb_run_id': wandb.run.id,
    }

    os.makedirs(os.path.join(args['outdir'], 'results'), exist_ok=True)

    sns.scatterplot(data=locs, x='x', y='y', hue='ds')
    plt.savefig(os.path.join(args['outdir'], 'results', 'data_split') + '.png')
    plt.close()
    print('Wrote scatterplot')

    for dirname, ds in [
        # ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]:
        print(f'Writing results for {dirname} dataset')
        os.makedirs(os.path.join(args['outdir'], 'results', dirname), exist_ok=True)
        images = []
        coords = []
        z = []
        # Get all images, xy coordinates and true z coords from the dataset
        for batch in ds.as_numpy_iterator():
            images.append(batch[0][0])
            coords.append(batch[0][1])
            z.append(batch[1])

        images = np.concatenate(images)
        coords = np.concatenate(coords)
        z = np.concatenate(z) * args['zrange']

        # Predict entire dataset's z coords
        preds = model.predict(ds, batch_size=N_GPUS * 4096).squeeze() * args['zrange']
        print('Finished predicting...')

        errors = abs(preds - z.squeeze())

        fname = f'{dirname}_all_preds'
        fig = plt.figure(layout="constrained", figsize=(12, 10), dpi=80)
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Plot 1 - true vs predicted Z vals
        plt.title(fname)
        ax1.scatter(z, preds)
        ax1.set_xlabel('True Z position (nm)')
        ax1.set_ylabel('Predicted Z pos (nm)')
        ds_df = pd.DataFrame.from_dict({'x': coords[:, 0], 'y': coords[:, 1], 'error': errors})
        ds_df = ds_df.groupby(['x', 'y']).mean().reset_index()

        # Plots to check bias with x/y location 
        sns.scatterplot(data=ds_df, x='x', y='y', hue='error', ax=ax2)
        sns.scatterplot(data=ds_df, x='x', y='error', ax=ax3)
        sns.scatterplot(data=ds_df, x='y', y='error', ax=ax4)
        img_fname = fname + '.png'
        outpath = os.path.join(args['outdir'], 'results', dirname, img_fname)
        plt.savefig(outpath)
        plt.close()
        wandb.log({img_fname: wandb.Image(outpath)})
        print('Writing plots...')

        coords2 = ['_'.join(x.astype(str)) for x in coords]
        ds_true_vals = []
        ds_pred_vals = []

        # Group beads by xy coordinate, eg by physical bead
        for num, c in tqdm(enumerate(sorted(set(coords2))), total=len(set(coords2))):
            idx = [i for i, val in enumerate(coords2) if val == c]
            idx = sorted(idx, key=lambda i: z.squeeze()[i])
            z_vals = z.squeeze()[idx]
            ds_true_vals.extend(z_vals)
            pred_vals = preds[idx]
            group_images = images[idx]
            mae_error = abs(z_vals - pred_vals)
            rmse_error = root_mean_squared_error(z_vals, pred_vals)
            mae = str(round(mae_error.mean(), 2))
            rmse = str(round(rmse_error.mean(), 2))

            fig = plt.figure(layout="constrained", figsize=(12, 10), dpi=80)
            gs = plt.GridSpec(4, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2:4, 1])


            # Plot 1 - true vs predicted z pos
            fig.suptitle(f'Bead: {num}')
            ax1.scatter(z_vals, pred_vals)
            ax1.plot([-1000, 1000], [-1000, 1000], c='orange')
            ax1.set_xlabel('estimated true z (nm)')
            ax1.set_ylabel('Predicted z (nm)')
            ax1.set_title(f'RMSE: {rmse}nm')

            # Plot 2 - images in order from Z stack
            ax2.imshow(grid_psfs(group_images.mean(axis=-1)).T)
            ax2.set_title('Ordered by frame')
            ax2.set_axis_off()

            # Plot 3 - images in order of prediction error (check for bias)
            ax3.imshow(grid_psfs(group_images[np.argsort(mae_error)].mean(axis=-1)).T)
            ax3.set_title('Ordered by increasing prediction error')
            ax3.set_xlabel(f'min error: {str(round(min(mae_error), 2))}, max error: {str(round(max(mae_error), 2))}')
            ax3.set_axis_off()

            ax4.scatter(coords[:, 0], coords[:, 1])
            gcoords = list(map(lambda x: [float(x)], c.split('_')))
            ax4.scatter(gcoords[0], gcoords[1])
            ax4.set_xlabel('x (nm)')
            ax4.set_ylabel('y (nm)')
            ax4.set_title('Position of bead within dataset')
            ax4.axis('equal')

            # wandb.log({img_fname: wandb.Image(outpath)})

            if dirname != 'train':
                # Ignore constant error in Z prediction errors, as errors in bead alignment methodology occur
                adj_mae, offset, zs, error = bestfit_error(z_vals, pred_vals)
                pred_vals -= offset
                adj_mae = str(round(adj_mae, 2))
                offset_fmt = str(round(offset, 2))
                ax1.set_title(f'RMSE: {rmse}nm, corrected MAE: {adj_mae}nm, offset: {offset_fmt}nm')
                mae = adj_mae
            else:
                offset = 0

            img_fname = f'{dirname}_bead' + f'_{num}.png'
            outpath = os.path.join(args['outdir'], 'results', dirname, img_fname)
            plt.savefig(outpath)

            plt.close()

            ds_pred_vals.extend(pred_vals)
            tbl_data.append((dirname, num, gcoords[0][0], gcoords[1][0], mae, rmse, min(error), max(error), offset))

            if num == 10 and dirname == 'train':
                break

        ds_mae = mean_absolute_error(ds_true_vals, ds_pred_vals)
        ds_rmse = root_mean_squared_error(ds_true_vals, ds_pred_vals)
        report_data[dirname + '_mae'] = float(ds_mae)
        report_data[dirname + '_rmse'] = float(ds_rmse)
        print(dirname, report_data[dirname + '_mae'])

        wandb.run.summary[dirname + '_mae'] = float(ds_mae)

    df = pd.DataFrame(tbl_data, columns=['dataset', 'id', 'x', 'y', 'mae', 'rmse', 'min_error', 'max_error', 'offset'])
    df.to_csv(os.path.join(args['outdir'], 'results', 'results.csv'))

    with open(os.path.join(args['outdir'], 'results', 'report.json'), 'w') as fp:
        json_dumps_str = json.dumps(report_data, indent=4)
        print(json_dumps_str, file=fp)


def add_shot_noise(image, min_poisson_lam):

    factor = np.random.uniform(min_poisson_lam, 1000)

    # Convert image to photon counts
    photon_count = image * factor

    # Apply Poisson noise
    noisy_image = np.random.poisson(photon_count.astype(int))

    # Convert back to image scale
    return noisy_image / factor


def aug_training_data(X_train, y_train, min_poisson_lam):
    # Baseline data augmentation using shot noise
    print('Augmenting training data...')
    print('Data to augment', X_train[0].shape, X_train[1].shape, y_train.shape)
    aug_imgs = []
    aug_xy = []
    aug_z = []
    n_aug_points = X_train[0].shape[0] * 2
    idx = np.random.randint(0, X_train[0].shape[0], size=n_aug_points)
    for i in tqdm(idx):
        img = X_train[0][i]
        xy = X_train[1][i]
        z = y_train[i]
        noisy_img = add_shot_noise(img, min_poisson_lam)
        aug_imgs.append(noisy_img)
        aug_xy.append(xy)
        aug_z.append(z)

    aug_imgs = np.stack(aug_imgs)
    aug_xy = np.stack(aug_xy)
    aug_z = np.stack(aug_z)
    X_train = np.concatenate((X_train[0], aug_imgs)), np.concatenate((X_train[1], aug_xy))
    y_train = np.concatenate((y_train, aug_z))

    return X_train, y_train


def prepare_data(args):
    stacks, locs, zs = load_data(args)

    locs = stratify_data(locs, args)
    train, val, test = split_train_val_test(stacks, locs, zs, args)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = filter_zranges(train, val, test, args)

    X_train, X_val, X_test = norm_xy_coords(X_train, X_val, X_test, args)
    y_train, y_val, y_test = norm_z_coords(y_train, y_val, y_test, args)

    X_train, y_train = aug_training_data(X_train, y_train, args['aug_poisson_lam'])

    print('Train pixel vals', X_train[0].min(), X_train[0].max())
    print('Val pixel vals', X_val[0].min(), X_val[0].max())
    print('Test pixel vals', X_test[0].min(), X_test[0].max())

    print('Train XY vals', X_train[1].min(), X_train[1].max())
    print('Val XY vals', X_val[1].min(), X_val[1].max())
    print('Test XY vals', X_test[1].min(), X_test[1].max())

    print('Train Z vals', y_train.min(), y_train.max())
    print('Val Z vals', y_val.min(), y_val.max())
    print('Test Z vals', y_test.min(), y_test.max())

    print('Train', X_train[0].shape, X_train[1].shape)
    print('Val', X_val[0].shape, X_val[1].shape)
    print('Test', X_test[0].shape, X_test[1].shape)

    train_data = get_dataset(X_train, y_train)
    val_data = get_dataset(X_val, y_val)
    test_data = get_dataset(X_test, y_test)

    train_data = preprocess_img_dataset(train_data, args['image_size'], args['norm'])
    val_data = preprocess_img_dataset(val_data, args['image_size'], args['norm'])
    test_data = preprocess_img_dataset(test_data, args['image_size'], args['norm'])

    train_data = prep_dataset(train_data, args, True)
    val_data = prep_dataset(val_data, args)
    test_data = prep_dataset(test_data, args)

    save_dataset(val_data, 'val', args)
    save_dataset(test_data, 'test', args)
    # save_dataset(train_data, 'train', args)
    return train_data, val_data, test_data, locs


def get_test_error(model, test_data):
    z = []
    for batch in test_data.as_numpy_iterator():
        z.append(batch[1])

    z = np.concatenate(z)

    preds = model.predict(test_data, batch_size=N_GPUS * 4096).squeeze()
    return mean_absolute_error(z, preds)


def save_model_plot(model, outdir):
    outpath = os.path.join(outdir, 'architecture.png')
    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=outpath)


def main(args):
    if not args['regen_report']:
        train_data, val_data, test_data, locs = prepare_data(args)
        model = train_model(train_data, val_data, args)
    else:
        # Run model evaluation only
        # train_data = load_dataset('train', args)
        train_data = None
        val_data = load_dataset('val', args)
        test_data = load_dataset('test', args)
        model = load_model(args['model_path'])
        stacks, locs, zs = load_data(args)
        # Used to regen train/val/test split in locs file
        split_train_val_test(stacks, locs, zs, args)

    write_report(model, locs, train_data, val_data, test_data, args)

    print('Output in:', args['outdir'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='smlm_z')

    parser.add_argument('-s', '--stacks', help='TIF file containing stacks in format N*Z*Y*X', default='./stacks.ome.tif')
    parser.add_argument('-l', '--locs', help='HDF5 locs file', default='./locs.hdf')
    parser.add_argument('-sc', '--stacks_config', help='JSON config file for stacks file (can be automatically found if in same dir)', default='./stacks_config.json')
    parser.add_argument('--zrange', help='Z to model (+-val) in nm', default=1000, type=int)
    parser.add_argument('-m', '--pretrained-model', help='Start training from existing model (path)')
    parser.add_argument('-o', '--outdir', help='Output directory', default='./out')

    parser.add_argument('--debug', action='store_true', help='Train on subset of data for fewer iterations')
    parser.add_argument('--seed', default=42, type=int, help='Random seed (for consistent results)')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size (per GPU)', default=1024)
    parser.add_argument('--aug-brightness', type=float, help='Brightness', default=0)
    parser.add_argument('--aug-gauss', type=float, help='Gaussian', default=0)
    parser.add_argument('--norm', default='frame-min')
    parser.add_argument('--aug-poisson-lam', type=float, help='Poisson noise lam', default=0)

    parser.add_argument('--image_size', type=int, default=64, help='Resize PSFs to this size for model')
    parser.add_argument('--dense1', type=int, default=128)
    parser.add_argument('--dense2', type=int, default=64)
    parser.add_argument('--architecture', default='mobilenet', choices=sorted(list(AVAILABLE_MODELS.keys())))
    parser.add_argument('--activation')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    parser.add_argument('--dataset', help='Dataset type, used for wandb', default='unknown')
    parser.add_argument('--system', help='Optical system', default='unknown')

    parser.add_argument('--regen-report', action='store_true', help='Regen only training report from existing dir')

    args = vars(parser.parse_args())

    if not os.path.exists(args['stacks_config']):
        print('Could not find stacks-config file, checking in dir of stacks...')
        args['stacks-config'] = os.path.join(os.path.dirname(args['stacks']), 'stacks_config.json')
        if not os.path.exists(args['stacks_config']):
            print('Could not find stacks-config file')
            quit(1)

    # Load stacks config into args
    with open(args['stacks_config']) as f:
        d = json.load(f)
    args.update(d)

    for k, v in args.items():
        print(f'{k}: {v}')

    if not args['batch_size']:
        args['batch_size'] = 512 * N_GPUS
    return args


def init_wandb(args):
    print('initiating wandb...', end='')
    code_dir = os.path.dirname(os.path.normpath(__file__))
    wandb.init(
        project=args['project'],
        config={
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


def run_tool():
    args = parse_args()

    i = 1
    # If output dir already exists, add a postfix _n to avoid overwriting data
    base_outdir = args['outdir']
    while os.path.exists(args['outdir']) and not args['regen_report']:
        args['outdir'] = f'{base_outdir}_{i}'
        # shutil.rmtree(args['outdir'])
        i += 1
    print(f"Saving data to {args['outdir']}")
    os.makedirs(args['outdir'], exist_ok=True)

    tf.keras.utils.set_random_seed(args['seed']) 
    tf.config.experimental.enable_op_determinism()

    args['model_path'] = os.path.join(args['outdir'], './model')

    init_wandb(args)
    main(args)
    wandb.finish()


if __name__ == '__main__':
    run_tool()
