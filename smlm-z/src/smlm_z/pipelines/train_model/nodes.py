"""
This is a boilerplate pipeline 'classify'
generated using Kedro 0.18.4
"""
from sklearn.metrics import max_error, mean_absolute_error
from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import operator as op

import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm.keras import TqdmCallback
from smlm_z.extras.visualise import grid_psfs

from ..preprocessing.nodes import norm_images

def split_train_val_test(X: Tuple[np.array, np.array], y: np.array, parameters: Dict):
    X, y = chunk_data(X, y)

    train_size = parameters['train_size']
    test_size = parameters['test_size'] / (parameters['test_size']+parameters['val_size'])
    idx = np.arange(len(y))
    train_idx, other_idx = train_test_split(
        idx, train_size=train_size, random_state=parameters['random_seed'])
    val_idx, test_idx = train_test_split(
        other_idx, test_size=test_size, random_state=parameters['random_seed'])
    print(np.array(X[0]).shape, np.array(X[1]).shape)

    X_train = X[train_idx][0], X[train_idx][1]
    X_val = X[val_idx][0], X[val_idx][1]
    X_test = X[test_idx][0], X[test_idx][1]

    y_train = np.stack(y[train_idx])
    y_val = np.stack(y[val_idx])
    y_test = np.stack(y[test_idx])

    print(X_train.shape, y_train.shape)
    splits = np.zeros(y.shape[0])
    splits[train_idx] = 0
    splits[val_idx] = 1
    splits[test_idx] = 2

    idx_dict = {0: 'train', 1: 'val', 2: 'test'}
    func = np.vectorize(lambda v: idx_dict[v])
    splits = func(splits)
    return X_train, y_train, X_val, y_val, X_test, y_test, splits

def norm_psfs(X_train, X_val, X_test, X):
    train_psfs, train_coords = X_train
    val_psfs, val_coords = X_val
    test_psfs, test_coords = X_test
    all_psfs, all_coords = X

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    datagen.fit(train_psfs)

    train_psfs = datagen.standardize(train_psfs)
    val_psfs = datagen.standardize(val_psfs)
    test_psfs = datagen.standardize(test_psfs)
    all_psfs = datagen.standardize(all_psfs)

    return (train_psfs, train_coords), (val_psfs, val_coords), (test_psfs, test_coords), (all_psfs, all_coords)


def compile_model(model, lr):
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(
        learning_rate=lr), metrics=['mean_absolute_error'])
    print(model.summary())


def get_model(parameters):
    img_input_shape = parameters['model_input_shape']
    lr = parameters['training']['learning_rate']
    bound, n_channels = img_input_shape[0], img_input_shape[-1]
    img_input = Input(img_input_shape)
    coords_input = Input((2,))

    weights = 'imagenet' if n_channels == 3 else None
    print(f'Using weights {weights}')
    
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights=weights,
        pooling='max',
        input_shape=img_input_shape,
    )
    x = img_input
    augmentation = Sequential([
        tf.keras.layers.GaussianNoise(0.001, seed=parameters['random_seed']),
        # tf.keras.layers.RandomTranslation(0.05, 0.05, seed=parameters['random_seed']),
    ])
    x = augmentation(x)
    x = resnet(x)

    x = tf.concat((x, coords_input), axis=1)

    mlp = Sequential([
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1),
        # Dense(1, activation='sigmoid'),
        # Lambda(lambda x: ((x-0.5)*2)*parameters['z_range'])
    ])

    x = mlp(x)
    model = Model(inputs=(img_input, coords_input), outputs=x)
    print(model.summary())
    compile_model(model, lr)
    return model


def train_classifier(X_train: Tuple[np.array, np.array], y_train: np.array, X_val: Tuple[np.array, np.array], y_val: np.array, parameters: Dict, extra_callbacks=None):
    tf.keras.utils.set_random_seed(parameters['random_seed'])
    tf.random.set_seed(parameters['random_seed'])

    batch_size = parameters['training']['batch_size']
    epochs = parameters['training']['max_epochs']
    model = get_model(parameters)
    # model = tf.keras.models.load_model('/home/miguel/Projects/uni/phd/smlm_z/smlm-z/data/06_models/warmstart_model')
    # from keras import backend as K
    # K.set_value(model.optimizer.learning_rate, parameters['training']['learning_rate'])
    
    callbacks = [
        ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.1,
                          patience=50, verbose=True, mode='min', min_delta=1, min_lr=1e-7,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=250,
                      verbose=False, min_delta=1, restore_best_weights=True),
        TqdmCallback(verbose=1),
    ]
    if extra_callbacks is not None:
        callbacks += extra_callbacks
    try:
        history = model.fit(X_train, y_train, epochs=epochs, shuffle=True, verbose=False,
                            batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)
    except KeyboardInterrupt:
        history = model.history

    plt.rcParams['figure.figsize'] = [10, 5]
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['mean_absolute_error'], label='mse')
    ax1.plot(history.history['val_mean_absolute_error'], label='val_mse')
    ax1.set_yscale('log')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(history.history['lr'], label='lr', color='red')
    ax2.legend(loc=0)
    return model, fig


def chunk_data(X, y, data_splits=None):
    idx = np.argwhere(np.diff(y.squeeze())<0).squeeze()
    idx += 1
    idx = np.concatenate([[0], idx, [len(y)]])
    Xs = []
    ys = []
    if data_splits is not None:
        splits = []
    for i in range(0, len(idx)-1):
        s = idx[i]
        end = idx[i+1]
        Xs.append((X[0][s:end], X[1][s:end]))
        ys.append(y[s:end])
        if data_splits is not None:
            splits.append(data_splits[s:end])

    if data_splits is not None:
        return Xs, ys, splits
    return Xs, ys


def eval_classifier(model, X, y, X_train: Tuple[np.array, np.array], y_train: np.array, X_val: Tuple[np.array, np.array], y_val: np.array, X_test: Tuple[np.array, np.array], y_test: np.array, data_splits: np.array):

    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    all_metrics = {}
    figs = []
    error_scatters = []

    Xs, ys, data_splits = chunk_data(X, y, data_splits)
    pred_ys = []
    true_ys = []
    fig = plt.figure()
    for i, (X, y, s) in enumerate(zip(Xs, ys, data_splits)):
        pred_y = model.predict(X).squeeze()
        pred_ys.append(pred_y)
        true_ys.append(y)

        non_train_idx = np.argwhere(s!='train').squeeze()

        plt.plot(y, pred_y, label=str(i))
        non_train_y = y[non_train_idx]
        non_train_pred_y = pred_y[non_train_idx]
        plt.scatter(non_train_y, non_train_pred_y)
        if i % 5 == 4 or (i == len(X)-1):
            plt.plot([-1000, 1000], [-1000, 1000], 'r--')
            plt.xlabel('True Z')
            plt.ylabel('Pred Z')
            plt.title('Offset checkers')
            plt.legend()
            figs.append(fig)
            fig = plt.figure()


    for k, (x, y) in datasets.items():
        pred_y = model.predict(x).squeeze()
        metrics = {
            f'{k}_mae': mean_absolute_error(y, pred_y),
            f'{k}_max_error': max_error(y, pred_y),
        }
        all_metrics.update(metrics)
        fig = plt.figure()
        plt.scatter(y, pred_y)
        plt.plot([-1000, 1000], [-1000, 1000], color='red')
        plt.xlabel('true z [nm]')
        plt.ylabel('pred z [nm]')
        plt.title(k)
        figs.append(fig)

        fig = plt.figure()
        plt.title(k)
        errors = abs(pred_y.squeeze()-y.squeeze())
        sns.scatterplot(x=x[1][:, 0], y=x[1][:, 1], hue=errors)
        plt.xlabel('X')
        plt.ylabel('Y')
        error_scatters.append(fig)

        # Pixel val / error
        max_pixel_val = x[0].max(axis=(1, 2, 3))
        fig = plt.figure()
        plt.scatter(max_pixel_val, errors)
        plt.xlabel('max pixel val')
        plt.ylabel('error')
        plt.title(k)
        error_scatters.append(fig)

        # Min val / error
        min_pixel_val = x[0].min(axis=(1, 2, 3))
        fig = plt.figure()
        plt.scatter(min_pixel_val, errors)
        plt.xlabel('min pixel val')
        plt.ylabel('error')
        plt.title(k)
        error_scatters.append(fig)

        # Mean val / error
        mean_pixel_val = x[0].mean(axis=(1, 2, 3))
        fig = plt.figure()
        plt.scatter(mean_pixel_val, errors)
        plt.xlabel('min pixel val')
        plt.ylabel('error')
        plt.title(k)
        error_scatters.append(fig)
        
        # X
        x_val = x[1][:, 0]
        fig = plt.figure()
        plt.scatter(x_val, errors)
        plt.xlabel('X')
        plt.ylabel('error')
        plt.title(k)
        error_scatters.append(fig)

        # Y
        y_val = x[1][:, 1]
        fig = plt.figure()
        plt.scatter(y_val, errors)
        plt.xlabel('Y')
        plt.ylabel('error')
        plt.title(k)
        error_scatters.append(fig)

    all_metrics = {k: {'value': v, 'step': 0} for k, v in all_metrics.items()}

    return all_metrics, figs, error_scatters

def create_hist(data, xlabel):
    fig = plt.figure()
    sns.histplot(data, stat='probability', common_norm=False)
    plt.xlabel(xlabel)
    return fig

def create_boxplot(data, ylabel):
    fig = plt.figure()

    # sort keys and values together
    sorted_keys, sorted_vals = zip(*sorted(data.items(), key=op.itemgetter(0)))

    # almost verbatim from question
    sns.set(context='notebook', style='whitegrid')
    plt.xlabel('groups')
    plt.ylabel(ylabel)
    sns.boxplot(data=sorted_vals, width=.18)

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)
    plt.title(ylabel)
    return fig


def check_data(X_train, y_train, X_val, y_val, X_test, y_test):
    figs = []
    for k, v in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        plt.rcParams["figure.figsize"] = (7, 7)
        idx = np.random.choice(list(range(len(v[1]))), size=100)
        idx_imgs = v[0][0][idx]
        idx_ys = v[1][idx].squeeze()
        order = np.argsort(idx_ys)
        imgs = idx_imgs[order].mean(axis=-1)
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        im = ax0.imshow(grid_psfs(imgs))
        plt.title(k)
        fig.colorbar(im)

        ax1.plot(list(range(len(idx_ys))), idx_ys[order])

        figs.append(fig)



    # z coords
    z_data = {
        'train': y_train.squeeze(),
        'val': y_val.squeeze(),
        'test': y_test.squeeze()
    }
    figs.append(create_hist(z_data, 'z [nm]'))

    pixel_data = {
        'train': X_train[0].mean(axis=(1, 2, 3)),
        'val': X_val[0].mean(axis=(1, 2, 3)),
        'test': X_test[0].mean(axis=(1, 2, 3)),
    }
    figs.append(create_hist(pixel_data, 'mean pixel val'))

    pixel_data = {
        'train': X_train[0].max(axis=(1, 2, 3)),
        'val': X_val[0].max(axis=(1, 2, 3)),
        'test': X_test[0].max(axis=(1, 2, 3)),
    }
    figs.append(create_hist(pixel_data, 'max pixel val'))

    pixel_data = {
        'train': X_train[0].min(axis=(1, 2, 3)),
        'val': X_val[0].min(axis=(1, 2, 3)),
        'test': X_test[0].min(axis=(1, 2, 3)),
    }
    figs.append(create_hist(pixel_data, 'min pixel val'))

    # Coord values
    x_data = {
        'train': X_train[1][:, 0].flatten(),
        'val': X_val[1][:, 0].flatten(),
        'test': X_test[1][:, 0].flatten(),
    }
    figs.append(create_hist(x_data, 'x'))

    # Coord values
    y_data = {
        'train': X_train[1][:, 1].flatten(),
        'val': X_val[1][:, 1].flatten(),
        'test': X_test[1][:, 1].flatten(),
    }
    figs.append(create_hist(y_data, 'y'))

    return figs


def random_translate(psf, parameters):
    max_translation_px = parameters['data_augmentation']['max_lateral_translation_px']
    translation_x = np.random.randint(-max_translation_px, max_translation_px)
    translation_y = np.random.randint(-max_translation_px, max_translation_px)
    psf = np.roll(psf, translation_y, axis=1)
    psf = np.roll(psf, translation_x, axis=0)
    return psf


def augment_psf(psf, parameters):
    psf += np.random.normal(0, parameters['data_augmentation']['noise_std'], size=psf.shape)
    psf[psf<0] = 0
    psf = random_translate(psf, parameters)
    return psf


def augment_datasets(X_train: Tuple[np.array, np.array], y_train: np.array, parameters: Dict):
    # TODO fix augmentation
    return X_train, y_train

    np.random.seed(parameters['random_seed'])
    n_aug = y_train.shape[0] * parameters['training']['aug_ratio']

    x_psfs, x_coords = X_train
    aug_psfs = np.zeros((n_aug, *x_psfs.shape[1:]))
    aug_coords = np.zeros((n_aug, *x_coords.shape[1:]))
    aug_zs = np.zeros((n_aug, 1))

    for store_i, select_i in enumerate(np.random.randint(0, y_train.shape[0], size=n_aug)):
        psf = augment_psf(x_psfs[select_i].copy(), parameters)
        aug_psfs[store_i] = psf
        aug_coords[store_i] = x_coords[select_i]
        aug_zs[store_i] = y_train[select_i]

    aug_psfs = norm_images(aug_psfs)

    all_psfs = np.concatenate((x_psfs, aug_psfs))
    all_coords = np.concatenate((x_coords, aug_coords))
    all_zs = np.concatenate((y_train, aug_zs))
    return (all_psfs, all_coords), all_zs
