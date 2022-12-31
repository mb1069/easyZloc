"""
This is a boilerplate pipeline 'classify'
generated using Kedro 0.18.4
"""
from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from tensorflow.keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm.keras import TqdmCallback
import seaborn as sns
import operator as op

def split_train_val_test(X: Tuple[np.array, np.array], y: np.array, parameters: Dict):
    train_size = parameters['train_size']
    test_size = parameters['test_size']
    idx = np.arange(0, y.shape[0])
    train_idx, other_idx = train_test_split(idx, train_size=train_size, random_state=42)
    val_idx, test_idx = train_test_split(
        other_idx, test_size=test_size, random_state=42)

    X_train = [x[train_idx] for x in X]
    y_train = y[train_idx]

    X_val = [x[val_idx] for x in X]
    y_val = y[val_idx]

    X_test = [x[test_idx] for x in X]
    y_test = y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def compile_model(model, lr):
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(
        learning_rate=lr, decay=1e-6), metrics=['mean_absolute_error'])
    print(model.summary())


def get_model(parameters):
    img_input_shape = parameters['model_input_shape']
    lr = parameters['training']['learning_rate']
    bound, n_channels = img_input_shape[0], img_input_shape[-1]
    img_input = Input(img_input_shape)
    coords_input = Input((2,))

    weights = 'imagenet' if n_channels == 3 else None
    print(f'Using weights {weights}')
    resnet = tf.keras.applications.resnet_v2.ResNet101V2(
        include_top=False,
        weights=weights,
        pooling='max',
        input_shape=img_input_shape
    )
    x = img_input
    augmentation = Sequential([
        tf.keras.layers.GaussianNoise(0.2, seed=42),
        tf.keras.layers.RandomTranslation(1/bound, 1/bound)
    ])
    x = augmentation(x)
    x = resnet(x)

    mlp = Sequential([
        Dense(1024),
        Dropout(0.5),
        Dense(1024),
        Dropout(0.5),
        Dense(1)
    ])
    x = tf.concat((x, coords_input), axis=1)
    x = mlp(x)
    model = Model(inputs=(img_input, coords_input), outputs=x)
    compile_model(model, lr)
    return model


def train_classifier(X_train: Tuple[np.array, np.array], y_train: np.array, X_val: Tuple[np.array, np.array], y_val: np.array, parameters: Dict):
    batch_size = parameters['training']['batch_size']
    epochs = parameters['training']['max_epochs']
    model = get_model(parameters)
    callbacks = [
        ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.1, patience=25, verbose=True, mode='min', min_delta=1, min_lr=1e-7,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=300, verbose=False, min_delta=1, restore_best_weights=True),
        TqdmCallback(verbose=1),
    ]

    history = model.fit(X_train, y_train, epochs=epochs, shuffle=True, verbose=False, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)
    
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


from sklearn.metrics import max_error, mean_absolute_error
def eval_classifier(model, X_train: Tuple[np.array, np.array], y_train: np.array, X_val: Tuple[np.array, np.array], y_val: np.array, X_test: Tuple[np.array, np.array], y_test: np.array):
    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    all_metrics = {}
    figs = []
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
        
    
    all_metrics = {k: {'value': v, 'step': 0} for k, v in all_metrics.items()}

    return all_metrics, figs

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
    # z coords
    z_data = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    figs.append(create_boxplot(z_data, 'z [nm]'))

    pixel_data = {
        'train': X_train[0].mean(axis=(1,2,3)),
        'val': X_val[0].mean(axis=(1,2,3)),
        'test': X_test[0].mean(axis=(1,2,3)),
    }
    figs.append(create_boxplot(pixel_data, 'mean pixel val'))

    pixel_data = {
        'train': X_train[0].max(axis=(1,2,3)),
        'val': X_val[0].max(axis=(1,2,3)),
        'test': X_test[0].max(axis=(1,2,3)),
    }
    figs.append(create_boxplot(pixel_data, 'max pixel val'))

    pixel_data = {
        'train': X_train[0].min(axis=(1,2,3)),
        'val': X_val[0].min(axis=(1,2,3)),
        'test': X_test[0].min(axis=(1,2,3)),
    }
    figs.append(create_boxplot(pixel_data, 'min pixel val'))

    # Coord values
    rho_data = {
        'train': X_train[1][0].flatten(),
        'val': X_val[1][0].flatten(),
        'test': X_test[1][0].flatten(),
    }
    figs.append(create_boxplot(rho_data, 'rho'))

    # Coord values
    theta_data = {
        'train': X_train[1][1].flatten(),
        'val': X_val[1][1].flatten(),
        'test': X_test[1][1].flatten(),
    }
    figs.append(create_boxplot(theta_data, 'theta'))


    return figs