from final_project.smlm_3d.data.visualise import scatter_3d
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.util import get_base_data_path
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.workflow_v2 import eval_model
from keras.layers import BatchNormalization
from keras import regularizers
DISABLE_LOAD_SAVED = False
FORCE_LOAD_SAVED = False
DEBUG = False

USE_GPU = True
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src/wavelets/wavelet_data/output')

model_path = os.path.join(os.path.dirname(__file__), 'tmp/model.json')
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

model_path = os.path.join(os.path.dirname(__file__), 'model.h5')

checkpoint_path =  os.path.join(os.path.dirname(__file__), 'chkp')


def load_regression_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),metrics=['mean_absolute_error'])
    return model


def train_model(dataset, val_dataset=None):
    if not val_dataset:
        val_dataset = dataset['val']
    for k in dataset:
        dataset[k] = list(dataset[k])
        dataset[k][0] = dataset[k][0] / dataset[k][0].max(axis=(1, 2))[:, np.newaxis, np.newaxis]

    model = load_regression_model()

    callbacks = [
        ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=3, verbose=True,
        mode='min', min_delta=0.01, cooldown=0, min_lr=0,),
        EarlyStopping(monitor='mean_absolute_error', patience=5, verbose=True, min_delta=0.01, restore_best_weights=True),
    ]
    print(dataset['train'][0].shape)
    print(dataset['train'][1].shape)

    history = model.fit(*dataset['train'], epochs=500, validation_data=(*val_dataset,), callbacks=callbacks)

    fig, ax1 = plt.subplots()
    ax1.plot(history.history['mean_absolute_error'], label='mse')
    ax1.plot(history.history['val_mean_absolute_error'], label='val_mse')
    ax1.set_yscale('log')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(history.history['lr'], label='lr')
    ax2.legend(loc=0)

    plt.show()
    return model

def save_model(model):
    model.save(model_path)

def load_model():
    return keras.models.load_model(model_path)

def main():
    
    z_range = 1000

    dataset = 'openframe'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=True)

    exp_dataset = TrainingDataSet(dataset_configs[dataset]['sphere_ground_truth'], z_range, transform_data=False)


    model = train_model(train_dataset.data, train_dataset.data['val'])


    save_model(model)
    model = load_model()


    eval_model(model, train_dataset.data['test'], 'Bead test (bead stack training)')
    eval_model(model, exp_dataset.data['test'], 'Sphere (bead stack training)')

    # eval_model(model, train_dataset.data['test'], 'Bead stack')


if __name__ == '__main__':
    main()
