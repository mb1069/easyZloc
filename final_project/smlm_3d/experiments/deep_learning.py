from scipy.sparse import data
from final_project.smlm_3d.data.visualise import scatter_3d
import os

import matplotlib.pyplot as plt
import numpy as np

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.workflow_v2 import eval_model, inspect_large_errors

DEBUG = False

USE_GPU = True
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src/wavelets/wavelet_data/output')

model_path = os.path.join(os.path.dirname(__file__), 'tmp/model.json')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16


model_path = os.path.join(os.path.dirname(__file__), 'model_ckpt')

checkpoint_path =  os.path.join(os.path.dirname(__file__), 'chkp')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import layers as Layers


class ResBlock(Model):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__(name='ResBlock')
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1


class ResNet34(Model):
    def __init__(self):
        super(ResNet34, self).__init__(name='ResNet34')
        self.conv1 = Conv2D(64, 7, 2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.conv2_1 = ResBlock(64)
        self.conv2_2 = ResBlock(64)
        self.conv2_3 = ResBlock(64)

        self.conv3_1 = ResBlock(128, 2)
        self.conv3_2 = ResBlock(128)
        self.conv3_3 = ResBlock(128)
        self.conv3_4 = ResBlock(128)

        self.conv4_1 = ResBlock(256, 2)
        self.conv4_2 = ResBlock(256)
        self.conv4_3 = ResBlock(256)
        self.conv4_4 = ResBlock(256)
        self.conv4_5 = ResBlock(256)
        self.conv4_6 = ResBlock(256)

        self.conv5_1 = ResBlock(512, 2)
        self.conv5_2 = ResBlock(512)
        self.conv5_3 = ResBlock(512)

        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.dp1 = Dropout(0.5)
        self.fc2 = Dense(512, activation='relu')
        self.dp2 = Dropout(0.5)
        self.fc3 = Dense(1)

    def call(self, inp):
        img, coords = inp
        x = self.conv1(img)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.pool(x)

        # Concat norm X/Y coordinates
        x = tf.concat((x, coords), axis=1)

        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        # x = tf.tanh(x) * 1000
        return x



def load_regression_model():

    # model = Sequential()
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='linear'))

    model = ResNet34()
    model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),metrics=['mean_absolute_error'])

    return model


def train_model(dataset, val_dataset=None):
    if not val_dataset:
        val_dataset = dataset['val']
    for k in dataset:
        imgs = dataset[k][0][0]
        norm_imgs = imgs / imgs.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
        dataset[k][0][0] = norm_imgs

    model = load_regression_model()

    callbacks = [
        ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=5, verbose=True,
        mode='min', min_delta=1, cooldown=0, min_lr=0,),
        EarlyStopping(monitor='val_mean_absolute_error', patience=25, verbose=True, min_delta=1, restore_best_weights=True),
    ]
    print(dataset['train'][0][0].shape)
    print(dataset['train'][0][1].shape)
    print(dataset['train'][1].shape)

    history = model.fit(*dataset['train'], batch_size=256, epochs=500, validation_data=(*val_dataset,), callbacks=callbacks)

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
    model.save(model_path, save_format="tf")
    print('Saved model!')

def load_model():
    return keras.models.load_model(model_path)

def main():
    
    z_range = 1000

    dataset = 'paired_bead_stacks'


    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=True)
    exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False, split_data=False)

    model = train_model(train_dataset.data)
    save_model(model)

    model = load_model()
    eval_model(model, train_dataset.data['test'], 'Bead test (bead training)')
    eval_model(model, exp_dataset.data['all'], 'Bead test 2 (bead training)', w_shift_correction=True)

    inspect_large_errors(model, exp_dataset)


    # eval_model(model, exp_dataset.data['train'], 'Bead test 2 (bead training)', shift_correction=False)

    # model = load_model()
    # coords = exp_dataset.predict_dataset(model)
    # scatter_3d(coords)

    # eval_model(model, exp_dataset.data['test'], 'Sphere (sphere training)')

    # eval_model(model, train_dataset.data['test'], 'Bead stack')

# MAE: 72.4392
# MAE: 136.6815

# Saved model!
# MAE: 51.9097
# Pred shift 177.51266366310273
# Post shift 127.69376473759141
# MAE: 127.6938



if __name__ == '__main__':
    main()
