from data.visualise import scatter_3d


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
import os
from data.datasets import TrainingDataSet, ExperimentalDataSet
from util import get_base_data_path
from config.datafiles import res_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model
import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization
from keras import regularizers
from data.datasets import TrainingDataSet, ExperimentalDataSet
from util import get_base_data_path
from config.datafiles import res_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose, Activation, Concatenate, LeakyReLU
from keras.models import Model, Input
from keras import regularizers, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.initializers import RandomNormal

log_dir = '/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/experiments/logdir'
def load_discriminator():
    channels = 64
    model = Sequential()
    model.add(Conv2D(channels, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(channels/2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(channels/2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(channels/4, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def load_datasets():
    z_range = 1000

    dataset = 'paired_bead_stacks'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=False)

    exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False)

    datasets = {k: [[], []] for k in train_dataset.data}
    for k in datasets:
        for i, ds in enumerate([train_dataset, exp_dataset], start=0):
            datasets[k][0].append(ds.data[k][0])
            datasets[k][1].append(np.tile([i], [ds.data[k][0].shape[0], 1]))
        datasets[k][0] = np.concatenate(datasets[k][0])
        datasets[k][1] = np.concatenate(datasets[k][1])

        # datasets[k][1] = datasets[k][1] * tf.one_hot(2, depth=2)
    return datasets

def view_feature_maps(dataset, model):
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = tf.keras.models.Model(model.input, layer_outputs)
    example_img = dataset['train'][0][0:1]
    feature_maps = feature_map_model.predict(example_img)

    image_belts = []
    for layer_name, feature_map in zip(model.layers, feature_maps):  
        if len(feature_map.shape) == 4:
            k = feature_map.shape[-1]  
            size=feature_map.shape[1]
            image_belt = np.zeros(shape=(feature_map.shape[1], feature_map.shape[2]*k))
            for i in range(k):
                feature_image = feature_map[0, :, :, i]
                feature_image-= feature_image.mean()
                feature_image/= feature_image.std ()
                feature_image*=  64
                feature_image+= 128
                feature_image= np.clip(feature_image, 0, 255).astype('uint8')
                image_belt[:, i * size : (i + 1) * size] = feature_image
            image_belts.append(image_belt)
            
    
    for img in image_belts:
        plt.imshow(img)
        plt.show()



def main():
    model_path = os.path.join(os.path.dirname(__file__), 'results', 'classifier.h5')
    dataset = load_datasets()

    model = load_discriminator()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),metrics=['accuracy'])
    print(model_path)
    if os.path.exists(model_path):
        model.build(input_shape=dataset['train'][0][0:1].shape)
        model.load_weights(model_path)
    else:
        callbacks = [
            ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=3, verbose=True,
            mode='min', min_delta=0.01, cooldown=0, min_lr=0,),
            EarlyStopping(monitor='accuracy', patience=5, verbose=True, min_delta=0.01, restore_best_weights=True),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        for d in dataset:
            print(d, dataset[d][0].shape, dataset[d][1].shape)
        model.fit(*dataset['train'], epochs=100, validation_data=(*dataset['val'],), callbacks=callbacks)
        model.save_weights(model_path)
    # view_feature_maps(dataset, model)


if __name__=='__main__':
    main()