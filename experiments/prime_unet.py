from experiments.gan_noiser import load_generator, load_datasets
from tensorflow import keras
from keras.optimizers import adam_v2
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from config.datafiles import res_file, weights_file

import matplotlib.pyplot as plt
import os


def prime_generator():
    datasets = load_datasets(batch=False)
    generator = load_generator((32, 32, 1))
    generator.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),metrics=['mean_squared_error'])

    train_ds = np.array([d[0] for d in list(datasets['train'])])
    val_ds = np.array([d[0] for d in list(datasets['val'])])

    callbacks = [
        ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=10, verbose=True,
        mode='min', min_delta=0.01, cooldown=0, min_lr=0,),
        EarlyStopping(monitor='val_mean_squared_error', patience=50, verbose=True, min_delta=0.01, restore_best_weights=True),
    ]
    history = generator.fit(train_ds, train_ds, epochs=500, 
        validation_data=(val_ds, val_ds), 
        callbacks=callbacks)

    plt.plot(history.history['mean_squared_error'], label='mse')
    plt.plot(history.history['val_mean_squared_error'], label='val_mse')
    plt.yscale('log')
    plt.legend(loc=1)
    plt.show()

    generator.save_weights(weights_file, save_format='h5')
    for i in range(0, 10):
        img = train_ds[i:i+1]
        img2 = generator.predict(img).squeeze()
        img = img.squeeze()
        img2 = img2 / img2.max()
        img = np.concatenate((img, img2), axis=1).squeeze()
        print(img.shape)
        plt.imshow(img)
        plt.show()


if __name__=='__main__':
    print(weights_file)
    if os.path.exists(weights_file):
        model = load_generator((32, 32, 1)).load_weights(weights_file)
        datasets = load_datasets(batch=False)
        generator = load_generator((32, 32, 1))
        generator.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),metrics=['mean_squared_error'])

        train_ds = np.array([d[0] for d in list(datasets['train'])])
        val_ds = np.array([d[0] for d in list(datasets['val'])])
        for i in range(0, 10):
            img = train_ds[i:i+1]
            img2 = generator.predict(img).squeeze()
            img = img.squeeze()
            img2 = img2 / img2.max()
            print(img.shape, img2.shape)
            img = np.concatenate((img, img2), axis=1).squeeze()
            print(img.shape)
            plt.imshow(img)
            plt.show()
    else:
        prime_generator()
