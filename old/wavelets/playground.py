import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, optimizers
# NN model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.wavelets.wavelet_data.datasets.base_dataset import BaseDataset
from src.wavelets.wavelet_data.datasets.datasets import load_green_lupus_nephritis_bead_stack


def normalise_psfs(X):
    psfs = []
    for psf in X:
        min_val = psf.min()
        max_val = psf.max()
        psf = (psf - min_val) / (max_val - min_val)
        psfs.append(psf)
    X = np.stack(psfs)
    return X


model = Sequential([
    Flatten(),
    Dense(512, activation='swish'),
    Dense(256, activation='swish'),
    Dense(128, activation='swish'),
    Dense(64, activation='swish'),
    Dense(32, activation='swish'),
    Dense(1, activation='linear')
])

optimizer = optimizers.SGD(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer, metrics=['MeanAbsoluteError'])

callbacks = [
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=True),
    ReduceLROnPlateau(monitor='val_mean_absolute_error', patience=10, verbose=True)
]

# ds = BaseDataset()
# X, y = ds.prepare_data(z_range=1000, max_psfs=10000, level=None)

ds = load_green_lupus_nephritis_bead_stack(dwt_level=None)

X_train, y_train = ds['train']
X_val, y_val = ds['val']

X_train = normalise_psfs(X_train)
X_val = normalise_psfs(X_val)

max_y = y_train.max()
y_train = y_train / max_y
y_val = y_val / max_y

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(5000)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(64)

history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset, callbacks=callbacks)

mse = mean_squared_error(y_val, model(X_val))
print(mse * max_y)
plt.plot(history.history['mean_absolute_error'], label='train')
plt.plot(history.history['val_mean_absolute_error'], label='val')
plt.legend()
plt.show()
