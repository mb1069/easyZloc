from scipy.sparse import data
from tensorflow.python.ops.gen_control_flow_ops import next_iteration_eager_fallback
from tifffile.tifffile import imagej_shape
from data.visualise import scatter_3d


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from data.datasets import TrainingDataSet, ExperimentalDataSet
from util import get_base_data_path
from config.datafiles import res_file, weights_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model, predict
import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization
from keras import regularizers
from data.datasets import TrainingDataSet, ExperimentalDataSet
from util import get_base_data_path
from config.datafiles import res_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model
from keras.layers import BatchNormalization, Conv2D, Dropout, Flatten, Dense, Conv2DTranspose, Activation, Concatenate, LeakyReLU, UpSampling2D, concatenate
from keras.models import Model, Input
from keras import regularizers, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from experiments.dataset_classifier import load_discriminator
import os




log_dir = os.path.join(os.path.dirname(__file__), 'logdir')
batch_size = 512

file_writer = tf.summary.create_file_writer(log_dir)


# tf.debugging.experimental.enable_dump_debug_info(log_dir)

def load_datasets(batch=True):
    z_range = 1000

    dataset = 'openframe'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False)

    exp_dataset = TrainingDataSet(dataset_configs[dataset]['sphere_ground_truth'], z_range, transform_data=False)

    datasets = {k: [[], []] for k in train_dataset.data}

    # train dataset
    for k in ['train', 'val']:
        ds_pos_psfs, ds_pos_z = train_dataset.data[k]
        ds_pos_src = np.tile([0], [train_dataset.data[k][0].shape[0], 1])
        ds_pos_z = train_dataset.data[k][1][:, np.newaxis]
        ds_pos_target = np.concatenate((ds_pos_src, ds_pos_z), axis=1)
        ds_pos_dataset = tf.data.Dataset.from_tensor_slices((ds_pos_psfs, ds_pos_target))

        ds_neg_psfs = exp_dataset.data[k][0]
        ds_neg_src = np.tile([0], [exp_dataset.data[k][0].shape[0], 1])
        ds_neg_z = np.tile([0.0], [exp_dataset.data[k][0].shape[0], 1])
        ds_neg_target = np.concatenate((ds_neg_src, ds_neg_z), axis=1)
        ds_neg_dataset = tf.data.Dataset.from_tensor_slices((ds_neg_psfs, ds_neg_target))

        dataset = tf.data.Dataset.zip((ds_pos_dataset, ds_neg_dataset))
        dataset = dataset.flat_map(lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg)))
        
        datasets[k] = dataset.batch(batch_size) if batch else dataset



        # for i, ds in enumerate([train_dataset, exp_dataset], start=0):
        #     datasets[k][0].append(ds.data[k][0])
        #     data_srcs = np.tile([i], [ds.data[k][0].shape[0], 1])
        #     if i == 0:
        #         data_z = ds.data[k][1][:, np.newaxis]
        #     else:
        #         # If 'experimental' data
        #         data_z = np.tile([0.0], [ds.data[k][0].shape[0], 1])
        #     datasets[k][1].append(np.concatenate((data_srcs, data_z), axis=1))

        # datasets[k][0] = np.concatenate(datasets[k][0])
        # datasets[k][1] = np.concatenate(datasets[k][1])

        # dataset = tf.data.Dataset.from_tensor_slices((datasets[k][0], datasets[k][1]))
        # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        # datasets[k] = dataset
    return datasets
    
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def load_generator(img_shape):
    fg = 64
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        print(layer_input.shape, '->', d.shape)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, fg, bn=False)
    d2 = conv2d(d1, fg*2)
    d3 = conv2d(d2, fg*4)
    d4 = conv2d(d3, fg*8)
    d5 = conv2d(d4, fg*8)
    # d6 = conv2d(d5, fg*8)
    # d7 = conv2d(d6, fg*8)

    # Upsampling
    # u1 = deconv2d(d7, d6, fg*8)
    # u2 = deconv2d(d6, d5, fg*8)
    u3 = deconv2d(d5, d4, fg*8)
    u4 = deconv2d(u3, d3, fg*4)
    u5 = deconv2d(u4, d2, fg*2)
    u6 = deconv2d(u5, d1, fg)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    model =  Model(d0, output_img)

    return model


    OUTPUT_CHANNELS = 1
    inputs = tf.keras.layers.Input(shape=[40, 40, 1])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
    ]

    up_stack = [
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]


    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs
    # Downsampling through the model
    skips = []
    for i, down in enumerate(down_stack):
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

abcstep = 0
class DANN(keras.Model):
    overall_step = 0
    # Domain adversarial NN
    def __init__(self, image_shape):
        super().__init__()
        self.discriminator = load_discriminator()
        self.generator = load_generator(image_shape)
        # self.regression_model = load_regression_model()
        self.gen_loss_tracker = keras.metrics.Mean(name='generator_loss')
        self.disc_loss_tracker = keras.metrics.Mean(name='discriminator_loss')
        # self.regression_loss_tracker = keras.metrics.Mean(name='regression_loss')

    @property
    def metrics(self):
        # return [self.regression_loss_tracker, self.gen_loss_tracker, self.disc_loss_tracker]
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self):
        super(DANN, self).compile(metrics=['accuracy'])
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0000001)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=1)
        # self.r_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        # self.regression_loss_fn = keras.losses.MeanSquaredError()
        
        self.generator.load_weights(weights_file)

    
    def gen_fake_dataset(self, experimental_psfs, transformed_psfs):
        n_fake = tf.shape(transformed_psfs)[0]
        n_real = tf.shape(experimental_psfs)[0]
        fake_labels = tf.zeros((n_fake, 1))
        
        real_labels = tf.ones((n_real, 1))

        # Train discriminator
        labels = tf.concat([fake_labels, real_labels], axis=0)
        all_psfs = tf.concat([transformed_psfs, experimental_psfs], axis=0)
        return all_psfs, labels

    def train_step(self, data):
        psfs = data[0]
        src_labels = data[1][:, 0]
        z_labels = data[1][:, 1]

        experimental_idx = tf.where(src_labels==1)
        experimental_psfs = tf.squeeze(tf.gather(psfs, experimental_idx), axis=1)
        print('EXP')
        print(experimental_idx.get_shape())
        print(experimental_psfs.get_shape())
        print('\n')
        calibration_idx = tf.where(src_labels==0)
        calibration_psfs = tf.squeeze(tf.gather(psfs, calibration_idx), axis=1)

        print('Calibration')
        print(calibration_idx.get_shape())
        print(calibration_psfs.get_shape())
        print('\n')
        calibration_z = tf.gather(z_labels, calibration_idx)

        # Create fake images and real/fake labels
        transformed_psfs = tf.cast(self.generator(calibration_psfs), tf.float64)

        all_psfs, labels = self.gen_fake_dataset(experimental_psfs, transformed_psfs)
        with tf.GradientTape() as t1:
            preds = self.discriminator(all_psfs)
            d_loss = self.loss_fn(labels, preds)
        d_grads = t1.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))


        # # Train regressor
        # with tf.GradientTape() as t2:
        #     pred_z = tf.squeeze(self.regression_model(transformed_psfs))
        #     r_loss = self.regression_loss_fn(calibration_z, pred_z)
        # r_grads = t2.gradient(r_loss, self.regression_model.trainable_weights)
        # self.r_optimizer.apply_gradients(zip(r_grads, self.regression_model.trainable_weights))

        # Train generator
        with tf.GradientTape(persistent=True) as t3:
            transformed_psfs = self.generator(calibration_psfs)
            # all_psfs, labels = self.gen_fake_dataset(experimental_psfs, transformed_psfs)
            # preds = self.discriminator(all_psfs)

            n_fake = tf.shape(transformed_psfs)[0]
            labels = tf.zeros((n_fake, 1))
            preds = self.discriminator(transformed_psfs)
            g_loss = -self.loss_fn(labels, preds)

            # z_preds = self.regression_model(transformed_psfs)
            # r_loss = self.regression_loss_fn(calibration_z, z_preds)
    
        grads = t3.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # grads = t3.gradient(r_loss, self.generator.trainable_weights)
        # self.r_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        # self.regression_loss_tracker.update_state(r_loss)


        # Using the file writer, log the reshaped image.
        imgs = []
        for i in range(10):
            transformed_psf = tf.cast(transformed_psfs[i], tf.float32)
            original_psf = tf.cast(calibration_psfs[i], tf.float32)
            img = tf.concat((original_psf, transformed_psf), axis=1)
            imgs.append(img)

        img = tf.concat(imgs, axis=0)
        img = tf.reshape(img, (-1, *img.get_shape()))
        with file_writer.as_default():
            tf.summary.image("Training data", img, step=self.overall_step)
        self.overall_step += 1
        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.disc_loss_tracker.result(),
            # 'r_loss': self.regression_loss_tracker.result(),
        }

def plot_history(history):
    plt.plot(history.history['g_loss'])
    plt.plot(history.history['d_loss'])
    plt.show()
def main():
    dataset = load_datasets()
    image_shape = (32, 32, 1)
    

    model = DANN(image_shape)
    model.compile()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    history = model.fit(dataset['train'], batch_size=batch_size, epochs=500, callbacks=callbacks)
    plot_history(history)

if __name__=='__main__':
    main()