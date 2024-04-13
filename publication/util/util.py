import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import tensorflow as tf

def norm_zero_one(s):
    return (s - s.min()) / (s.max() - s.min())


def concat_psf_axial(psf, subsample_n, perc_disp=0.6):
    margin = (1 - perc_disp) / 2
    start = round(psf.shape[0] * margin) + 1
    end = round(psf.shape[0] * (1 - margin))
    sub_psf = np.concatenate(psf[slice(start, end + 1, subsample_n)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    # sub_psf = img_as_ubyte(sub_psf)
    return sub_psf


def show_psf_axial(psf, title=None, subsample_n=7):
    psf = np.copy(psf)
    sub_psf = concat_psf_axial(psf, subsample_n).T

    if title:
        plt.title(title)
    plt.axis('off')
    plt.grid()
    plt.imshow(sub_psf)

    plt.show()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def grid_psfs(psfs, cols=10):
    rows = (len(psfs) // cols) + (1 if len(psfs) % cols != 0 else 0)
    n_spaces = int(cols * rows)
    if n_spaces > len(psfs):
        placeholder = np.zeros((n_spaces-len(psfs), *psfs[0].shape))
        placeholder[:] = np.mean(psfs)
        psfs = np.concatenate((psfs, placeholder))
        cols = len(psfs) // rows
    psfs = list(chunks(psfs, cols))
    psfs = [np.concatenate(p, axis=-1) for p in psfs]
    psfs = np.concatenate(psfs, axis=-2)
    return psfs


def load_model(args):
    return keras.models.load_model(os.path.join(args['outdir'], './latest_vit_model3'))

def save_dataset(dataset, name, args):
    dataset.save(os.path.join(args['outdir'], name))

def load_dataset(name, args):
    return tf.data.Dataset.load(os.path.join(args['outdir'], name))

from tensorflow.keras import Sequential, layers

image_size = 64
imshape = (image_size, image_size)


def apply_resizing(img_xy, z):
    img_preprocessing = Sequential([
        layers.Resizing(*imshape),
        layers.Lambda(tf.image.grayscale_to_rgb)
    ])
    img_xy = list(img_xy)
    img_xy[0] = img_preprocessing(img_xy[0])
    return tuple(img_xy), z



def apply_img_norm(img_xy, z):
    img_xy = list(img_xy)
    imgs = img_xy[0]
    means = tf.math.reduce_mean(imgs, axis=(1,2,3), keepdims=True)
    imgs -= means
    maxs = tf.math.reduce_max(imgs, axis=(1,2,3), keepdims=True)
    imgs = tf.nn.relu(imgs / maxs)
    return (imgs, img_xy[1]), z


def preprocess_img_dataset(dataset, batch_size=None):
    dataset = dataset.map(apply_resizing, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size:
        dataset = dataset.batch(batch_size)
    dataset = dataset.map(apply_img_norm, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
