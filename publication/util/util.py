import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json


def get_model_report(model_dir):
    model_report = os.path.join(model_dir, 'results', 'report.json')

    with open(model_report) as f:
        d = json.load(f)
    return d


def get_model_img_norm(model_report):
    return model_report['args']['norm']


def read_multipage_tif(impath):
    print(f'Reading {impath} using PIL')
    d = Image.open(impath)
    n_frames = d.n_frames
    shape = np.array(d).shape
    img = np.zeros((n_frames, *shape))
    for i in range(n_frames):
        d.seek(i)
        img[i] = np.array(d)
    return img

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


def show_psf_axial(psf, title=None, subsample_n=7, perc_disp=0.6):
    psf = np.copy(psf)
    sub_psf = concat_psf_axial(psf, subsample_n, perc_disp).T

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


def apply_resizing(img_xy, z, image_size=64, inter='bicubic'):
    imshape = (image_size, image_size)
    img_preprocessing = Sequential([
        layers.Resizing(*imshape, interpolation=inter),
        # layers.Lambda(tf.image.grayscale_to_rgb)
    ])
    img_xy = list(img_xy)
    img_xy[0] = img_preprocessing(img_xy[0])
    return tuple(img_xy), z


from functools import partial


def _apply_img_norm(img_xy, z, img_norm):
    img_xy = list(img_xy)
    imgs = img_xy[0]
    if img_norm == 'frame-mean':
        means = tf.math.reduce_mean(imgs, keepdims=True)
        imgs -= means
        maxs = tf.math.reduce_max(imgs, keepdims=True)
        imgs = tf.nn.relu(imgs / maxs)
    elif img_norm == 'frame-min':
        mins = tf.math.reduce_min(imgs, keepdims=True)
        imgs -= mins
        maxs = tf.math.reduce_max(imgs, keepdims=True)
        imgs = tf.nn.relu(imgs / maxs)
    elif img_norm == 'frame-max':
        maxs = tf.math.reduce_max(imgs, keepdims=True)
        imgs = imgs / maxs
    elif img_norm == 'fov-max':
        maxs = 65535
        imgs = tf.nn.relu(imgs / maxs)
    # elif img_norm == 'fov-minmax':
    #     maxs = tf.math.reduce_max(imgs)
    #     mins = tf.math.reduce_min(imgs)
    #     imgs = (imgs - mins) / (maxs-mins)
    elif img_norm == 'standard':
        mean = tf.math.reduce_mean(imgs)
        std = tf.math.reduce_std(imgs)
        imgs = (imgs - mean) / std
    else:
        print(f'img_norm: {img_norm} not supported')
        raise NotImplementedError()
    return (imgs, img_xy[1]), z


def preprocess_img_dataset(dataset, image_size, img_norm):
    apply_img_norm = partial(_apply_img_norm, img_norm=img_norm)

    f = partial(apply_resizing, image_size=image_size, inter='bicubic')
    dataset = dataset.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    # if args.get('batch_size'):
    #     dataset = dataset.batch(args['batch_size'])
    dataset = dataset.map(apply_img_norm, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
