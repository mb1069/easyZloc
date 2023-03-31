"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.4
"""
from .new_align import get_offsets
from typing import Dict

import numpy as np
import pandas as pd
from skimage.transform import resize
import tensorflow as tf
from skimage.feature import match_template
from .align_psfs import classic_align_psfs
import seaborn as sns
import matplotlib.pyplot as plt

def get_spot_corner_coords(loc, spot_size):
    x, y = loc['x'], loc['y']
    x0 = round(x - (spot_size/2))
    y0 = round(y - (spot_size/2))
    return x0, y0


def extract_training_stacks(spots: np.array, bead_stack: np.array, parameters: Dict) -> np.array:
    picasso_params = parameters['picasso']
    spot_size = picasso_params['spot_size']
    frame_idx = picasso_params['localised_frame']
    frame = bead_stack[frame_idx]
    stacks = []
    if parameters['DEBUG']:
        spots = spots[0:3]
    for spot in spots:
        res = match_template(frame, spot)
        i, j = np.unravel_index(np.argmax(res), res.shape)
        stack = bead_stack[:, i:i+spot_size, j:j+spot_size]
        stacks.append(stack)
    return np.array(stacks)


def resize_stacks(stacks: np.array, target_shape: Dict) -> np.array:
    stacks = np.array([resize(s, target_shape) for s in stacks])
    return stacks


def align_stacks(bead_stacks: np.array, locs: pd.DataFrame, parameters: Dict):
    offsets = classic_align_psfs(bead_stacks, locs)
    # offsets = get_offsets(locs, bead_stacks)
    fig = plt.figure()
    z_step = parameters['images']['train']['z_step']
    sns.scatterplot(data=locs, x='x', y='y', hue=(offsets - offsets.min()) * z_step)
    plt.ylabel('y [nm]')
    plt.xlabel('x [nm]')
    return offsets, fig


def norm_coordinates(locs: pd.DataFrame, parameters: np.array) -> pd.DataFrame:
    if parameters['DEBUG']:
        locs = locs.iloc[0:3]
    xy = locs[['x', 'y']]
    img_y, img_x = parameters['images']['train']['shape'][1:]
    xy['y'] = xy['y'] - (img_y/2)
    xy['x'] = xy['x'] - (img_x/2)

    xy['y'] /= img_y
    xy['x'] /= img_x
    # TODO remove in production
    xy[['x', 'y']] = 0
    return xy


def stacks_to_training_data(stacks: np.array, norm_coords: pd.DataFrame, offsets: np.array, parameters: Dict):
    psfs = []
    xy_coords = []
    z_coords = []
    for psf, xy_coord, offset in zip(stacks, norm_coords.to_dict(orient="records"), offsets):
        xy = np.repeat([[xy_coord['x'], xy_coord['y']]], psf.shape[0], axis=0)
        z = (np.arange(0, psf.shape[0]) + offset) * parameters['images']['train']['z_step']
        psfs.append(psf)
        xy_coords.append(xy)
        z_coords.append(z)

    psfs = np.concatenate(psfs)
    xy_coords = np.concatenate(xy_coords)
    z_coords = np.concatenate(z_coords)[:, np.newaxis]
    return psfs, xy_coords, z_coords


def norm_zero_one(img):
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max - img_min)


def norm_images(psfs: np.array):
    imgs =  np.stack([norm_zero_one(psf) for psf in psfs])
    return imgs


def norm_stacks(stacks: np.array):
    print('Stacks', stacks.shape)
    norm_stacks = []
    for stack in stacks:
        stack = norm_images(stack)
        norm_stacks.append(stack)
    return np.stack(norm_stacks)


def merge_model_inputs(resized_psfs, xy_coords, z_coords):
    return (resized_psfs, xy_coords), z_coords

def trim_stacks(psfs, xy_coords, z_coords, parameters):
    z_range = parameters['z_range']
    z_coords -= z_coords.mean()
    idx = np.argwhere(abs(z_coords.squeeze()) < z_range).squeeze()
    psfs = psfs[idx]
    xy_coords = xy_coords[idx]
    z_coords = z_coords[idx]

    return psfs, xy_coords, z_coords