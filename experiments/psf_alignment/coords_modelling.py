#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import pickle
import numpy as np
import matplotlib.pyplot as plt
from data.visualise import grid_psfs, show_psf_axial
from tifffile import imread
import pandas as pd
import seaborn as sns

def load_pickle_file(dpath):
    with open(dpath, 'rb') as f:
        return pickle.load(f)

# # MQ_data     
stacks = '/home/miguel/Projects/uni/data/smlm_3d/20231004_20nm_beads_3um_range_10nm_step/combined/stacks.ome.tif'
locs = '/home/miguel/Projects/uni/data/smlm_3d/20231004_20nm_beads_3um_range_10nm_step/combined/locs.hdf'
# exclude_idx = [5, 7, 11, 14, 22, 24, 26, 27, 28, 31, 32, 35, 37, 38, 40, 45, 50, 51, 54, 68, 69, 71, 72, 82, 87, 89, 91, 98, 102, 108, 109, 112, 113, 115, 116, 121, 122, 123, 127, 129, 131, 132, 133, 138, 141, 144, 150, 151, 154, 161, 167, 169, 170, 172, 178, 179, 181, 182, 184, 185, 186, 187, 190, 200, 201, 205, 206, 210, 214, 219, 221, 224, 226, 230, 233, 234, 235, 236, 237, 243]
exclude_idx = []
Z_STEP = 10

# FD-deeploc data
# stacks = '/home/miguel/Projects/uni/data/smlm_3d/fd-deeploc-data/Astigmatism_beads_stacks_2um/combined/stacks.ome.tif'
# locs = '/home/miguel/Projects/uni/data/smlm_3d/fd-deeploc-data/Astigmatism_beads_stacks_2um/combined/locs.hdf'
# exclude_idx = []
# Z_STEP = 10

all_psfs = imread(stacks)
all_locs = pd.read_hdf(locs, key='locs')

all_psfs = all_psfs[:, :, :, :, np.newaxis]

print(all_psfs.shape, all_psfs.dtype)

# # for i, psf in enumerate(psfs.sum(axis=-1)):
# #     plt.title(str(i))
# #     show_psf_axial(psf)


# # exclude_idx = [0, 5, 7, 12, 22, 26, 32, 35, 38, 40, 45, 50, 51, 54, 68, 69, 71, 72, 82, 87, 89, 91, 98, 102, 108, 109, 112, 113, 115, 116, 121, 122, 123, 124, 127, 129, 131, 132, 133, 138, 141, 144, 150, 151, 154, 161, 167, 169, 170, 172, 178, 179, 181, 182, 184, 185, 186, 187, 190, 200, 201, 205, 206, 210, 214, 219, 221, 224, 226, 230, 233, 234, 235, 236, 237, 243]

# # print('Excluded PSFs \n\n\n\n\n')
# # for i in exclude_idx:
# #     show_psf_axial(psfs[i].mean(axis=-1), str(i))
# #     plt.plot(psfs[i].max(axis=(1,2)))
# #     plt.show()
# # print('End of excluded PSFs \n\n\n\n\n')

# # print(psfs.shape[0])
# # for i in range(psfs.shape[0]):
# #     if i in exclude_idx:
# #         continue
# #     plt.title(str(i))
# #     show_psf_axial(psfs[i].mean(axis=-1))
# #     plt.plot(psfs[i].max(axis=(1,2,3)), label='max')
# #     plt.legend()
# #     plt.title(str(i))
# #     plt.show()

# idx = [i for i in range(psfs.shape[0]) if i not in exclude_idx]
# psfs = psfs[idx]
# locs = locs.iloc[idx]
all_locs['idx'] = np.arange(all_locs.shape[0])


# In[2]:


# xlim = ((450, 750))
# ylim = ((450, 750))


# FD-DEEPLOCO
# xlim = ((810, 810+250))
# ylim = ((790, 790+250))


# OpenFrame
xlim = (all_locs['x'].min()-1, all_locs['x'].max()+1)
ylim = (all_locs['y'].min()-1, all_locs['y'].max()+1)

idx = (xlim[0] < all_locs['x']) & (all_locs['x'] < xlim[1]) & (ylim[0] < all_locs['y']) & (all_locs['y'] < ylim[1])

locs = all_locs[idx]
psfs = all_psfs[locs['idx']]

print(psfs.shape)

ys = []
for i in range(psfs.shape[0]):
    y = np.arange(psfs.shape[1]) * Z_STEP
    y = y - 1000
    ys.append(y)
ys = np.stack(ys)


# In[3]:


print(psfs.shape, psfs.min(), psfs.max())
print(ys.shape)
print(locs.shape)


# In[4]:


# from data.visualise import show_psf_axial
# plt.rcParams['figure.figsize'] = [5, 3]
# for i, psf in enumerate(psfs[0:500]):
#     show_psf_axial(psf.mean(axis=-1), str(i))


# In[5]:


# exclude_idx = [35, 55, 60, 96, 104, 113, 128, 132, 230, 234]
# FD-DEEPLOCO
# exclude_idx = [0, 82, 109, 114, 138, 141, 149, 153]

# MIQI
exclude_idx = [35, 87, 91, 109, 112, 113, 115, 122, 141, 144, 161, 181, 182, 185, 190, 200, 201, 219, 221, 224, 226, 230, 233, ]
exclude_idx = []
idx = [i for i in range(psfs.shape[0]) if i not in exclude_idx][0:10]
psfs = psfs[idx]
locs = locs.iloc[idx]


# In[6]:


# # Spline peak finding
# from tqdm import tqdm
# import numpy as np

# from scipy.interpolate import UnivariateSpline
# from data.align_psfs import norm_zero_one
# from scipy import signal
# import cv2
# import numpy as np

# plt.rcParams['figure.figsize'] = [5, 3]

# DEBUG = True

# UPSCALE_RATIO = 10

# bad_psfs_idx = []

# def denoise(img):
#     from scipy.ndimage import gaussian_filter
    
#     return gaussian_filter(img.copy(), sigma=(2, 1, 1))
    

# def find_peak(i, psf):
#     if psf.ndim == 4:
#         psf = psf.mean(axis=-1)
#     x = np.arange(psf.shape[0]) * Z_STEP
# #     psf = denoise(psf)
    
#     inten = norm_zero_one(psf.max(axis=(1,2)))

#     cs = UnivariateSpline(x, inten, k=3, s=1.1)

#     x_ups = np.linspace(0, psf.shape[0], len(x) * UPSCALE_RATIO) * Z_STEP

#     peak_xups = x_ups[np.argmax(cs(x_ups))] 

#     fit = cs(x_ups)
    
#     peak = max(fit)
#     low = min(fit)
#     half_max = (peak - low) / 2
    
#     peak_idx = np.argmax(fit)
#     center_x = len(x_ups) / 2
    
#     half_max_crossings = np.where(np.diff(np.sign(fit-half_max)))[0]
#     if len(half_max_crossings) < 2:
#         print(half_max_crossings)
#         bad_psfs_idx.append(i)
#         show_psf_axial(psf)
#         plt.plot(x-peak, inten, label='raw')
#         plt.plot(x_ups-peak, cs(x_ups), label='fit')
#         plt.legend()
#         plt.show()
    
#     if DEBUG:
#         show_psf_axial(psf)
#         plt.plot(x-peak, inten, label='raw')
#         plt.plot(x_ups-peak, cs(x_ups), label='fit')
#         plt.legend()
#         plt.show()
#     return peak_xups

# offsets = np.array([find_peak(i, psf) for i, psf in tqdm(enumerate(psfs))])

# good_idx = [i for i in range(len(psfs)) if i not in bad_psfs_idx]

# offsets = offsets[good_idx]
# psfs = psfs[good_idx]
# locs = locs.iloc[good_idx]


# In[7]:


# Spline sharpness peak finding
from tqdm import tqdm
import numpy as np

from scipy.interpolate import UnivariateSpline
from data.align_psfs import norm_zero_one
from sklearn.metrics import mean_squared_error
from scipy import signal
import cv2
import numpy as np

plt.rcParams['figure.figsize'] = [5, 3]

DEBUG = True

UPSCALE_RATIO = 10

bad_psfs_idx = []

def denoise(img):
    from scipy.ndimage import gaussian_filter
    
    sigmas = np.array([3, 2, 2])
    return gaussian_filter(img.copy(), sigma=sigmas)

def get_sharpness(array):
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def reduce_img(psf):
    return np.stack([get_sharpness(x) for x in psf])
#     return psf.max(axis=(1,2))

def find_peak(i, raw_psf):
    if raw_psf.ndim == 4:
        raw_psf = raw_psf.mean(axis=-1)
    x = np.arange(raw_psf.shape[0]) * Z_STEP
    psf = denoise(raw_psf)
    
    inten = norm_zero_one(reduce_img(psf))

    cs = UnivariateSpline(x, inten, k=3, s=0.8)

    x_ups = np.linspace(0, psf.shape[0], len(x) * UPSCALE_RATIO) * Z_STEP

    peak_xups = x_ups[np.argmax(cs(x_ups))] 

    fit = cs(x_ups)
    
    peak = max(fit)
    low = min(fit)
    half_max = (peak - low) / 2
    
    peak_idx = np.argmax(fit)
    center_x = len(x_ups) / 2
    
    half_max_crossings = np.where(np.diff(np.sign(fit-half_max)))[0]
    if len(half_max_crossings) != 2:
        bad_psfs_idx.append(i)
        plt.plot(x, norm_zero_one(raw_psf.max(axis=(1,2))), label='max')
        plt.plot(x-peak, inten, label='smoothed')
        plt.plot(x_ups-peak, cs(x_ups), label='fit')
        plt.xlabel('z (nm)')
        plt.ylabel('Norm. pixel intensity')
        plt.legend()
        plt.show()
        show_psf_axial(psf)
    
#     if DEBUG:
#         plt.plot(x-peak, inten, label='smoothed')
#         plt.plot(x_ups-peak, cs(x_ups), label='fit')
#         plt.legend()
#         plt.show()
#         show_psf_axial(psf)
    return peak_xups

offsets = np.array([find_peak(i, psf) for i, psf in tqdm(enumerate(psfs))])

good_idx = [i for i in range(len(psfs)) if i not in bad_psfs_idx]

# offsets = offsets[good_idx]
# psfs = psfs[good_idx]
# locs = locs.iloc[good_idx]


# In[8]:


# # Roll peak finding

# from skimage.filters import gaussian
# from data.align_psfs import norm_zero_one

# Z_STEP = 20
# def norm_sum_imgs(psf):
#     psf_sums = psf.sum(axis=(1,2, 3))
#     psf = psf / psf_sums[:, np.newaxis, np.newaxis, np.newaxis]
#     return psf

# def eval_roll(fixed, moving, roll, debug=False):
#     fixed_section = fixed
#     moving_section = moving    
#     if roll < 0:
#         moving_section = moving[-roll:]
#         fixed_section = fixed_section[:moving_section.shape[0]]
#     elif roll > 0:
#         fixed_section = fixed[roll:]
#         moving_section = moving_section[:fixed_section.shape[0]]

#     score = ((fixed_section-moving_section)**2).sum()
#     if debug:
#         plt.rcParams['figure.figsize'] = [10, 3]
#         x = np.arange(fixed.shape[0])
#         x_moving = np.arange(moving.shape[0]) + roll
#         plt.title(f'full roll {roll}')
#         plt.plot(x_moving, moving.max(axis=(1,2)), label='moving')
#         plt.plot(x, fixed.max(axis=(1,2)), label='fixed')
#         plt.xlim((-5, 200))
#         plt.legend()
#         plt.show()
        
#         plt.title(f'{round(score, 2)} length {moving_section.shape[0]} roll {roll}')
#         plt.plot(moving_section.max(axis=(1,2)), label='moving')
#         plt.plot(fixed_section.max(axis=(1,2)), label='fixed')
#         plt.legend()
#         plt.show()   
        

#         show_psf_axial(fixed.mean(axis=-1), 'fixed', 3)
#         show_psf_axial(moving.mean(axis=-1), 'moving', 3)
#         show_psf_axial(np.roll(moving.mean(axis=-1), roll, 0), f'shifted {roll}', 3)
        

#     return score

# def find_best_roll(fixed, moving):
#     roll_min = -fixed.shape[0] // 2
#     roll_max = fixed.shape[0] // 2
#     best_roll = None
#     min_score = np.inf
    
#     rolls = []
#     scores = []
#     for roll in range(roll_min, roll_max, 1):
#         score = eval_roll(fixed, moving, roll, debug=False)
#         if score < min_score:
#             best_roll = roll
#             min_score = score
#         rolls.append(roll)
#         scores.append(score)
#     plt.plot(rolls, scores)
#     plt.xlabel('roll')
#     plt.ylabel('MSE')
#     plt.show()
#     eval_roll(fixed, moving, best_roll, debug=True)
#     return best_roll * Z_STEP
        


# psf1 = norm_zero_one(gaussian(psfs[1]).copy())

# psf2 = norm_zero_one(gaussian(psfs[3]).copy())

# roll = find_best_roll(psf1, psf2)

# # z = np.arange(psf1.shape[0]) * Z_STEP
# # z2 = z - roll

# # zs = np.concatenate((z, z2))
# # psf = np.concatenate((psf1, psf2))
# # idx = np.argsort(zs)
# # psf = psf[idx]
# # plt.imshow(grid_psfs(psf.squeeze()))
# # plt.show()



# In[9]:


ys = []
for i, offset in enumerate(offsets):
    zs = ((np.arange(psfs.shape[1])) * Z_STEP) -offset
    ys.append(zs)

ys = np.array(ys)


# In[10]:


# In[11]:


# Stratify according to area of FOV

from sklearn.preprocessing import KBinsDiscretizer
def cart2pol(xy):
    x, y = xy
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

center = locs[['x', 'y']].mean().to_numpy()
coords = locs[['x', 'y']].to_numpy() - center

polar_coords = np.stack([cart2pol(xy) for xy in coords])

discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal')
groups = discretizer.fit_transform(polar_coords[:, 1:2]).astype(str)

center_radius = 150
idx = np.argwhere(polar_coords[:, 0] <= center_radius).squeeze()
groups[idx] = -1

groups[:] = 0
locs['group'] = groups

sns.scatterplot(data=locs, x='x', y='y', hue='group')


# In[12]:


# Withold some PSFs for evaluation

from sklearn.model_selection import train_test_split

SEED = 42

idx = np.arange(psfs.shape[0])

train_idx, test_idx = train_test_split(idx, train_size=0.9, random_state=SEED, stratify=locs['group'])

_train_val_psfs = psfs[train_idx]
test_psfs = psfs[test_idx]

_train_val_ys = ys[train_idx]
test_ys = ys[test_idx]

train_fov_groups = locs['group'].to_numpy()[train_idx]

train_val_coords = locs[['x', 'y']].to_numpy()[train_idx]
test_coords = locs[['x', 'y']].to_numpy()[test_idx]

ds_cls = np.zeros((psfs.shape[0]), dtype=object)
ds_cls[train_idx] = 'train/val'
ds_cls[test_idx] = 'test'
locs['ds'] = ds_cls
plt.rcParams['figure.figsize'] = [5, 5]

# sns.scatterplot(data=locs, x='x', y='y', hue='ds')
# plt.show()


# In[13]:


groups = np.repeat(np.arange(len(train_idx))[:, np.newaxis], psfs.shape[1], axis=1).flatten()

coords = np.repeat(train_val_coords[:, :, np.newaxis], psfs.shape[1], axis=0)

train_val_psfs = np.concatenate(_train_val_psfs)
train_val_ys = np.concatenate(_train_val_ys)
split_idx = np.arange(train_val_psfs.shape[0])

train_idx, val_idx = train_test_split(split_idx, train_size=0.9, random_state=SEED, stratify=groups)

train_psfs = train_val_psfs[train_idx]
train_ys = train_val_ys[train_idx][:, np.newaxis]

val_psfs = train_val_psfs[val_idx]
val_ys = train_val_ys[val_idx][:, np.newaxis]

val_coords = coords[val_idx].squeeze()
train_coords = coords[train_idx].squeeze()



# In[14]:


# print(train_psfs.shape, train_ys.shape, _train_groups.shape)
# print(val_psfs.shape, val_ys.shape, _val_groups.shape)


# In[15]:


# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder().fit(_train_groups)

# train_groups = encoder.transform(_train_groups).toarray()
# val_groups = encoder.transform(_val_groups).toarray()

# print(train_psfs.shape, train_ys.shape, train_groups.shape)
# print(val_psfs.shape, val_ys.shape, val_groups.shape)


# In[16]:


# Trim stacks

def filter_z_range(X, zs):
    psfs, groups = X
    valid_ids = np.argwhere(abs(zs.squeeze()) < Z_RANGE).squeeze()
    return [psfs[valid_ids], groups[valid_ids]], zs[valid_ids]
    
Z_RANGE = 1000
X_train, y_train = filter_z_range((train_psfs, train_coords), train_ys)

X_val, y_val = filter_z_range((val_psfs, val_coords), val_ys)


# In[17]:


# data augmentation


from tensorflow.keras import layers, Sequential
from data.visualise import grid_psfs

def aug_dataset(X_train, y_train):
    AUG_RATIO = 2
    MAX_TRANSLATION_PX = 2
    MAX_GAUSS_NOISE = 0.005
    img_size = X_train[0].shape[1]

    aug_pipeline = Sequential([
        layers.GaussianNoise(stddev=MAX_GAUSS_NOISE*X_train[0].max(), seed=SEED),
        layers.RandomTranslation(MAX_TRANSLATION_PX/img_size, MAX_TRANSLATION_PX/img_size, seed=SEED),
        layers.RandomBrightness(0.2, [X_train[0].min(), X_train[0].max()], seed=SEED)
    ])

    idx = np.random.randint(0, X_train[0].shape[0], size=int(AUG_RATIO*X_train[0].shape[0]))

    aug_psfs = aug_pipeline(X_train[0][idx].copy(), training=True).numpy()
    aug_coords = X_train[1][idx]

    aug_z = y_train[idx]

    subset_psfs = np.concatenate((aug_psfs[0:100], X_train[0][idx][0:100]))
    # plt.imshow(grid_psfs(subset_psfs.mean(axis=-1)))
    # plt.show()

    train_psfs = np.concatenate([aug_psfs, X_train[0]])
    train_coords = np.concatenate([aug_coords, X_train[1]])
    train_zs = np.concatenate([aug_z, y_train])

    X_train = [train_psfs, train_coords]
    y_train = train_zs
    return X_train, y_train

X_train[0] = X_train[0].astype(float)
print(X_train[0].shape, X_train[0].min(), X_train[0].max(), X_train[0].mean())
X_train, y_train = aug_dataset(X_train, y_train)
print(X_train[0].shape, X_train[0].min(), X_train[0].max(), X_train[0].mean())

X_train[0] = X_train[0].astype(np.uint16)


# In[ ]:


from skimage.transform import resize

def resize_psfs(X):
    print('Resizing...')
    target_size = 128
#     target_size = 32
    imshape = (target_size, target_size, 3)
    X[0] = np.stack([resize(psf, imshape) for psf in X[0].astype(np.float32)]).astype(np.float32)
    print(X[0].shape)
    print('Finished')

resize_psfs(X_train)
resize_psfs(X_val)


# In[ ]:


print(X_train[0].shape, X_train[1].shape, y_train.shape)
print(X_val[0].shape, X_val[1].shape, y_val.shape)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


datagen = ImageDataGenerator(
    rescale=1.0/65336.0,
    samplewise_center=False,
    samplewise_std_normalization=False,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=False)

print('Fitting datagen...')
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(X_train[0])
print('Fitted')

X_train_preproc = [X_train[0].copy(), X_train[1].copy()]
X_val_preproc = [X_val[0].copy(), X_val[1].copy()]

X_train_preproc[0] = datagen.standardize(X_train_preproc[0].astype(float))
X_val_preproc[0] = datagen.standardize(X_val_preproc[0].astype(float))

# preprocessors = {
#     'psfs': datagen,
#     'coords': coords_scaler
# }

# import pickle
# with open('./scalers.p', 'wb') as f:
#     pickle.dump(preprocessors, f)


# In[ ]:


test_psfs = psfs[test_idx]

test_ys = ys[test_idx]


test_groups = np.repeat(np.arange(len(test_psfs))[:, np.newaxis], test_psfs.shape[1], axis=1)
test_groups = np.concatenate(test_groups)

test_coords = np.repeat(test_coords[:, :, np.newaxis], test_psfs.shape[1], axis=0).squeeze()

test_psfs = np.concatenate(test_psfs)


test_ys = np.concatenate(test_ys)[:, np.newaxis]


def filter_z_range(X, zs):
    psfs, groups = X
    valid_ids = np.argwhere(abs(zs.squeeze()) < Z_RANGE).squeeze()
    return [psfs[valid_ids], groups[valid_ids]], zs[valid_ids]


X_test, y_test = filter_z_range((test_psfs, test_coords), test_ys)

test_groups = X_test[1].copy()


resize_psfs(X_test)


X_test_preproc = [X_test[0].copy(), X_test[1].copy()]
X_test_preproc[0] = datagen.standardize(X_test_preproc[0].astype(float))



# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_preproc[1] = scaler.fit_transform(X_train_preproc[1])
X_val_preproc[1] = scaler.transform(X_val_preproc[1])
X_test_preproc[1] = scaler.transform(X_test_preproc[1])


# In[ ]:

def fix_psf_dtype(psfs):
    return psfs.astype(np.float32)


X_train_preproc[0] =  fix_psf_dtype(X_train_preproc[0])
X_val_preproc[0] = fix_psf_dtype(X_val_preproc[0])
X_test_preproc[0] = fix_psf_dtype(X_test_preproc[0])

print(X_train_preproc[0].min(), X_train_preproc[0].max())
print(X_val_preproc[0].min(), X_val_preproc[0].max())
print(X_test_preproc[0].min(), X_test_preproc[0].max())

print(X_train_preproc[1].min(), X_train_preproc[1].max())
print(X_val_preproc[1].min(), X_val_preproc[1].max())
print(X_test_preproc[1].min(), X_test_preproc[1].max())

print(X_train_preproc[0].shape, X_train_preproc[1].shape)
print(X_val_preproc[0].shape, X_val_preproc[1].shape)
print(X_test_preproc[0].shape, X_test_preproc[1].shape)

print(X_train_preproc[0].dtype, X_train_preproc[1].dtype)
print(X_val_preproc[0].dtype, X_val_preproc[1].dtype)
print(X_test_preproc[0].dtype, X_test_preproc[1].dtype)


import tensorflow as tf
train_data = tf.data.Dataset.from_tensor_slices(((X_train_preproc[0].astype(np.float32), X_train_preproc[1].astype(np.float32)), y_train.astype(np.float32)))
val_data = tf.data.Dataset.from_tensor_slices(((X_val_preproc[0].astype(np.float32), X_val_preproc[1].astype(np.float32)), y_val.astype(np.float32)))

# In[ ]:


# train_idx = np.argwhere(abs(y_train.squeeze()) < 50).squeeze()
# tmp_psfs = X_train_preproc[0][train_idx].mean(axis=-1)
# print(tmp_psfs.shape)
# plt.rcParams['figure.figsize'] = [10, 10]
# plt.imshow(grid_psfs(tmp_psfs))
# plt.show()


# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers, Sequential
# from tensorflow.keras import optimizers
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tqdm.keras import TqdmCallback
# from tensorflow.keras import regularizers

# import tensorflow as tf

# def train_model(X_train_preproc, y_train, X_val_preproc, y_val):
#     img_input = layers.Input((X_train_preproc[0][0].shape))
#     x = img_input
    
#     coords_input = layers.Input(X_train_preproc[1][0].shape)
#     x_coords = layers.Dense(64)(coords_input)
    
#     x_coords = layers.Dense(64)(x_coords)

#     x = keras.applications.ResNet101V2(
#         input_tensor = img_input,
#         include_top=False,
#         pooling='max',
#         weights='imagenet',
#     )(x)

# #     x = keras.applications.MobileNetV3Small(
# #         input_tensor=img_input,
# #         include_top=False,
# #         pooling='avg',
# #     )(x)
# #     x = keras.applications.MobileNet(
# #         input_tensor=img_input,
# #         include_top=False,
# #         weights='../mobilenet_1_0_128_tf_no_top.h5',
# #         pooling='max',
# #     )(x)

#     x = tf.concat([x, x_coords], axis=-1)
    
# #     x = layers.Dense(64, activation='relu')(x)
# #     x = layers.Dropout(0.5)(x)
# #     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     out = layers.Dense(1, activation="linear",
#                        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                        bias_regularizer=regularizers.L2(1e-4),
#                        activity_regularizer=regularizers.L2(1e-5)
#                       )(x)

#     model = keras.Model(inputs=(img_input, coords_input), outputs=out)

#     model.summary(expand_nested=False)


#     batch_size = 256
#     epochs = 5000
#     lr = 0.0001

#     model.compile(loss='mean_squared_error', optimizer=optimizers.AdamW(learning_rate=lr), metrics=['mean_absolute_error'])

#     callbacks = [
#         ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
#                           patience=50, verbose=True, mode='min', min_delta=5, min_lr=1e-6,),
#         EarlyStopping(monitor='val_mean_absolute_error', patience=75,
#                       verbose=False, min_delta=1, restore_best_weights=True),
#         TqdmCallback(verbose=1),
#     ]


#     history = model.fit(X_train_preproc, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val_preproc, y_val), callbacks=callbacks, shuffle=True, verbose=True)

# #     model.save('./latest_model')
#     return model, history

# # X_train_preproc[1][:] = 0
# # X_val_preproc[1][:] = 0
# model, history = train_model(X_train_preproc, y_train, X_val_preproc, y_val)


# In[ ]:


# Vision transformer training

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm.keras import TqdmCallback
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from vit_keras import vit

# Assuming your input images have size (image_size, image_size, num_channels)
image_size = X_train_preproc[0].shape[1]
num_channels = X_train_preproc[0].shape[-1]
num_classes = 1  # Regression task, predicting a single continuous value

def get_model():
   # Create the Vision Transformer model using the vit_keras library
   inputs = Input(shape=(image_size, image_size, num_channels))
   
   coords_input = layers.Input(X_train_preproc[1][0].shape)
   x_coords = layers.Dense(64)(coords_input)
   
   x_coords = layers.Dense(64)(x_coords)
   
   
   vit_model = vit.vit_b16(image_size=image_size, 
                           activation='sigmoid',
                           pretrained=True,
                           include_top=False,
                           pretrained_top=False)
   
   x = vit_model(inputs)
   # Add additional layers for regression prediction
   x = Flatten()(x)
   x = tf.concat([x, x_coords], axis=-1)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.5)(x)
   x = Dense(64, activation='relu')(x)
   x = Dropout(0.5)(x)
   regression_output = Dense(num_classes, activation='linear')(x)  # Linear activation for regression
   model = Model(inputs=[inputs, coords_input], outputs=regression_output)

   return model

batch_size = 64
epochs = 5000
lr = 0.0001

train_data = train_data.batch(batch_size).cache().prefetch(1024)
val_data = val_data.batch(batch_size).cache().prefetch(1024)



# # Model refining
# model = keras.models.load_model('./latest_vit_model/')
   
# n_layers = len(model.layers)
# for i in range(0, len(model.layers)-4):
#     model.layers[i].trainable = False
# assert model.trainable == True

   
# # Print a summary of the model architecture
# model.summary()

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# # Open a strategy scope.
with strategy.scope():
   # Combine the Vision Transformer backbone with the regression head
    model = get_model()

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizers.AdamW(learning_rate=lr), metrics=['mean_absolute_error'])


callbacks = [
   ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1,
                     patience=50, verbose=True, mode='min', min_delta=5, min_lr=1e-6,),
   EarlyStopping(monitor='val_mean_absolute_error', patience=75,
                 verbose=False, min_delta=1, restore_best_weights=True),
#    TqdmCallback(verbose=1),
]


history = model.fit(X_train_preproc, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val_preproc, y_val), callbacks=callbacks, shuffle=True, verbose=True)


# In[ ]:


# tf.keras.utils.plot_model(
#     model,
#     to_file="model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False,
#     show_trainable=False,
# )


# In[ ]:


print('finished!')


# In[ ]:


# Prev aug settings
# AUG_RATIO = 5
# MAX_TRANSLATION_PX = 2
# MAX_GAUSS_NOISE = 0.001


# In[ ]:


model.save('./latest_vit_model_openframe2')


# In[ ]:


#     AUG_RATIO = 1
#     MAX_TRANSLATION_PX = 2
#     MAX_GAUSS_NOISE = 0.001
#     img_size = X_train[0].shape[1]

#     aug_pipeline = Sequential([
#         layers.GaussianNoise(stddev=MAX_GAUSS_NOISE*X_train[0].max(), seed=SEED),
#         layers.RandomTranslation(MAX_TRANSLATION_PX/img_size, MAX_TRANSLATION_PX/img_size, seed=SEED),
#         layers.RandomBrightness(0.2, [X_train[0].min(), X_train[0].max()], seed=SEED)
#     ])


# Peak illumination
# Un-filtered spline alignment
# 446/446 [==============================] - 19s 39ms/step
# train 46.027
# 16/16 [==============================] - 1s 39ms/step
# val 52.553
# 19/19 [==============================] - 1s 42ms/step
# test 78.961 w/o, 44nm with bias removal
# Mean test offset: 67.992

# Denoised spline alignment (gaussian kernel 2, 1, 1)
# 446/446 [==============================] - 18s 37ms/step
# train 43.796
# 16/16 [==============================] - 1s 36ms/step
# val 46.722
# 19/19 [==============================] - 1s 37ms/step
# test 73.313 w/o bias removal, 36nm with bias removal
# mean test offset: 62
 




# In[ ]:


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [5, 5]
fig, ax1 = plt.subplots()
ax1.plot(history.history['mean_absolute_error'], label='mse')
ax1.plot(history.history['val_mean_absolute_error'], label='val_mse')
ax1.set_ylim([0, 500])
ax1.legend(loc=1)
ax2 = ax1.twinx()
ax2.plot(history.history['lr'], label='lr', color='red')
ax2.legend(loc=0)


# In[ ]:


get_ipython().system('realpath ./latest_vit_model/')


# In[ ]:





# In[ ]:


# import pickle
# import keras

# # with open('./scalers.p', 'rb') as f:
# #     preprocessors = pickle.load(f)

# model = keras.models.load_model('./latest_vit_model/')

# # datagen = preprocessors['psfs']
# # coords_scaler = preprocessors['coords']


# In[ ]:


from sklearn.metrics import mean_absolute_error
ds = [
    ('train', (X_train_preproc, y_train)), 
    ('val', (X_val_preproc, y_val)),
    ('test', (X_test_preproc, y_test))
]
for k, (X, y) in ds:
    res = model.predict(X, verbose=True)
    error = mean_absolute_error(res, y)
    print(k, round(error, 3))


# In[ ]:


# MAE without located error
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
plt.rcParams['figure.figsize'] = [10,5]

# def plot_psf_inten(psfs, z):
#     plt.plot(psfs.max(axis=(1,2)))
#     plt.show()
    
def bestfit_error(z_true, z_pred):
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = mean_absolute_error(y_fit, y)
    # plt.plot(x, x, label=f'x=y')
    # plt.plot(x, y_fit, label=f'best_fit c={round(popt[0], 3)}')
    # plt.scatter(z_true, z_pred, marker='x', c='orange')
    # plt.legend()
    # plt.xlabel('True z (nm)')
    # plt.ylabel('Predicted z (nm)')
    # plt.show()
    return error, popt[0], y_fit-y

ds = [
#     ('train', (X_train_preproc, y_train)),
    ('test', (X_test_preproc, y_test))
]

offsets = {}
errors = {}
for k, (X, y) in ds:
    pred_z = model.predict(X, verbose=False)
    offsets[k] = []
    errors[k] = []
    
    labels = X[1].astype(str)
    labels = [','.join(list(arr)) for arr in labels]
    label_ids = LabelEncoder().fit_transform(labels)
    y = y.squeeze()
    for g in set(label_ids):
        idx = np.argwhere(label_ids==g)[:, 0]
        group_psfs = X[0][idx]
        group_zs = y[idx]
        # plot_psf_inten(group_psfs, group_zs)

        # show_psf_axial(group_psfs.mean(axis=-1), '', 2)
        group_true_zs = y[idx]
        group_pred_zs = pred_z[idx][:, 0]
        if len(idx) == 1:
            res[k].append([mean_absolute_error(group_true_zs, group_pred_zs)])
        else:
            error, offset, _ = bestfit_error(group_true_zs, group_pred_zs)
            errors[k].append(error)
            offsets[k].append(offset)

for k in errors.keys():
    print(k)
    print('Error', round(np.mean(errors[k]), 3))
    print('Mean offset', round(np.mean(np.abs(offsets[k])), 3))
    print('\n')


# In[ ]:


for k in errors.keys():
    print(k)
    print('Error', round(np.mean(errors[k]), 3))
    print('Mean offset', round(np.mean(np.abs(offsets[k])), 3))
    print('\n')


# In[ ]:


# Error with xy coords
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder

def bestfit_error(z_true, z_pred):
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = mean_absolute_error(y_fit, y)
    plt.plot(x, x, label=f'x=y')
    plt.plot(x, y_fit, label=f'best_fit c={popt[0]}')
    plt.scatter(z_true, z_pred, marker='x', c='orange')
    return error, popt[0], y_fit-y

ds = [
    ('val', (X_val_preproc, y_val)),
    ('test', (X_test_preproc, y_test))
]

res = {}
for k, (X, y) in ds:
    pred_z = model.predict(X, verbose=False)
    res[k] = []
    labels = X[1].astype(str)
    labels = [','.join(list(arr)) for arr in labels]
    label_ids = LabelEncoder().fit_transform(labels)
    
    X2 = X[0].copy(), X[1].copy()
    X2[1][:] = 0
    pred_z_no_coords = model.predict(X2, verbose=False)
    
    y = y.squeeze()
    plt.scatter(y, pred_z, label='w/ coords', marker='.')
    plt.scatter(y, pred_z_no_coords, label='w/o coords', marker='.')
    plt.legend()
    plt.show()
    print(mean_absolute_error(y, pred_z))
    print(mean_absolute_error(y, pred_z_no_coords))

#     for g in set(label_ids):
#         idx = np.argwhere(label_ids==g)[:, 0]
#         group_psfs = X[0][idx]
#         show_psf_axial(group_psfs.mean(axis=-1), '', 2)
#         group_true_zs = y[idx]
#         group_pred_zs = pred_z[idx][:, 0]
#         if len(idx) == 1:
#             res[k].append([mean_absolute_error(group_true_zs, group_pred_zs)])
#         else:
#             error, offset, errors = bestfit_error(group_true_zs, group_pred_zs)
#             error, offset, errors = bestfit_error(group_true_zs, pred_z_no_coords[idx][:, 0])

#         plt.show()


# In[ ]:


for k, v in res.items():
    print(k, round(np.mean(np.abs(v)), 3))


# In[ ]:


# Results
# w/               groups    no groups   no groups larger FOV
# train            18.952    11.403      12
# val              55.52     53.929      68
# test             102.356   99.47       74
# test_wo_offsets  48.838    48.318      42

# w/ No reg        groups    no groups   no groups larger FOV
# train            ______    7.9___      ______
# val              ______    54____      ______
# test             ______    126___      ______
# test_wo_offsets  ______    84____      ______


# In[ ]:


import pickle
import keras

# with open('./scalers.p', 'rb') as f:
#     preprocessors = pickle.load(f)

model = keras.models.load_model('./latest_vit_model/')

# datagen = preprocessors['psfs']
# coords_scaler = preprocessors['coords']


# In[ ]:


import pandas as pd
import h5py
import numpy as np

# MQ_DATA
# dirname = '/home/miguel/Projects/uni/data/smlm_3d/20230601_MQ_celltype/nup/fov2/storm_1/'
# locs = 'storm_1_MMStack_Default.ome_locs_undrift.hdf5'
# spots = 'storm_1_MMStack_Default.ome_spots.hdf5'

# FD-DEEPLOC-data

dirname = '/home/miguel/Projects/uni/data/smlm_3d/fd-deeploc-data/demo2_FD_astig_NPC/'
locs = 'roi_startpos_810_790_split.ome_locs.hdf5'
spots = 'roi_startpos_810_790_split.ome_spots.hdf5'

all_locs = pd.read_hdf(dirname+locs, key='locs')
picked_locs = pd.read_hdf(dirname+locs.replace('_locs', '_locs_picked'), key='locs')

with h5py.File(dirname+spots, 'r') as f:
    spots = np.array(f['spots']).astype(np.uint16)

print(all_locs.shape)
print(picked_locs.shape)
print(spots.shape)


# In[ ]:


# MQ_data_only
if '20230601_MQ_celltype' in dirname:
    sns.scatterplot(data=all_locs, x='x', y='y', marker='.')
    plt.rcParams['figure.figsize'] = [20, 20]

#     xlim = 500, 600
#     ylim = 300, 400
#     plt.xlim(*xlim)
#     plt.ylim(*ylim)
#     xlim = ((450, 750))
#     ylim = ((450, 750))

    
    xlim = 105, 110
    ylim = 60, 65
    l2 = picked_locs[(xlim[0]<picked_locs['x']) & (xlim[1]>picked_locs['x']) & (ylim[0]<picked_locs['y']) & (ylim[1]>picked_locs['y'])]
    all_locs = all_locs.iloc[l2.index]
    picked_locs = all_locs
    spots = spots[l2.index]

    print(all_locs.shape)
    print(spots.shape)


# In[ ]:


if 'demo2_FD_astig_NPC' in dirname:

#     print(all_locs['x'].max())
#     xlim = 150, 170
#     ylim = 130, 150
    
    xlim = 50, 120
    ylim = 50, 120
    l2 = picked_locs[(xlim[0]<picked_locs['x']) & (xlim[1]>picked_locs['x']) & (ylim[0]<picked_locs['y']) & (ylim[1]>picked_locs['y'])]
    all_locs = all_locs.iloc[l2.index]
    picked_locs = all_locs
    spots = spots[l2.index]

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    sns.scatterplot(data=all_locs, x='x', y='y', alpha=0.1)
    plt.show()

    all_locs['x'] += 810
    all_locs['y'] += 790
print(all_locs.shape)
print(picked_locs.shape)
print(spots.shape)


# In[ ]:


if all_locs.shape[0] == picked_locs.shape[0]:
    idx = np.arange(all_locs.shape[0])
else:
    all_keys = list(all_locs[['bg', 'photons']].astype(str).agg('-'.join, axis=1))
    picked_keys = picked_locs[['bg', 'photons']].astype(str).agg('-'.join, axis=1)
    idx = [all_keys.index(k) for k in picked_keys]

exp_psfs = spots[idx]
print(exp_psfs.shape, picked_locs.shape)
print(exp_psfs.min(), exp_psfs.max())
try:
    print(psfs.min(), psfs.max())
    print(psfs.dtype, exp_psfs.dtype)
except NameError:
    pass


# In[ ]:


import matplotlib.pyplot as plt
from data.visualise import grid_psfs
plt.rcParams['figure.figsize'] = [10, 10]
plt.imshow(grid_psfs(exp_psfs[0:100]))
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [3, 3]
sns.scatterplot(data=picked_locs, x='x', y='y', alpha=0.1)
plt.show()


# In[ ]:


# print(exp_psfs_preproc.min(), exp_psfs_preproc.max())
# print(X_train_preproc[0].min(), X_train_preproc[0].max())
# print(exp_coords_preproc.min(), exp_coords_preproc.max())
# print(exp_psfs_preproc.shape, exp_coords_preproc.shape)


# In[ ]:


print(exp_psfs.dtype, psfs.dtype)


# In[ ]:


import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 3]

exp_coords = picked_locs[['x', 'y']].to_numpy()
exp_coords_preproc = scaler.transform(exp_coords)


# In[ ]:


X_exp = [exp_psfs, exp_coords_preproc]
resize_psfs(X_exp)
X_exp[0] = datagen.standardize(X_exp[0].astype(float))


# In[ ]:


for X in (X_exp, X_train_preproc):
    print(X[0].min(), X[0].mean(), X[0].max())
    print(X[1].min(), X[1].max())


# In[ ]:


pred_z = model.predict(X_exp)
plt.rcParams['figure.figsize'] = [3,3]
sns.histplot(pred_z)
plt.show()




# In[ ]:


picked_locs['z'] = pred_z

picked_locs.to_csv('./locs_w_z.csv')


# In[ ]:


from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from natsort import natsorted

picked_locs = pd.read_csv('./locs_w_z.csv')

param_grid = {
    "n_components": range(1, 5),
    "covariance_type": ["full"],
    "n_init": [10],
}


def get_n_components(data, override_params=None):
    
    res = []
    estimators = []
    for param in ParameterGrid(param_grid):
        if override_params:
            param.update(override_params)
        gm = GaussianMixture(**param).fit(data)
        param['score'] = gm.bic(data)
        res.append(param)
        estimators.append(gm)
    
    df = pd.DataFrame.from_records(res)
    best_params = np.argmin(df['score'].to_numpy())
    return estimators[best_params]



def get_cov(gm, i):
    cov_type = gm.covariance_type
    if cov_type == 'tied':
        cov = gm.covariances_.squeeze()
    elif cov_type == 'full' or cov_type == None:
        cov = gm.covariances_[i][0][0]
    elif cov_type == 'spherical':
        cov = gm.covariances_[i]
    elif cov_type == 'diag':
        cov = gm.covariances_[i]
    
    return cov


def fit_gmm(cluster_locs, override_params=None):
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:4])
    
    hist_params = {
        'stat': 'density',
        'ax': ax2,
        'multiple': 'stack',
    }
    
    data = cluster_locs[['z']].to_numpy()
    # Remove bottom % and top %
#     plow = np.percentile(data, 1)
#     phigh = np.percentile(data, 99)
    
#     data = data.squeeze()
#     data = data[(plow <= data) & (data <= phigh)]
#     data = data[:, np.newaxis]
    
    sns.scatterplot(data=cluster_locs, x='x', y='y', marker='.', hue='sub_clusterID', ax=ax1, legend=False)

    point_count = len(data)
    title = f'N={point_count}'
        
    if len(data) < 10:
        ax.set_title(title)
        sns.histplot(data, **hist_params)
        return


    gm = get_n_components(data, override_params=None)
    df = pd.DataFrame.from_dict({
        'z': data.squeeze(),
        'labels': gm.predict(data).squeeze().astype(str)
    })
    
    sns.histplot(data=df, x='z', **hist_params)
    sns.kdeplot(data=df, x='z', bw_adjust=0.5)
    
    x = np.linspace(data.min(), data.max(), 100)
    
    for i in range(gm.n_components):
        cov = get_cov(gm, i)
        mean = float(gm.means_[i][0])
        weight = gm.weights_[i]
        ax2.plot(x, norm.pdf(x, mean, np.sqrt(cov))*weight, label=str(i))
        ax2.vlines(mean, 0, 0.01, label=str(round(mean)))
    
    print('Means', [round(x, 3) for x in sorted(gm.means_[:, 0])])
    if gm.n_components > 1:
        diff_dist = [round(x) for x in np.diff(sorted(gm.means_.squeeze()))]
#         title += f' - {diff_dist}nm'
    ax.set_title(title)
    ax2.set_xlim((-50, 500))
    return gm
    

xyz = picked_locs[['x', 'y', 'z']].to_numpy()
xy = xyz[:, [0, 1]]

cls = DBSCAN(eps=0.3, min_samples=15).fit_predict(xy).astype(str)

picked_locs['clusterID'] = cls

plt.rcParams['figure.figsize'] = [5, 5]
sns.scatterplot(data=picked_locs, x='x', y='y', hue='clusterID', legend=False)
plt.show()


plt.rcParams['figure.figsize'] = [10, 3]
for cluster_id in natsorted(set(cls)):
    if cluster_id == '-1':
        print('Ignoring cluster -1')
        continue
        
        
    cluster_locs = picked_locs[picked_locs['clusterID'] == cluster_id].copy()
    cluster_coords = cluster_locs[['x', 'y']].to_numpy()
    
    cluster_locs['sub_clusterID'] = KMeans(n_init=8, n_clusters=8).fit_predict(cluster_coords).astype(str)
    
    if cluster_locs.shape[0] < 50:
        continue
        
    if int(cluster_id) not in [60]:
        continue
        
        
    tmp_cid = int(cluster_id)
    tmp_locs = cluster_locs
    print(f'Cluster ID {cluster_id}')
#     sns.histplot(data=cluster_locs, x='z', ax=ax2, bins=40, stat='density')
    fit_gmm(cluster_locs, {'covariance_type': 'full'})
    plt.show()

#     fig2 = plt.figure()
#     gs2 = fig2.add_gridspec(1, 8)
#     axes = []
#     for i in sorted(set(cluster_locs['sub_clusterID'])):
#         if int(i) > 0:
#             ax = fig2.add_subplot(gs2[0, int(i)], sharey=axes[0])
#             plt.setp(ax.get_yticklabels(), visible=False)
#         else:
#             ax = fig2.add_subplot(gs2[0, int(i)])
#         axes.append(ax)
        
#         z_data = cluster_locs[cluster_locs['sub_clusterID']==i]['z'].to_numpy()[:, np.newaxis]
#         fit_gmm(ax, z_data)
        
        
#     plt.show()
    



# In[ ]:


if tmp_cid == 40:
    tmp_locs = tmp_locs[tmp_locs['y'] < 870.5]
if tmp_cid == 60:
    tmp_locs = tmp_locs[tmp_locs['z'] > 0]


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

pred_rmse = 17.847
MIN_STD = pred_rmse
MAX_STD = pred_rmse

class BoundedStdGaussianMixture(GaussianMixture):
    def fit(self, X, y=None):
        # Fit the GMM using the standard fit method
        super().fit(X, y)

        # Enforce bounds on standard deviations
        min_std = MIN_STD  # Minimum allowed standard deviation
        max_std = MAX_STD   # Maximum allowed standard deviation

        for i in range(self.n_components):
            self.covariances_[i] = np.clip(self.covariances_[i], min_std**2, max_std**2)

            
    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        n = self.n_components + (self.n_components - 1)
        return n

        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        print(f'BIC N parameters', self._n_parameters())

        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        print(f'AIC N parameters', self._n_parameters())
        return -2 * self.score(X) * X.shape[0] + 2

        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

data = tmp_locs[['z']].to_numpy()
# Create a custom GMM with bounded standard deviations
for n_components in range(1, 6):
    print(f'N components {n_components}')
    gmm = BoundedStdGaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

    # Fit the GMM to your data
    gmm.fit(data.reshape(-1, 1))

    print('Means', gmm.means_.squeeze())
    print('AIC', gmm.aic(data))
    # Plot the data and the fitted GMM
    # x = np.linspace(data.min() - 1, data.max() + 1, 1000)
    # plt.hist(data, bins=30, density=True, alpha=0.5, color='blue')
    # plt.plot(x, np.exp(gmm.score_samples(x.reshape(-1, 1))), color='red', linewidth=2)
    # plt.xlabel('Data')
    # plt.ylabel('Probability Density')
    # plt.show()

    sns.histplot(data=tmp_locs, x='z', stat='density')
    ax = plt.gca()
    for i in range(gmm.n_components):
        cov = get_cov(gmm, i)
        mean = float(gmm.means_[i][0])
        weight = gmm.weights_[i]
        x = np.linspace(data.min(), data.max(), 100)
        dens = norm.pdf(x, mean, np.sqrt(cov))*weight
        ax.plot(x, dens, label=str(i))
        ax.vlines(mean, 0, max(dens), label=str(round(mean)))

    plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 3]

param_grid = {
    "n_components": range(1, 5),
    "covariance_type": ["full"],
}


def kde_plot(data):
    for bw_adjust in np.linspace(0.1, 1.5, 10):
        sns.histplot(data, bins=20, stat='density')
        plt.title('Kernel scaling factor:'+ str(round(bw_adjust, 3)))
        sns.kdeplot(data.squeeze(), bw_adjust=bw_adjust)
        plt.show()
    

def plot_gmm(data, gm):
    plt.rcParams['figure.figsize'] = [3, 3]
    sns.histplot(data, bins=20, stat='density')
    x = np.linspace(data.min(), data.max(), 100)
    for i in range(gm.n_components):
        cov = get_cov(gm, i)
        mean = float(gm.means_[i][0])
        weight = gm.weights_[i]
        plt.plot(x, norm.pdf(x, mean, np.sqrt(cov))*weight, label=str(i))
    plt.show()   
    plt.rcParams['figure.figsize'] = [5, 5]


def get_n_components(data):
    
    res = []
    estimators = []
    kde_plot(data)
    for param in ParameterGrid(param_grid):
        gm = GaussianMixture(**param).fit(data)
        param['score'] = gm.bic(data)
        res.append(param)
        estimators.append(gm)
        plt.title(str(param))
        plot_gmm(data, gm)

    
    df = pd.DataFrame.from_records(res)
    best_params = np.argmin(df['score'].to_numpy())
    return estimators[best_params]


def get_cov(gm, i):
    cov_type = gm.covariance_type
    if cov_type == 'tied':
        cov = gm.covariances_.squeeze()
    elif cov_type == 'full' or cov_type == None:
        cov = gm.covariances_[i][0][0]
    elif cov_type == 'spherical':
        cov = gm.covariances_[i]
    elif cov_type == 'diag':
        cov = gm.covariances_[i]
    return cov


def fit_gmm(ax, data):
    hist_params = {
        'stat': 'density',
        'ax': ax,
    }
    point_count = len(data)
    title = f'N={point_count}'

    if len(data) < 10:
        ax.set_title(title)
        sns.histplot(data, **hist_params)
        return


    gm = get_n_components(data)
    df = pd.DataFrame.from_dict({
        'z': data.squeeze(),
        'labels': gm.predict(data).squeeze().astype(str)
    })
    
    sns.histplot(data=df, x='z', hue='labels', bins=20, **hist_params)
    
    x = np.linspace(data.min(), data.max(), 100)
    
    for i in range(gm.n_components):
        cov = get_cov(gm, i)
        mean = float(gm.means_[i][0])
        weight = gm.weights_[i]
        ax.plot(x, norm.pdf(x, mean, np.sqrt(cov))*weight, label=str(i))
    
    if gm.n_components > 1:
        diff_dist = [round(x) for x in np.diff(sorted(gm.means_.squeeze()))]
#         title += f' - {diff_dist}nm'
    ax.set_title(title)
    

fig2 = plt.figure()
gs2 = fig2.add_gridspec(1, 8)
axes = []
for i in sorted(set(cluster_locs['sub_clusterID'])):
    if int(i) > 0:
        ax = fig2.add_subplot(gs2[0, int(i)])
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.ylim((0, 0.01))
    else:
        ax = fig2.add_subplot(gs2[0, int(i)])
    axes.append(ax)

    z_data = tmp_locs[tmp_locs['sub_clusterID']==i]['z'].to_numpy()[:, np.newaxis]

    fit_gmm(ax, z_data)
plt.show()
    


# In[ ]:


data = picked_locs['z'].to_numpy()[:, np.newaxis]
sns.histplot(data, stat='density', bins=100)


fit = get_n_components(data)
means = fit.means_
print(means)
print(fit.__dict__)
x = np.linspace(data.min(), data.max(), 100)

cov = fit.covariances_.squeeze()
for i in range(fit.n_components):
    mean = float(fit.means_[i][0])
    weight = fit.weights_[i]
    plt.plot(x, norm.pdf(x, m, np.sqrt(cov))*weight)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 20]

idx = np.argsort(pred_z.squeeze())
sorted_psfs = exp_psfs[idx]
plt.imshow(grid_psfs(sorted_psfs[::10]))


# In[ ]:


plt.rcParams['figure.figsize'] = [5, 3]
for sd in [40]:
    p1 = np.random.normal(0, sd, size=10000)
    p2 = np.random.normal(50, sd, size=10000)
    data = np.concatenate((p1, p2))
    plt.title(f'Stdev: {sd}')
    sns.histplot(data)
    plt.xlabel('Z (nm)')
    plt.show()


# In[ ]:




