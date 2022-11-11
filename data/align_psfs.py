from multiprocessing.spawn import prepare
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import trange
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product
from functools import partial
import tqdm
from data.estimate_offset import get_peak_sharpness
from data.visualise import show_psf_axial
from keras.metrics import mean_squared_error

UPSCALE_RATIO = 10

DEBUG = False

def find_outlier_pixels(data,tolerance=3,worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset. 
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    return blurred
    difference = data - blurred
    threshold = 10*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    if worry_about_edges == True:
        height,width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med
        print(len(hot_pixels))
        if len(hot_pixels) > 0:
            plt.imshow(np.concatenate((data, fixed_image, abs(data-fixed_image)), axis=1))
            plt.show()
    return fixed_image

def norm_zero_one(s):
    max_s = s.max()
    min_s = s.min()
    return (s - min_s) / (max_s - min_s)

def pad_and_fit_spline(coords, psf, z, z_ups):
    x, y = coords
    zs = psf[:, x, y]
    cs = UnivariateSpline(z, zs, k=3, s=1e-1)
    # if max(cs(z_ups)) > 2:
    #     plt.plot(z, zs, label='raw')
    #     plt.plot(z_ups, cs(z_ups), '.', label='smooth')
    #     plt.legend()
    #     plt.show()
    return x, y, cs(z_ups)
    
def upsample_psf(psf, ratio=UPSCALE_RATIO):
    pad_width = 5
    z = np.arange(-pad_width, psf.shape[0] + pad_width)
    z_ups = np.arange(0, psf.shape[0], 1/ratio)
    upsampled_psf = np.zeros((z_ups.shape[0], *psf.shape[1:]))
    
    psf = np.pad(psf, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
    xys = list(product(np.arange(psf.shape[1]), np.arange(psf.shape[2])))
    func = partial(pad_and_fit_spline, psf=psf, z=z, z_ups=z_ups)
    res = list(map(func, xys))
    # with Pool(8) as p:
    #     res = list(p.imap(func, xys))
    for x, y, z_col in res:
        upsampled_psf[:, x, y] = z_col

    return upsampled_psf

def pad_psf(psf):
    return np.pad(psf, ((20, 20), (0, 0), (0, 0)), mode='edge')

def mse(A,B):
    return ((A-B)**2).mean()

def plot_correction(target, img, psf_corrected, errors):
    if DEBUG:
        plt.plot(target.max(axis=(1,2)), label='target')
        plt.plot(img.max(axis=(1,2)),  label='original')
        plt.plot(psf_corrected.max(axis=(1,2)), label='corrected', )

        plt.legend()
        plt.show()

def tf_find_optimal_roll(target, img, upscale_ratio=UPSCALE_RATIO):
    ref_tf = tf.convert_to_tensor(target)
    img_tf = tf.convert_to_tensor(img)
    errors = []

    for i in range(img.shape[0]):
        error = tf.reduce_mean(mean_squared_error(ref_tf, img_tf))
        errors.append(error)
        img_tf = tf.roll(img_tf, 1, axis=0)

    best_i = tf.argmin(errors).numpy()
    # Prefer small backwards roll to large forwards roll
    if abs(best_i - img.shape[0]) < best_i:
        best_i = best_i - img.shape[0]

    psf_corrected = np.roll(img, int(best_i), axis=0)
    plot_correction(target, img, psf_corrected, errors)

    return best_i/upscale_ratio

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2).astype(int)

    mask = dist_from_center <= radius
    return mask

def mask_img_stack(stack, radius):
    mask = create_circular_mask(stack.shape[1], stack.shape[2], radius=radius)
    for i in range(stack.shape[0]):
        stack[i][~mask] = 0
    return stack

from skimage.filters import gaussian

def prepare_psf(psf):
    psf = gaussian(psf, sigma=1)
    psf = norm_zero_one(psf.copy())
    psf = pad_psf(psf)
    psf = upsample_psf(psf)
    # psf = mask_img_stack(psf, 12)
    return psf


import tensorflow as tf

from skimage.exposure import match_histograms
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def find_seed_psf(df):
    # Seed PSF - most centered PSF in FOV
    center = df[['x', 'y']].mean(axis=0).to_numpy()
    coords = df[['x', 'y']].to_numpy()
    dists = euclidean_distances([center], coords).squeeze()
    first_point = np.argmin(dists)
    return first_point

def gen_minimum_spanning_tree(df):
    coords = df[['x', 'y']].to_numpy()
    dists = euclidean_distances(coords, coords)
    m = csr_matrix(dists)
    adj_matrix = minimum_spanning_tree(m).toarray()
    adj_matrix[adj_matrix>0] = 1
    sim_adj_matrix = adj_matrix + adj_matrix.T
    return sim_adj_matrix

def get_or_prepare_psf(prepped_psfs, raw_psfs, idx):
    if idx not in prepped_psfs:
        prepped_psfs[idx] = prepare_psf(raw_psfs[idx])
    return prepped_psfs[idx]

def align_psfs(psfs, df):
    print(f'Aligning {psfs.shape} psfs...')

    seed_psf = find_seed_psf(df)

    ref_psf = prepare_psf(psfs[0])
    offsets = np.zeros((psfs.shape[0]))

    ref_0 = get_peak_sharpness(psfs[seed_psf])

    for i in trange(1, psfs.shape[0]):
        psf = psfs[i]
        psf = prepare_psf(psf)
        print(psf.min(), psf.max())
        print(ref_psf.min(), ref_psf.max())
        psf = match_histograms(psf, ref_psf)
        offset = tf_find_optimal_roll(ref_psf, psf)
        offsets[i] = offset
        if DEBUG:
            offset_psf = np.roll(psf, shift=-int(offset), axis=0)
            imgs = np.concatenate((ref_psf, offset_psf), axis=2)
            show_psf_axial(imgs, subsample_n=30)

    offsets -= ref_0

    return offsets