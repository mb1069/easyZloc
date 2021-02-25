import numpy as np
import scipy.io as sio
from skimage import io
import pandas as pd
import os
from glob import glob

# Use ImageJ macro to generate bead frames, then use this code to
# extract bead PSFs and positions
from src.data import image_processing


def process_jonny_datadir(directory, bound=16, y_dims=1, pixel_size=106, normalise_images=True, datasets=None):
    print("Processing Jonny data")
    # Storage arrays
    x_all = np.zeros((1, bound * 2, bound * 2))
    y_all = np.zeros(y_dims)
    sample_images = glob(os.path.join(directory, '*', 'MMStack_Default.ome.tif'))
    if datasets:
        sample_images = [sample_images[i] for i in datasets]
    print(f'Taking {len(sample_images)} datasets.')
    for image in sample_images:
        truth = os.path.join(os.path.dirname(image), 'stack', 'MMStack_Default.csv')
        x, y = process_jonny_tif(image, truth, bound=bound, pixel_size=pixel_size, normalise_images=normalise_images)
        x_all = np.append(x_all, x, axis=0)
        y_all = np.append(y_all, y)

    # Remove placeholder zeroth elements
    x_train = np.delete(x_all, 0, axis=0)
    y_train = np.delete(y_all, 0)

    # Reshape x_train for direct input into CNN
    # Add channel in
    print(f'Collected {x_train.shape[0]} datapoints')
    print(f'Z-range: {y_train.min()} - {y_train.max()}')
    return x_train, y_train


def process_jonny_tif(image, localisations_csv, pixel_size=106, bound=16, normalise_images=True, known_zpos=True):
    print(image)
    image = io.imread(image)
    truth = pd.read_csv(localisations_csv, skiprows=28)
    data = truth[["x0 (um)", "y0 (um)", "z0 (um)"]]
    # Convert to nm for thresholding
    for a, b in (("x0 (um)", "x0 (nm)"), ("y0 (um)", "y0 (nm)"), ("z0 (um)", "z0 (nm)")):
        data[b] = data[a] * 1000
        del data[a]
    pixel_x, pixel_y = round(data["x0 (nm)"] / pixel_size), round(data["y0 (nm)"] / pixel_size)
    pixel_locations = pd.DataFrame({"x": pixel_x, "y": pixel_y, "z": data["z0 (nm)"]})
    filtered_emitters = image_processing.filter_points(pixel_locations, bound=bound)
    x, y = image_processing.get_emitter_data(image, filtered_emitters, bound=bound,
                                             normalise=normalise_images, with_zpos=known_zpos)
    return x, y


def process_STORM_tif(image, localisations_csv, pixel_size=106, bound=16, normalise_images=True, known_zpos=True):
    image = io.imread(image)
    truth = pd.read_csv(localisations_csv)
    data = truth[["x [nm]", "y [nm]", "z [nm]"]]

    pixel_x, pixel_y = round(data["x [nm]"] / pixel_size), round(data["y [nm]"] / pixel_size)
    pixel_locations = pd.DataFrame({"x": pixel_x, "y": pixel_y, "z": data["z [nm]"]})

    filtered_emitters = image_processing.filter_points(pixel_locations, bound=bound)
    x, y = image_processing.get_emitter_data(image, filtered_emitters, bound=bound,
                                             normalise=normalise_images, with_zpos=known_zpos)
    return x, y


def process_STORM_datadir(directory, samples=500, bound=16, y_dims=1,
                          pixel_size=106, normalise_images=True):
    """
    Processes STORM-generated images and known truth positions.

    :param directory: Directory to where samples are stored
    :param samples: No. of samples to process
    :param bound: "radius" of window around emitter
    :param y_dims: output array shape
    :param pixel_size: Pixel size in nm
    :param normalise_images: min-max normalising
    :return: X_train, y_train which are PSF arrays and corresponding z-pos
    """
    print("Processing STORM data")
    # Storage arrays
    x_all = np.zeros((1, bound * 2, bound * 2))
    y_all = np.zeros(y_dims)

    for i in range(samples):
        image = f'{directory}/{i}.tif'
        truth = f'{directory}/{i}.csv'
        x, y = process_STORM_tif(image, truth, bound=bound, pixel_size=pixel_size, normalise_images=normalise_images)

        x_all = np.append(x_all, x, axis=0)
        y_all = np.append(y_all, y)

    # Remove placeholder zeroth elements
    x_train = np.delete(x_all, 0, axis=0)
    y_train = np.delete(y_all, 0)

    # Reshape x_train for direct input into CNN
    # Add channel in
    return x_train, y_train


def scale_zpos(zpos, center=None, scale=None):
    """
    Normal scaling of output z-positions

    :param zpos:
    :param center:
    :param scale:
    :return:
    """
    if center is not None and scale is not None:
        return zpos * scale + center

    center = np.mean(zpos)
    scale = np.std(zpos)
    normalised_data = (zpos - center) / scale

    return normalised_data, center, scale


def process_MATLAB_data(dirname, psf_path, zpos_path, normalise_images=True, is_multi=False):
    """
    Processes MATLAB-generated PSF arrays and z-pos.

    :param psf_path: Path to .m PSF array
    :param zpos_path: Path to .m z-pos array
    :param normalise_images: min-max scaling (16-bit)
    :return: X_train, y_train
    """
    print(f"Processing MATLAB data: {os.path.basename(psf_path)}")
    # Load matlab arrays
    PSFs = sio.loadmat(os.path.join(dirname, psf_path))
    Zpos = sio.loadmat(os.path.join(dirname, zpos_path))

    # Extract arrays into numpy
    psf = PSFs[list(PSFs.keys())[-1]]
    zpos = Zpos[list(Zpos.keys())[-1]]
    if (psf.shape[-1] != zpos.shape[0]) and not is_multi:
        raise ValueError("Warning: Number of PSFs and number of z-positions are different!")

    if normalise_images:
        psf = image_processing.normalise_image(psf)

    # Reshape arrays into standard input format for CNNs
    # Note that MATLAB array storage is different from numpy! Need to transpose!
    x_train = psf.transpose([2, 0, 1])

    y_train = zpos.reshape(zpos.shape[0])

    return x_train, y_train


def process_multiple_MATLAB_data(dirname, psf_paths: list, zpos_path, normalise_images=False):
    assert isinstance(psf_paths, list), "Only for lists of paths to multiple PSF arrays"

    X_train, y_train = process_MATLAB_data(dirname, psf_paths[0], zpos_path, normalise_images=normalise_images,
                                           is_multi=True)

    for psf_path in psf_paths[1:]:
        print(f"Processing MATLAB data: {os.path.basename(psf_path)}")
        x_, y_ = process_MATLAB_data(dirname, psf_path, zpos_path, normalise_images=normalise_images, is_multi=True)
        X_train = np.append(X_train, x_, axis=0)

    if X_train.shape[0] != y_train.shape[0]:
        print("Warning: Number of PSFs from combined files and number of z-positions are different!")
    return X_train, y_train

# def process_STORM_zstack(image, emitter_positions, z_data, bound=16, y_dims=1,
#                          intensity_threshold=10000, filter_points=True,
#                          pixel_size=106, normalise_images=True):
#     """
#     Extracts PSFs from a zstack of beads (image) with emitter_positions given by STORM.
#     The z-stack consists of defined starting z-pos, ending z-pos, and step. From this, a z-position
#     or depth can be calculated for each frame in the image. We then apply this z-position to points
#     in each frame and filter points/extract PSFs using previously defined functions.
#
#     :param filter_points: To filter points
#     :param z_data: a tuple/list of start, stop, step of z-positions.
#     :param image:
#     :param emitter_positions:
#     :param bound:
#     :param y_dims:
#     :param intensity_threshold:
#     :param pixel_size:
#     :param normalise_images:
#     :return:
#     """
#
#     # Check that the z_data = [start,stop,step] does indeed yield the expected number
#     # of frames in the image
#
#     start, stop, step = z_data
#     expected_frames = abs((stop - start)) / step + 1
#     actual_frames = image.shape[0]
#     assert expected_frames == actual_frames, "Check z-data, does not match frames of the image."
#     z_stack_depths = np.linspace(start, stop, actual_frames)
#
#     print("Processing STORM zstack")
#     # Storage arrays
#     x_all = np.zeros((1, bound * 2, bound * 2))
#     y_all = np.zeros(y_dims)
#
#     # Filter emitters by intensity threshold, more filters can be chained as necessary
#     data = emitter_positions[emitter_positions['intensity [photon]'] > intensity_threshold]
#
#     # Extract x and y pos in terms of pixels
#     pixel_x, pixel_y = round(data["x [nm]"] / pixel_size), round(data["y [nm]"] / pixel_size)
#     emitters = pd.DataFrame({"x": pixel_x, "y": pixel_y, "frame": data['frame']})
#
#     # Apply z-positions to points based on frame number using calculated stack depths above
#     emitters['z'] = emitters.apply(lambda row: z_stack_depths[int(row['frame'] - 1)], axis=1)
#
#     # Now we iterate through each frame, filter points for proximity, extract PSFs and z-positions
#     for i in range(actual_frames):
#         emitters_in_frame = emitters[data["frame"] == i + 1]
#         if filter_points:
#             emitters_in_frame = image_processing.filter_points(emitters_in_frame,
#                                                                bound=bound)
#         x, y = image_processing.get_emitter_data(image[i], emitters_in_frame, bound=bound,
#                                                  normalise=normalise_images)
#
#         x_all = np.append(x_all, x, axis=0)
#         y_all = np.append(y_all, y)
#
#     # Remove placeholder zeroth elements
#     x_train = np.delete(x_all, 0, axis=0)
#     y_train = np.delete(y_all, 0)
#
#     # Reshape x_train for direct input into CNN
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#
#     return x_train, y_train
#
# def process_STORM_test(imagestack, storm_positions:pd.DataFrame, bound=16,
#                         filter_points=True, pixel_size=106, normalise_images=True):
#
#     data = storm_positions[["frame", "x [nm]", "y [nm]"]]
#
#     pixel_x, pixel_y = round(data["x [nm]"] / pixel_size), round(data["y [nm]"] / pixel_size)
#     pixel_locations = pd.DataFrame({'frame': data['frame'], "x": pixel_x, "y": pixel_y})
#
#
#
#
#
#     return
#
#
#
# def process_blobs(image, bound=16, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.01,
#                   overlap=.5, normalise=True):
#     """
#     Uses sk-image feature.blob.log (lorentzian-on-gaussian) blob detection based on params
#     to detect blobs in an image. Function takes in an image and crops out PSFs of window radius
#     "bound" around center of blob. Params involved enable fine-tuning of blob detection.
#
#     :param image: image array
#     :param bound: window radius around blob center
#     :param min_sigma: the minimum standard deviation for Gaussian kernel. Keep this low to
#         detect smaller blobs
#     :param max_sigma: The maximum standard deviation for Gaussian kernel. Keep this high to
#         detect larger blobs.
#     :param num_sigma: The number of intermediate values of standard deviations to consider
#         between `min_sigma` and `max_sigma`. Optional.
#     :param threshold: The absolute lower bound for scale space maxima. Local maxima smaller
#         than thresh are ignored. Reduce this to detect blobs with less
#         intensities.
#     :param overlap:A value between 0 and 1. If the area of two blobs overlaps by a
#         fraction greater than `threshold`, the smaller blob is eliminated.
#     :param normalise:
#     :return: PSFs of blob centers.
#     """
#
#     blobs = image_processing.detect_blobs(image, min_sigma=min_sigma,
#                                           max_sigma=max_sigma, num_sigma=num_sigma,
#                                           threshold=threshold,
#                                           overlap=overlap)
#
#     blobs_filtered = image_processing.filter_points(blobs, bound=bound)
#
#     blob_PSFs = image_processing.get_emitter_data(image, blobs_filtered, bound=bound,
#                                                   normalise=normalise, with_zpos=False)
#
#     return blob_PSFs
#
#
# def process_blob_zstack(image, z_data, bound=16, y_dims=1,
#                         min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.01,
#                         overlap=.5, normalise=True):
#     """
#     Uses process_blobs, see implementation above. Merely scans through z-stack of beads frame
#     by frame, extracts PSFs of blobs for each frame and tags with corresponding known z-pos.
#     Returns PSFs, zpos as usual.
#
#     :param image:
#     :param z_data:
#     :param bound:
#     :param y_dims:
#     :param min_sigma:
#     :param max_sigma:
#     :param num_sigma:
#     :param threshold:
#     :param overlap:
#     :param normalise:
#     :return:
#     """
#     # Check that the z_data = [start,stop,step] does indeed yield the expected number
#     # of frames in the image
#
#     start, stop, step = z_data
#     expected_frames = abs((stop - start)) / step + 1
#     actual_frames = image.shape[0]
#     assert expected_frames == actual_frames, "Check z-data, does not match frames of the image."
#     z_stack_depths = np.linspace(start, stop, actual_frames)
#
#     print("Processing blob zstack")
#     # Storage arrays
#     x_all = np.zeros((1, bound * 2, bound * 2))
#     y_all = np.zeros(y_dims)
#
#     # Now we iterate through each frame, filter points for proximity, extract PSFs and z-positions
#     for i in range(actual_frames):
#         x = process_blobs(image[i], bound=bound, min_sigma=min_sigma, max_sigma=max_sigma,
#                           num_sigma=num_sigma, threshold=threshold, overlap=overlap,
#                           normalise=normalise)
#         y = np.repeat(z_stack_depths[i], x.shape[0])
#
#         x_all = np.append(x_all, x, axis=0)
#         y_all = np.append(y_all, y)
#
#     # Remove placeholder zeroth elements
#     x_train = np.delete(x_all, 0, axis=0)
#     y_train = np.delete(y_all, 0)
#
#     # Reshape x_train for direct input into CNN
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#
#     return x_train, y_train
#
#
# def split_MATLAB_data(data, mags, zpos, iterations, split_by='mags'):
#     """
#     Returns MATLAB data as a dictionary (key:value pair) where the keys are
#     magnitude of aberrations (mags) and values are the corresponding PSFs of various
#     zpos for that particular simulated aberration. Iterations = no. of iterations per PSF.
#
#     :param data:
#     :param mags:
#     :param zpos:
#     :param iterations:
#     :param split_by:
#     :return:
#     """
#     X, y = data
#     N = X.shape[0]
#
#     # Sanity check, N must match splitting criteria
#     assert N == len(mags) * len(zpos) * iterations, "Chosen inputs do not match input data size."
#     split_data_x = {}
#     split_data_y = {}
#
#     if split_by == 'mags':
#         # Splitting dataset by magnitude of aberrations
#         for i in range(len(mags)):
#             index_offset = i * iterations * len(zpos)
#             split_data_x[f'{mags[i]}'] = X[0 + index_offset:iterations * len(zpos) + index_offset - 1]
#             split_data_y[f'{mags[i]}'] = y[0 + index_offset:iterations * len(zpos) + index_offset - 1]
#
#     else:
#         raise NotImplementedError
#
#     return split_data_x, split_data_y
#
#
# def process_MATLAB_split_data(data, train_keys_size):
#     """
#     Process data split into dictionaries by split_MATLAB_data. Separates data into
#     training and testing based on train_keys_size. Suppose dictionary x has A keys, and
#     train_keys_size is B, then x_train will have B keys while x_test will have the remaining
#     A-B keys.
#
#     :param data:
#     :param train_keys_size:
#     :return:
#     """
#
#     x, y = data
#
#     keys = list(x.keys())
#     random.shuffle(keys)
#
#     train_keys, test_keys = keys[:train_keys_size], keys[train_keys_size:]
#
#     x_ = {key: x[key] for key in train_keys}
#     y_ = {key: y[key] for key in train_keys}
#     x__ = {key: x[key] for key in test_keys}
#     y__ = {key: y[key] for key in test_keys}
#
#     x_train = np.concatenate(list(x_.values()))
#     y_train = np.concatenate(list(y_.values()))
#     x_test = np.concatenate(list(x__.values()))
#     y_test = np.concatenate(list(y__.values()))
#
#     return train_keys, test_keys, x_train, y_train, x_test, y_test
#
#
# def split_test_train(data, split=0.2):
#     X, y = data
#
#     N = X.shape[0]  # No. of samples
#
#     indices = np.random.permutation(N)
#     N_test = int(np.floor((1 - split) * N))
#
#     train_idx, test_idx = indices[:N_test], indices[N_test:]
#
#     X_train, y_train = X[train_idx], y[train_idx]
#     X_test, y_test = X[test_idx], y[test_idx]
#
#     return X_train, y_train, X_test, y_test
