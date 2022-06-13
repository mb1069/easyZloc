import numpy as np
import pandas as pd
from skimage import io, feature
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist

from src.wavelets.wavelet_data.util import cut_image_stack
from src.data.estimate_offset import estimate_offset


def normalise_image(image_data_array, bit_16=True):  # Normalize images to 0 - 1

    if bit_16:
        rescale = 65535
    else:
        rescale = 255

    return np.divide(image_data_array, rescale)


def convert_to_8bit(image):  # Convert from 16-bit to 8-bit
    converted_image = image / 256
    converted_image = converted_image.astype('uint8')
    return converted_image


def cut_image(image, center, width=16, show=False):
    """

    :param image: numpy array representing image
    :param center: emitter position from truth or STORM
    :param width: window around emitter
    :param show: io.imshow returns cut-out of PSF
    :return: cut out of PSF as a numpy array
    """

    # NOTE: for some reason numpy images seem to have x and y swapped in the logical
    # order as would be for a coordinate point (x,y). I.e. point x,y in the image
    # is actually image[y,x]
    x_min, x_max = int(center[1] - width), int(center[1] + width)
    y_min, y_max = int(center[0] - width), int(center[0] + width)
    cut = image[x_min:x_max, y_min:y_max]

    if show:
        io.imshow(cut)

    return cut


def filter_points(points: pd.DataFrame, bound=16):
    """
    Filters emitters using list of emitter locations and a boundary

    :param points: pixel locations of emitters, must be a dataframe with columns "x" and "y"
    :param bound: Exclusion boundary around any emitter
    :return: Viable emitter positions
    """
    rejected_points = set()
    # Reset point indices in case we have already filtered points out before this step
    points = points.reset_index(drop=True)

    points = points.sample(n=10000)
    points = points.reset_index(drop=True)

    df = points[['x', 'y']].to_numpy()
    print(df.shape)
    m = cdist(df, df)
    print(m.shape)
    np.fill_diagonal(m, np.inf)
    print(m.shape)
    m = np.min(m, axis=1).squeeze()
    points_to_keep = np.argwhere(m > bound).squeeze()

    points = points.loc[points_to_keep]
    print(points.shape)
    return points
    for i in trange(len(x) - 1):
        for j in range(i + 1, len(x)):
            x_diff = x[j] - x[i]
            y_diff = y[j] - y[i]
            dist = np.linalg.norm([x_diff, y_diff])
            if dist <= bound:
                rejected_points.add(i)
                rejected_points.add(j)

    rejected_points = list(rejected_points)

    return points.drop(rejected_points)


def get_emitter_data(image, points, bound=16, normalise=True, with_zpos=True):
    """
    Cuts out the (bound, bound) shape around filtered emitters given by points,
    normalise will normalise the final image to 0-1.

    with_zpos indicates whether z-positions accompany points in the points list. Set True for
    known/simulated data, set to False if working with test data for prediction (unknown zpos).

    :param with_zpos: Boolean indicating if z-position is given in points list
    :param image: Full image of all emitters
    :param points: Filtered list of emitters
    :param bound: Exclusion zone
    :param normalise: Normalise image 0-1 BEFORE processing
    :return: Array of (bound, bound) PSFs of each emitter and corresponding z-position
    """
    image_data = np.zeros((1, bound * 2, bound * 2))  # Store cut images
    z_data = np.zeros(1)  # Store corresponding z-position if needed

    if normalise:
        image = normalise_image(image)

    # Check if emitters are near edge of image so PSF cannot be cropped properly
    for i in tqdm(points.index):
        if points["x"][i] - bound < 0 or points["x"][i] + bound > image.shape[0]:
            continue
        if points["y"][i] - bound < 0 or points["y"][i] + bound > image.shape[1]:
            continue

        # Cut out image with emitter i at center.
        if len(image.shape) == 3:
            psf = cut_image_stack(image, (points["x"][i], points["y"][i]),
                        width=bound)
        else:
            psf = cut_image(image,
                            (points["x"][i], points["y"][i]),
                            width=bound)

        # Append to respective arrays
        print(psf.shape)
        image_data = np.append(image_data, [psf], axis=0)

        if with_zpos:
            if with_zpos == 'corrected':
                z_position = estimate_offset(psf)
            else:
                z_position = points["z"][i]  # Corresponding z-pos if available
            z_data = np.append(z_data, z_position)

    # Delete zeroth initiated elements
    image_data = np.delete(image_data, 0, axis=0)
    z_data = np.delete(z_data, 0)

    if with_zpos:
        # Returns both PSFs and corresponding z-pos
        return image_data, z_data
    else:
        # Returns PSFs only
        return image_data, None


def detect_blobs(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.01,
                 overlap=.5):
    blob_positions = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                                      num_sigma=num_sigma, threshold=threshold,
                                      overlap=overlap)

    # Note the swap in columns, this is for use with cut_image where x & y are swapped for images
    positions = pd.DataFrame({"x": blob_positions[:, 1], "y": blob_positions[:, 0]})

    return positions
