import math
from abc import abstractmethod
from glob import glob
import os
import numpy as np
import pandas as pd
from tifffile import imread, imshow, imwrite
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.data_manager import jonny_data_dir


def cut_image_stack(image, center, width=16, show=False):
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
    cut = image[:, x_min:x_max, y_min:y_max]

    if show:
        imshow(cut)

    return cut


def filter_localisations(coordinates, im_size, bound, pixel_size):
    valid_emitters = []
    for i in range(len(coordinates)):
        c_x, c_y = coordinates[i]
        # Check emitter against edges of images
        if not ((bound <= c_x <= im_size[2] - bound) and (bound <= c_y <= im_size[1] - bound)):
            # print(c_x, c_y, im_size, 'too close to edge')
            continue

        reject = False
        for i2 in range(i + 1, len(coordinates)):
            c2_x, c2_y = coordinates[i2]
            x_diff = c2_x - c_x
            y_diff = c2_y - c_y
            dist = np.linalg.norm([x_diff, y_diff])
            dist_pixels = dist / pixel_size
            if dist_pixels <= bound*math.sqrt(2):
                reject = True
                # print(c_x, c2_x, c_y, c2_y, x_diff, y_diff, 'too close to point')
                break

        if not reject:
            valid_emitters.append(coordinates[i])
    return valid_emitters


class DataSource:
    voxel_sizes = None

    def __init__(self, directory):
        self.dir = directory

    @abstractmethod
    def get_all_images(self):
        raise NotImplemented()

    @abstractmethod
    def get_image_csv_pairs(self):
        raise NotImplemented()


class JonnyDataSource(DataSource):
    voxel_size = (0.1, 0.085, 0.085)

    def localisation2pixel(self, df, im_shape):
        x_pixel = (im_shape[2] / 2) + (df['x0 (um)'] / self.voxel_size[2])
        y_pixel = im_shape[1] - ((im_shape[1] / 2) + (df['y0 (um)'] / self.voxel_size[1]))
        return x_pixel, y_pixel

    def get_all_images(self):
        return glob(os.path.join(self.dir, '*', 'MMStack_Default.ome.tif'))

    def get_all_image_stacks(self):
        return glob(os.path.join(self.dir, '*', 'stack', 'MMStack_Default.ome.tif'))

    def get_image_csv_pairs(self):
        images = self.get_all_images()
        img_csv = lambda img: os.path.join(os.path.dirname(img), 'stack', 'MMStack_Default.csv')
        image_csvs = [(img, img_csv(img)) for img in images]
        return image_csvs

    def get_img_stack_csv_pairs(self):
        images = self.get_all_image_stacks()
        img_csv = lambda img: os.path.join(os.path.dirname(img), 'MMStack_Default.csv')
        image_csvs = [(img, img_csv(img)) for img in images]
        return image_csvs

    def get_all_emitter_stacks(self, bound=16, pixel_size=85):
        imstack_csvs = self.get_img_stack_csv_pairs()

        emitters = []
        coordinates = []
        pixel_xs = []
        pixel_ys = []
        for img, csv in tqdm(imstack_csvs):
            print(img)
            image = imread(img)
            truth = pd.read_csv(csv, skiprows=28)
            data = truth[["x0 (um)", "y0 (um)", "z0 (um)"]]
            # Convert to nm for thresholding
            for a, b in (("x0 (um)", "x0 (nm)"), ("y0 (um)", "y0 (nm)"), ("z0 (um)", "z0 (nm)")):
                data[b] = data[a] * 1000

            pixel_x, pixel_y = self.localisation2pixel(data, image.shape)

            # pixel_x, pixel_y = round(data["x0 (nm)"] / pixel_size), round(data["y0 (nm)"] / pixel_size)

            emitter_coords = list(zip(pixel_x, pixel_y))
            emitter_coords = filter_localisations(emitter_coords, image.shape, bound, pixel_size)
            # pixel_x, pixel_y = list(zip(*emitter_coords))
            pixel_xs.extend(pixel_x)
            pixel_ys.extend(pixel_y)
            for coord in tqdm(emitter_coords):
                psf = cut_image_stack(image, coord, width=bound)

                emitters.append(psf)
                coordinates.append(coord)

                # imshow(psf[20])
                # plt.show()

        pd.DataFrame.from_dict({'x': pixel_xs, 'y': pixel_ys}).to_csv('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/coords.csv')
        # quit()
        emitters = np.stack(emitters)
        return emitters, coordinates
        # pixel_locations = pd.DataFrame({"x": pixel_x, "y": pixel_y, "z": data["z0 (nm)"]})
        # filtered_emitters = image_processing.filter_points(pixel_locations, bound=bound)
        # x, y = image_processing.get_emitter_data(image, filtered_emitters, bound=bound


if __name__ == '__main__':
    ds = JonnyDataSource(jonny_data_dir)
    psfs, _ = ds.get_all_emitter_stacks(bound=20, pixel_size=85)
    for i, psf in enumerate(psfs):
        imshow(psf[int(psf.shape[0]/2)])
        plt.show()
        imwrite(f'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/{i}.tif', psf, compress=6)

