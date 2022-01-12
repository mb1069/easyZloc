import math
from operator import sub
import os
from pyotf.utils import prep_data_for_PR, remove_bg

from scipy.spatial.distance import cdist
from tifffile import imread, imshow, imwrite
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from final_project.smlm_3d.data.estimate_offset import estimate_offset
from final_project.smlm_3d.data.visualise import scatter_3d, scatter_yz, show_psf_axial
from final_project.smlm_3d.debug_tools.est_calibration_stack_error import fit_plane
from final_project.smlm_3d.util import get_base_data_path, split_for_training, dwt_dataset

from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.experiments.noise.noise_psf import generate_noisy_psf

from final_project.smlm_3d.experiments.noise.low_pass_filter import apply_low_pass_img_stack

DEBUG = False


class GenericDataSet:
    filter_emitters_edges = True
    voxel_sizes = None
    bound = 16

    def __init__(self, config, transform_data, normalize_psf=True):
        self.config = config
        self.normalize_psf = normalize_psf
        self.impath = os.path.join(config['bpath'], config['img'])
        self.img = imread(self.impath)
        self.transform_data = transform_data

        csv_path = os.path.join(config['bpath'], config['csv'])

        print(f'Loading {self.impath}')
        print(f'Loading {csv_path}')

        self.csv_data = pd.read_csv(csv_path)
        if config['reverse_stack']:
            print('\n\nreversing stack!!\n\n')
            self.img = self.img[::-1]
            self.csv_data['frame'] = self.img.shape[0] - self.csv_data['frame']
        self.voxel_sizes = config['voxel_sizes']

    # @staticmethod
    # def normalise_input(psfs):  
    #     return psfs / psfs.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
        
    @staticmethod
    def transform_input(psfs):
        return dwt_dataset(psfs)

    def remap_emitter_coords(self, image, df):
        # Change origin from centre of image to top left corner
        # x_center = (image.shape[-1] * self.voxel_sizes[-1] / 2)
        # y_center = (image.shape[-2] * self.voxel_sizes[-2] / 2)
        # df['x [nm]'] += x_center
        # df['y [nm]'] = (-df['y [nm]']) + y_center
        return df

    @staticmethod
    def localisation2pixel(df, voxel_size):
        x_pixel = (df['x [nm]'] / voxel_size[-1])
        y_pixel = (df['y [nm]'] / voxel_size[-2])
        return x_pixel, y_pixel
    
    @staticmethod
    def normalise_image(psf):
        if len(psf.shape) == 3:
            min_z = psf.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
            max_z = psf.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
            psf = (psf - min_z) / (max_z - min_z)
        else:
            psf = (psf - psf.min()) / (psf.max() - psf.min())
        return psf

    def cut_image_stack(self, image, center, width=16, show=False):
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
        if image.ndim == 3:
            cut = image[:, x_min:x_max, y_min:y_max]
        else:
            cut = image[x_min:x_max, y_min:y_max]

        if show:
            imshow(cut)
            plt.show()
        return cut

    @staticmethod
    def filter_localisations(df, im_size, bound, voxel_sizes, edges=True, proximity=True):
        x_col = 'x [nm]'
        y_col = 'y [nm]'

        def check_borders(emitter):
            bound_mult = bound * 1.5
            x_nm, y_nm = emitter[x_col], emitter[y_col]
            x_pixel = x_nm / voxel_sizes[-1]
            y_pixel = y_nm / voxel_sizes[-2]

            x_valid = (0 + bound_mult <= x_pixel <= im_size[-1] - bound_mult)
            y_valid = (0 + bound_mult <= y_pixel <= im_size[-2] - bound_mult)

            return x_valid and y_valid

        print(f'{df.shape[0]} emitters before filtering')

        if proximity:
            from sklearn.metrics.pairwise import euclidean_distances
            min_seperation = 1.5 * bound * voxel_sizes[-1]
            xy_coords = df[[x_col, y_col]].to_numpy()
            distances = euclidean_distances(xy_coords, xy_coords)
            distances[distances==0] = np.inf
            min_distances = distances.min(axis=1)
            emitter_coords = np.where(min_distances>min_seperation)[0]
            df = df.iloc[emitter_coords, :]
            
            # num_emitters = df.shape[0]
            # filters = []
            # for i in range(0, num_emitters - 1):
            #     emitter = df.iloc[i, :]
            #     emitter_coords = emitter[[x_col, y_col]].to_numpy()[np.newaxis]

            #     other_emitters = df.iloc[i + 1:, :]
            #     other_coords = other_emitters[[x_col, y_col]].to_numpy()

            #     min_dist = cdist(emitter_coords, other_coords).min()
            #     filters.append(min_dist >= min_seperation)
            print(f'{df.shape[0]} emitters after proximity filtering.')
        if edges:
            df = df[df.apply(check_borders, axis=1)]
            print(f'{df.shape[0]} emitters after borders')
        return df

    def estimate_ground_truth(self):
        pixel_coords, filtered_df = self.fetch_emitters_coords(self.img, self.csv_data)
        self.csv_data = filtered_df
        xyz_coords = []
        
        psfs = []
        for i, pixel_coord in enumerate(tqdm(pixel_coords)):
            psf = self.cut_image_stack(self.img, pixel_coord, width=self.bound)
            try:
                z_pos = -estimate_offset(psf, self.voxel_sizes)[0]
                psf = self.normalise_image(psf)

                xy_coord = filtered_df.iloc[i][['x [nm]', 'y [nm]']].to_numpy()
                xyz_coord = np.array([*xy_coord, z_pos])[np.newaxis, :]
                xyz_coords.append(xyz_coord)
                psfs.append(psf)
                if DEBUG and i > 10:
                    break
            except RuntimeError as e:
                print(e)
        
        xyz_coords = np.concatenate(xyz_coords)
        xyz_coords[:,2] = xyz_coords[:,2] - xyz_coords[:,2].min()
        return [psfs, xyz_coords]

    def fetch_emitters_modelled_z(self):
        psfs, xyz_coords = self.estimate_ground_truth()
        psfs = np.stack(psfs)
        dists_to_plane, _, subset_idx = fit_plane(xyz_coords, subset=True)
        print(f'{len(subset_idx)} emitters after fitting plane')
        self.csv_data = self.csv_data.iloc[subset_idx]

        psfs = psfs[subset_idx]

        min_max_z = psfs[0].shape[0]/2
        z_placeholder = np.linspace(-min_max_z, min_max_z, psfs[0].shape[0]) * self.voxel_sizes[0]

        psfs, z_pos = zip(*[self.trim_stack(psf, z_placeholder.copy() + offset) for psf, offset in zip(psfs, dists_to_plane)])
        z_pos = np.stack(z_pos)
        xy_coords = xyz_coords[subset_idx][:, [0,1]]

        xy_coords = xy_coords[:, np.newaxis, :]
        z_pos = z_pos[:, :, np.newaxis]
        xy_coords_rep = np.repeat(xy_coords, z_pos.shape[1], axis=1)

        zxy_coords = np.concatenate((z_pos, xy_coords_rep), axis=2)
        return [psfs, zxy_coords]


    def trim_stack(self, psf, z_pos):
        valid_ids = np.where(abs(z_pos) < self.z_range)
        psf = psf[valid_ids]
        z_pos = z_pos[valid_ids]
        psf = psf[:199]
        z_pos = z_pos[:199]
        return [psf, z_pos]

    def fetch_emitters(self, img=None, df=None):
        if img is None:
            img = self.img
        if df is None:
            df = self.csv_data
        pixel_coords, filtered_df = self.fetch_emitters_coords(img, df)
        self.csv_data = filtered_df
        psfs = []
        zxy_coords = []

        total_emitters = len(pixel_coords)
        kept_emitters = 0
        for i, pixel_coord in enumerate(tqdm(pixel_coords)):
            psf = self.cut_image_stack(img, pixel_coord, width=self.bound)
    
            try:
                z_pos = estimate_offset(psf, self.voxel_sizes)
                psf = self.normalise_image(psf)
                psf, z_pos = self.trim_stack(psf, z_pos)
                xy_coord = np.tile(filtered_df.iloc[i][['x [nm]', 'y [nm]']].to_numpy(), reps=(len(z_pos), 1))
                zxy_coord = np.hstack((z_pos[:, np.newaxis], xy_coord))

                # psf = apply_low_pass_img_stack(psf)
                psfs.append(psf)
                zxy_coords.append(zxy_coord)
                if DEBUG and i > 20:
                    break
                kept_emitters +=1
            except RuntimeError as e:
                print(e)
        print(f'Retained {round((kept_emitters/total_emitters)*100, 2)}% of emitters')

        return psfs, zxy_coords

    def fetch_emitters_coords(self, img, df):
        if 'Fit valid' in df:
            df = df[df['Fit valid'] == 1]

        # if self.filter_low_intensity_emitters and 'intensity [photon]' in df:
        #     df = df[df['intensity [photon]'] > 2500]

        if 'x0 (um)' in df:
            df = df[["x0 (um)", "y0 (um)", "z0 (um)"]]
            # Convert to nm for thresholding
            for a, b in (("x0 (um)", "x [nm]"), ("y0 (um)", "y [nm]"), ("z0 (um)", "z [nm]")):
                df.loc[:, b] = df.loc[:, a] * 1000
                del df[a]

        df = self.remap_emitter_coords(img, df)

        df = self.filter_localisations(df, img.shape, self.bound, self.voxel_sizes,
                                       edges=self.filter_emitters_edges, proximity=self.filter_emitters_proximity)

        pixel_x, pixel_y = self.localisation2pixel(df, voxel_size=self.voxel_sizes)
        pixel_x = np.round(pixel_x).astype(int)
        pixel_y = np.round(pixel_y).astype(int)
        emitter_coords = list(zip(pixel_x, pixel_y))
        return emitter_coords, df

    def prepare_debug(self):
        self.pixel_coords, self.filtered_df = self.fetch_emitters_coords(self.img, self.csv_data)
        self.total_emitters = self.filtered_df.shape[0]

    def debug_emitter(self, i, z_range):
        psf = self.cut_image_stack(self.img, self.pixel_coords[i], width=self.bound)

        coords = self.filtered_df[['x [nm]', 'y [nm]']].iloc[i].to_numpy()

        z = estimate_offset(psf, self.voxel_sizes)
        if self.normalize_psf:
            psf = self.normalise_image(psf)
        if self.transform_data:
            input_data = self.transform_input(psf)
            dwt = np.hstack((input_data, np.tile(coords, (input_data.shape[0], 1))))
        else:
            dwt = None

        idx = np.where(abs(z) < z_range)
        psf = psf[idx]
        if self.transform_data:
            dwt = input_data[idx]
        z = z[idx]

        return psf, dwt, coords, z, self.filtered_df.iloc[i]
        
class TrainingDataSet(GenericDataSet):
    def __init__(self, config, z_range, filter_emitters_proximity=True, filter_low_intensity_emitters=True, split_data=True, lazy=False, transform_data=True, add_noise=True, normalize_psf=True):
        super().__init__(config, transform_data, normalize_psf)
        self.filter_emitters_proximity = filter_emitters_proximity
        self.filter_low_intensity_emitters = filter_low_intensity_emitters

        self.z_range = z_range
        self.split_data = split_data
        self.add_noise = add_noise
        if not lazy:
            self.prepare_data()

    def norm_xy_coords(self, xy_coords):
        imshape_xy = list(reversed(self.img.shape[1:]))
        for axis in [0,1]:
            axis_dim = imshape_xy[axis] * self.voxel_sizes[axis+1]
            xy_coords[:, axis] = 2*(xy_coords[:, axis] / axis_dim) - 1
        return xy_coords

    def prepare_data(self):
        psfs, zxy_pos = self.fetch_emitters_modelled_z()
        # psfs, zxy_pos = self.fetch_emitters(self.img, self.csv_data)
        self.all_psfs = psfs
        self.all_coords = zxy_pos

        split_type = 'all' 
        if self.split_data:
            if split_type == 'stacks':
                self.data = split_for_training(psfs, zxy_pos)
                for k in self.data:
                    self.data[k][0] = np.concatenate(self.data[k][0])
                    self.data[k][1] = np.concatenate(self.data[k][1])
            else:
                psfs = np.concatenate(psfs)
                zxy_pos = np.concatenate(zxy_pos)
                self.data = split_for_training(psfs, zxy_pos)
        else:
            self.data = {'all': [np.concatenate(psfs), np.concatenate(zxy_pos)]}
            
        if self.add_noise:
            print('Adding noise')
            psfs = []
            coords = []
            for psf, coord in zip(*self.data['train']):
                psfs.append(psf)
                coords.append(coord)
                for _ in range(10):
                    noisy_psf = generate_noisy_psf(psf)
                    psfs.append(noisy_psf)
                    coords.append(coord)
            psfs = np.stack(psfs)
            coords = np.stack(coords)
            self.data['train'] = [psfs, coords]

        img_center = [[self.img.shape[i]*self.config['voxel_sizes'][i] / 2 for i in [1,2]]]

        for k in self.data:
            input_data, zxy_pos = self.data[k]
            if self.transform_data:
                input_data = self.transform_input(input_data)
            else:
                input_data = input_data[:, :, :, np.newaxis]
            if len(zxy_pos.shape) == 1:
                target_data = zxy_pos
            else:
                target_data = zxy_pos[:, 0]
            xy_coords = zxy_pos[:, (1,2)]

            # self.data[k] = [[input_data, self.norm_xy_coords(xy_coords)], target_data]

            def cart2pol(coords):
                x, y = coords
                rho = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return(rho, phi)

            re_mapped_coords = xy_coords - img_center
            polar_coords = np.apply_along_axis(cart2pol, 1, re_mapped_coords)
            print('\n')
            # norm angles to [0, 1]
            polar_coords[:, 1] = (polar_coords[:, 1] + np.pi) / (2*np.pi)

            # norm distances to [0, 1]
            img_dims = [self.img.shape[i]*self.config['voxel_sizes'][i] for i in [1,2]]
            max_distance = np.hypot(*img_dims) / 2

            polar_coords[:, 0] = polar_coords[:, 0] / max_distance

            # dists = cdist(img_center, xy_coords).T
            # dists = (dists - dists.min()) / (dists.max() - dists.min())

            self.data[k] = [[input_data, polar_coords], target_data]



        # psfs = np.concatenate(psfs)
        # input_data = self.transform_input(psfs)
        # input_data = np.hstack((input_data, xyz_pos[:, (1,2)]))
        # target_data = xyz_pos[:, 0]

        # if not self.split_data:
        #     self.data = (input_data, target_data)
        # else:
        #     # Split slices into distinct train/val/test
        #     # self.data = split_for_training(input_data, target_data)


        #     # Split emitter stacks into distinct train/val/test
        #     self.data = split_for_training(psfs, xyz_pos)

        #     train_psfs, train_coords = self.data['train']
        #     psfs = []
        #     coords = []
        #     for psf, coord in zip(train_psfs, train_coords):
        #         for _ in range(2):
        #             noise_psf = generate_noisy_psf(psf.copy())
        #             psfs.append(noise_psf)
        #             coords.append(coord)


        #     for k in ['test', 'val', 'train']:
        #         print(k, self.data[k][0].shape, self.data[k][1].shape)
        #         self.data[k][0] = np.stack(self.data[k][0])
        #         print(k, self.data[k][0].shape, self.data[k][1].shape)
        #         self.data[k][0] = self.transform_input(self.data[k][0])
        #         self.data[k][1] = np.concatenate(self.data[k][1])
            
        #         # Add XY coordinates into input data
        #         self.data[k][0] = np.hstack((self.data[k][0], self.data[k][1][:, (1, 2)]))
        #         self.data[k][1] = self.data[k][1][:, 0]


class ExperimentalDataSet(GenericDataSet):
    def __init__(self, config, filter_localisations=True, lazy=False, transform_data=True):
        super().__init__(config, transform_data)
        if not filter_localisations:
            self.filter_emitters_proximity = False
            self.filter_low_intensity_emitters = False
    
        if 'z_step' not in config:
            self.z_step = config['voxel_sizes'][0]
        else:
            self.z_step = 0

        if len(self.img.shape) == 2:
            self.img = self.img[np.newaxis]
        
        if not lazy:
            self.prepare_data()

    def prepare_data(self):
        psfs, coords = self.fetch_all_emitters()

        psfs = psfs.astype(np.float32)
        # print(psfs.shape)
        # imwrite('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/tmp/out.tif', psfs, compress=6)
        # quit()

        if self.transform_data:
            input_data = self.transform_input(psfs)
            input_data = np.hstack((input_data, coords[:, (0, 1)]))
        else:
            input_data = psfs[:, :, :, np.newaxis]

        self.data = (input_data, coords)


    def fetch_emitters(self, img, df, z_coord=0):
        pixel_coords, filtered_df = self.fetch_emitters_coords(img, df)
        xyz_coords = []

        psfs = np.zeros((len(pixel_coords), self.bound * 2, self.bound * 2))
        for i, pixel_coord in enumerate(tqdm(pixel_coords)):
            psf = self.cut_image_stack(img, pixel_coord, width=self.bound)
            psfs[i] = self.normalise_image(psf)
            xy_coord = filtered_df.iloc[i][['x [nm]', 'y [nm]']].to_numpy()
            xyz_coords.append(list(xy_coord) + [z_coord])
        return psfs, xyz_coords

    def fetch_all_emitters(self):
        all_psfs = []
        all_coords = []
        for slice_id in range(self.img.shape[0]):
            img = self.img[slice_id]
            df = self.csv_data[self.csv_data['frame'] == (slice_id + 1)]
            if len(df) == 0:
                continue

            psfs, coords = self.fetch_emitters(img, df, slice_id * self.z_step)
            all_psfs.append(psfs)
            all_coords.extend(coords)

        all_coords = np.array(all_coords)
        all_psfs = np.concatenate(all_psfs)

        return all_psfs, all_coords


    def predict_dataset(self, model, frame):
        input_data, coords = self.data
        input_data = input_data[frame]
        print(input_data.shape)
        coord_diff = model.predict(input_data).squeeze()
        print(coord_diff.shape)
        # plt.hist(coord_diff)
        # plt.show()
        coords[:, 2] -= coord_diff
        print(coord_diff.min(), coord_diff.mean(), coord_diff.max())
        return coords

if __name__=='__main__':
    from final_project.smlm_3d.config.datasets import dataset_configs

    z_range = 1000

    dataset = 'paired_bead_stacks'

    # train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=True)
    exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False, split_data=False)

    print(np.stack(exp_dataset.all_psfs).shape)
    print(exp_dataset.csv_data.shape)