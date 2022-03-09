import math
from operator import sub
import os
from pickle import FALSE
from cv2 import norm
from scipy.sparse import data

from scipy.spatial.distance import cdist
from tifffile import imread, imshow, imwrite
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from functools import partial
from scipy.optimize import least_squares
from final_project.smlm_3d.data.estimate_offset import estimate_offset, get_peak_sharpness, norm_zero_one
from final_project.smlm_3d.data.visualise import scatter_3d, scatter_yz, show_psf_axial, plot_with_sphere
from final_project.smlm_3d.debug_tools.est_calibration_stack_error import fit_plane
from final_project.smlm_3d.util import get_base_data_path, split_for_training, dwt_dataset

from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.experiments.noise.noise_psf import generate_noisy_psf

from final_project.smlm_3d.experiments.noise.low_pass_filter import apply_low_pass_img_stack

DEBUG = False

if DEBUG:
    print('DEBUG enabled in datasets.py')

from skimage.filters import butterworth

def remove_bg(img):
    img = img / img.max()
    bg_level = np.percentile(img.ravel(), 5)
    mult = 1.2
    img = img - (bg_level * mult)
    img[img<0] = 0
    img = norm_zero_one(img)
    return img

def butter_psf(psf):
    psf = remove_bg(psf)
    return np.stack([butterworth(img, 0.1, False) for img in psf])

class GenericDataSet:
    filter_emitters_edges = True
    voxel_sizes = None
    bound = 16

    def __init__(self, config, transform_data, normalize_psf=True):
        self.config = config
        self.normalize_psf = normalize_psf
        self.impath = os.path.join(config['bpath'], config['img'])
        print('Reading img...')
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
            min_seperation = np.hypot(bound, bound) * voxel_sizes[-1]
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
        self.kept_idx = []
        for i, pixel_coord in enumerate(pixel_coords):
            psf = self.cut_image_stack(self.img, pixel_coord, width=self.bound)
            # psf = butter_psf(psf)
            try:
                z_pos = estimate_offset(psf, self.voxel_sizes, self.disable_emitter_peak_boundary)[0]
                if self.normalize_psf:
                    psf = self.normalise_image(psf)

                xy_coord = filtered_df.iloc[i][['x [nm]', 'y [nm]']].to_numpy()
                xyz_coord = np.array([*xy_coord, z_pos])[np.newaxis, :]
                xyz_coords.append(xyz_coord)
                psfs.append(psf)
                self.kept_idx.append(i)
                if DEBUG:
                    break
            except RuntimeError as e:
                print(e)
                pass
        
        xyz_coords = np.concatenate(xyz_coords)
        xyz_coords[:,2] = xyz_coords[:,2] - xyz_coords[:,2].min()
        return [psfs, xyz_coords]

    def trim_stack(self, psf, z_pos):
        
        valid_ids = np.where(abs(z_pos) <= self.z_range)[0]
        # if len(valid_ids) == 0:
        #     show_psf_axial(psf)
        psf = psf[valid_ids]
        z_pos = z_pos[valid_ids]
        return [psf, z_pos]

    def fetch_emitters_coords(self, img, df):
        if 'Fit valid' in df:
            df = df[df['Fit valid'] == 1]

        if 'x0 (um)' in df:
            df = df[["x0 (um)", "y0 (um)", "z0 (um)"]]
            # Convert to nm for thresholding
            for a, b in (("x0 (um)", "x [nm]"), ("y0 (um)", "y [nm]"), ("z0 (um)", "z [nm]")):
                df.loc[:, b] = df.loc[:, a] * 1000
                del df[a]

        df = self.remap_emitter_coords(img, df)
        df = self.filter_localisations(df, img.shape, self.bound, self.voxel_sizes,
                                       edges=self.filter_emitters_edges, proximity=self.filter_emitters_proximity)
        self.csv_data = df
        pixel_x, pixel_y = self.localisation2pixel(df, voxel_size=self.voxel_sizes)
        pixel_x = np.round(pixel_x).astype(int)
        pixel_y = np.round(pixel_y).astype(int)
        emitter_coords = list(zip(pixel_x, pixel_y))
        return emitter_coords, df

    def prepare_debug(self):
        self.pixel_coords, self.filtered_df = self.fetch_emitters_coords(self.img, self.csv_data)
        self.total_emitters = self.filtered_df.shape[0]

    def debug_emitter(self, i, z_range, disable_boundary_check=False, normalize=False):
        psf = self.cut_image_stack(self.img, self.pixel_coords[i], width=self.bound)

        coords = self.filtered_df[['x [nm]', 'y [nm]']].iloc[i].to_numpy()

        offset = get_peak_sharpness(psf, 0.4) * self.voxel_sizes[0]
        z = np.linspace(0, psf.shape[0]-1, num=psf.shape[0]) * self.voxel_sizes[0] - offset
        if normalize:
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

    def convert_xy_coords_to_polar(self, xy_coords):
        def cart2pol(xy_coords):
            x, y = xy_coords
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return(rho, phi)
        img_center = [[self.img.shape[i]*self.config['voxel_sizes'][i] / 2 for i in [1,2]]]

        re_mapped_coords = xy_coords - img_center
        polar_coords = np.apply_along_axis(cart2pol, 1, re_mapped_coords)
        # norm angles to [0, 1]
        polar_coords[:, 1] = (polar_coords[:, 1] + np.pi) / (2*np.pi)

        # norm distances to [0, 1]
        img_dims = [self.img.shape[i]*self.config['voxel_sizes'][i] for i in [1,2]]
        max_distance = np.hypot(*img_dims) / 2

        polar_coords[:, 0] = polar_coords[:, 0] / max_distance

        return polar_coords

class TrainingDataSet(GenericDataSet):
    def __init__(self, config, z_range, filter_emitters_proximity=True, split_data=True, lazy=False, transform_data=False, add_noise=True, normalize_psf=True, disable_emitter_peak_boundary=False, fit_plane_z=True):
        super().__init__(config, transform_data, normalize_psf)
        self.filter_emitters_proximity = filter_emitters_proximity
        self.disable_emitter_peak_boundary = disable_emitter_peak_boundary
        self.fit_plane_z = fit_plane_z

        self.z_range = z_range
        self.split_data = split_data
        self.add_noise = add_noise
        if not lazy:
            self.prepare_data()

    def fit_plane(self, coords):
        import seaborn as sns
        from skspatial.objects import Plane
        from skspatial.objects import Points
        from skspatial.plotting import plot_3d
        plt.rcParams['figure.figsize'] = [30, 10] 
        points = Points(coords)
        plane = Plane.best_fit(points)
        plot_3d(
            points.plotter(c='k', s=75, alpha=0.2, depthshade=False),
            plane.plotter(alpha=0.8, lims_x=(-50000, 50000), lims_y=(-50000, 50000)),
        )
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.show()
        dists = np.array([plane.distance_point_signed(p) for p in points])
        sns.scatterplot(coords[:, 0], coords[:, 2], hue=abs(dists))
        plt.show()

        sns.scatterplot(coords[:, 1], coords[:, 2], hue=abs(dists))
        plt.show()
        
        plt.hist(dists, bins=20)
        plt.show()
        return dists

    def fetch_emitters_modelled_z(self):
        print('using image sharpness')
        pixel_coords, filtered_df = self.fetch_emitters_coords(self.img, self.csv_data)
        self.csv_data = filtered_df

        psfs = []
        z_coords_frame = []
        for i in range(len(self.csv_data)):
            psf = self.cut_image_stack(self.img, pixel_coords[i], width=self.bound)
            z_coords_frame.append(get_peak_sharpness(psf, 0.4))
            psf = self.normalise_image(psf)
            psfs.append(psf)


        z_coords_arr = np.array(z_coords_frame) * self.voxel_sizes[0]
        coords = np.concatenate((self.csv_data[['x [nm]', 'y [nm]']].to_numpy(), z_coords_arr[:, np.newaxis]), axis=1)

        polar_coords = self.convert_xy_coords_to_polar(coords[:, [0, 1]])

        if self.fit_plane_z:
            dists_to_plane = self.fit_plane(coords)
            z_coords_arr -= dists_to_plane
        plt.rcParams['figure.figsize'] = [3, 5]
        all_coords = []
        for i in range(len(self.csv_data)):
            psf = psfs[i].squeeze()
            offset = z_coords_arr[i].squeeze()
            z = np.linspace(0, psf.shape[0]-1, num=psf.shape[0]) * self.voxel_sizes[0] - offset
            
            if DEBUG:
                idx = np.argwhere(abs(z) < self.voxel_sizes[0])
                for i2 in idx:
                    plt.imshow(psf[i2].squeeze())
                    plt.title(z[i2])
                    plt.show()
            psf, z = self.trim_stack(psf, z)
            polar_coords = np.tile(coords[i], (psf.shape[0], 1))

            psfs[i] = psf
            all_coords.append(np.hstack((z[:, np.newaxis], polar_coords)))
            
        return psfs, all_coords

        
    # def fetch_emitters_modelled_z(self):
    #     print('using peak illumination')
    #     psfs, xyz_coords = self.estimate_ground_truth()
    #     psfs = np.stack(psfs)
    #     dists_to_plane, _, subset_idx = fit_plane(xyz_coords)
    #     if subset_idx is not None:
    #         print(f'{len(subset_idx)} emitters after fitting plane')
    #         self.csv_data = self.csv_data.iloc[subset_idx]
    #         psfs = psfs[subset_idx]
    #         xyz_coords = xyz_coords[subset_idx]

    #     min_max_z = psfs[0].shape[0]/2
    #     z_placeholder = np.linspace(-min_max_z, min_max_z, psfs[0].shape[0]) * self.voxel_sizes[0]
    #     psfs, z_pos = zip(*[self.trim_stack(psf, z_placeholder.copy() + offset) for psf, offset in zip(psfs, dists_to_plane)])
    #     psfs, z_pos = self.trim_equal_length(psfs, z_pos)
    #     z_pos = np.stack(z_pos)
    #     xy_coords = xyz_coords[:, [0,1]]

    #     xy_coords = xy_coords[:, np.newaxis, :]
    #     z_pos = z_pos[:, :, np.newaxis]
    #     xy_coords_rep = np.repeat(xy_coords, z_pos.shape[1], axis=1)

    #     zxy_coords = np.concatenate((z_pos, xy_coords_rep), axis=2)
    #     return [psfs, zxy_coords]

    def norm_xy_coords(self, xy_coords):
        imshape_xy = list(reversed(self.img.shape[1:]))
        for axis in [0,1]:
            axis_dim = imshape_xy[axis] * self.voxel_sizes[axis+1]
            xy_coords[:, axis] = 2*(xy_coords[:, axis] / axis_dim) - 1
        return xy_coords


    @staticmethod
    def trim_equal_length(psfs, z_pos):
        min_depth = min([len(zs) for zs in z_pos])
        z_pos = [zs[:min_depth] for zs in z_pos]
        psfs = [psf[:min_depth] for psf in psfs]
        return psfs, z_pos

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
                    psfs.append(self.normalise_image(noisy_psf))
                    coords.append(coord)
            psfs = np.stack(psfs)
            coords = np.stack(coords)
            self.data['train'] = [psfs, coords]

        

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
            
            polar_coords = self.convert_xy_coords_to_polar(xy_coords)


            # dists = cdist(img_center, xy_coords).T
            # dists = (dists - dists.min()) / (dists.max() - dists.min())

            self.data[k] = [[input_data, polar_coords], target_data]

class ExperimentalDataSet(GenericDataSet):
    def __init__(self, config, filter_localisations=True, lazy=False, transform_data=False, normalize_psf=True):
        super().__init__(config, transform_data)
        self.filter_emitters_proximity = filter_localisations
        self.normalize_psf = normalize_psf
    
        if 'z_step' not in config:
            self.z_step = config['voxel_sizes'][0]
        else:
            self.z_step = 0

        if len(self.img.shape) == 2:
            self.img = self.img[np.newaxis]
        
        if not lazy:
            self.prepare_data()

    def prepare_data(self):
        psfs, xyz_coords = self.fetch_all_emitters()
        from final_project.smlm_3d.data.visualise import grid_psfs
        # plt.imshow(grid_psfs(psfs))
        # plt.show()
        # psfs = np.stack([remove_bg(p) for p in psfs])
        # plt.imshow(grid_psfs(psfs))
        # plt.show()
        if DEBUG:
            scatter_3d(xyz_coords)
        psfs = psfs.astype(np.float32)
        # print(psfs.shape)
        # imwrite('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/tmp/out.tif', psfs, compress=6)
        # quit()

        if self.transform_data:
            input_data = self.transform_input(psfs)
            input_data = np.hstack((input_data, xyz_coords[:, (0, 1)]))
        else:
            input_data = psfs[:, :, :, np.newaxis]

        xy_coords = xyz_coords[:, (0, 1)]

        self.xyz_coords = xyz_coords
        polar_coords = self.convert_xy_coords_to_polar(xy_coords)


        self.data = [input_data, polar_coords]


    def fetch_all_emitters(self):
        all_psfs = []
        all_xyz_coords = []

        pixel_coords, filtered_df = self.fetch_emitters_coords(self.img, self.csv_data)
        self.kept_idx = list(filtered_df.index)
        # for frame in set(filtered_df['frame']):
        #     print(frame)
        #     plt.imshow(self.img[frame-1])
        #     plt.show()
        #     df = filtered_df[filtered_df['frame'] == frame]
        #     df.plot.scatter('x [nm]', 'y [nm]')
            # plt.show()

        for i, pixel_coord in enumerate(tqdm(pixel_coords)):
            n_detections = filtered_df.iloc[i].get('detections') or 1
            frame_idx = int(((filtered_df.iloc[i]['frame']) - 1) + (n_detections - 1)) 
            frame = self.img[frame_idx]
            # fig, ax = plt.subplots()
            # ax.imshow(frame)
            # import matplotlib.patches as patches
            # rect = patches.Rectangle([p-16 for p in pixel_coord], 32, 32, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)   
            # plt.show()         
            psf = self.cut_image_stack(frame, pixel_coord, width=self.bound)
            if self.normalize_psf:
                psf = self.normalise_image(psf)
                psf = remove_bg(psf)
            xy_coord = filtered_df.iloc[i][['x [nm]', 'y [nm]']].to_numpy()
            all_psfs.append(psf)
            all_xyz_coords.append(list(xy_coord) + [frame_idx * self.voxel_sizes[0]])
        all_xyz_coords = np.array(all_xyz_coords)
        all_psfs = np.stack(all_psfs)

        return all_psfs, all_xyz_coords


    def predict_dataset(self, model):
        coord_diff = model.predict(self.data).squeeze()

        # plt.hist(coord_diff)
        # plt.show()
        # plt.plot(coord_diff)
        # plt.show()

        output_coords = self.xyz_coords.copy()
        output_coords[:, 2] += coord_diff

        # for x, y_orig, y_new in zip(output_coords[:, 0], output_coords[:, 2], output_coords[:, 2]+coord_diff):
        #     plt.plot([x, x], [y_orig, y_new])
        
        # plt.scatter(output_coords[:, 0], output_coords[:, 2]+coord_diff, label='final_pos', marker='+', alpha=0.3)
        # plt.legend()
        # plt.xlabel('x')
        # plt.ylabel('z')
        # plt.show()
        # plt.scatter(output_coords[:, 1], output_coords[:, 2])
        # plt.xlabel('y')
        # plt.ylabel('z')
        # plt.show()

        psfs = self.data[0]
        idx = np.argsort(coord_diff)
        psfs = psfs[idx]
        # for psf in psfs:
        #     plt.imshow(psf)
        #     plt.title(f'{psf.max()}')
        #     plt.show()

        return output_coords

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2).astype(int)

    mask = dist_from_center <= radius
    return mask


class StormDataset(ExperimentalDataSet):
    def __init__(self, config, lazy=False, transform_data=False, normalize_psf=True):
        super().__init__(config, filter_localisations=False, lazy=True, transform_data=transform_data, normalize_psf=normalize_psf)
        self.filter_emitters_proximity=False
        self.filter_emitters_edges=True
        print(self.img.shape)
        if not lazy:
            self.prepare_data()
        mask = create_circular_mask(self.bound*2, self.bound*2, radius=8)
        for i in range(self.data[0].shape[0]):
            self.data[0][i][~mask] = 0

# class SphereTrainingDataset(TrainingDataSet):
#     filter_emitters_proximity = True
    
#     def __init__(self, config, z_range, filter_emitters_proximity=True, split_data=False, lazy=False, transform_data=False, add_noise=False, normalize_psf=True, radius=5e5+50, exclude_coverslip=True, disable_emitter_peak_boundary=False):
#         self.radius = radius
#         self.exclude_coverslip = exclude_coverslip
#         super().__init__(config, z_range, filter_emitters_proximity, split_data, lazy, transform_data, add_noise, normalize_psf, disable_emitter_peak_boundary)

#     def sphere_loss(self, p, fit_data=None):
#         x0, y0, z0 = p
#         x, y, z = fit_data.T
#         return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - self.radius

#     def fit_sphere(self, coords):
#         initial_guess = [coords[:, 0].mean(), coords[:, 1].mean(), coords[:, 2].mean() + self.radius]

#         low_bounds = [
#             coords[:, 0].min(),
#             coords[:, 1].min(),
#             coords[:, 2].min(),
#         ]
#         high_bounds = [
#             coords[:, 0].max(),
#             coords[:, 1].max(),
#             coords[:, 2].max() + (self.radius * 1.5),
#         ] 
#         sphere_loss = partial(self.sphere_loss, fit_data=coords)
#         res = least_squares(sphere_loss, initial_guess, bounds=(low_bounds, high_bounds), verbose=True, ftol=1e-100,
#                             xtol=1e-100, max_nfev=1000)
#         centre = res.x[0:3]
#         print(res.fun.min(), res.fun.max())
#         print('Initial\t', initial_guess)
#         print('Lower\t', low_bounds)
#         print('Final\t', centre)
#         print('Upper\t', high_bounds)
        
#         return centre, res.fun
    
#     def double_fit_sphere(self, _coords, subset=True, subset_idx=None):

#         if subset_idx is None:
#             coords = _coords
#             subset_idx = np.arange(0, coords.shape[0])
#         else:
#             coords = _coords[subset_idx]
#         centre, residuals = self.fit_sphere(coords)
#         residuals = abs(residuals)
#         # self.measure_error(coords, centre)
#         if not subset:
#             # cutoff = 500
#             # points_idx = np.where(residuals < cutoff)[0]
#             # subset_idx = subset_idx[points_idx]
#             # print(f'Keeping {len(points_idx)} points from {_coords.shape[0]}.')
#             # plot_with_sphere(_coords[subset_idx], centre, self.radius)
#             return centre, subset_idx

#         percentile = 95
#         cutoff = np.percentile(residuals, percentile)
#         points_idx = np.where(residuals < max(cutoff, 1))[0]

#         print(f'Keeping {len(points_idx)} points from {_coords.shape[0]}.')

#         subset_idx = subset_idx[points_idx]
#         return self.double_fit_sphere(_coords, subset=False, subset_idx=subset_idx)
    

#     def measure_error(self, coords, centre):
#         corrected_coords = self.correct_z_pos(coords.copy(), centre)

#         abs_error = abs(corrected_coords[:, 2] - coords[:, 2])
#         if DEBUG:
#             plt.boxplot(abs_error)
#             plt.show()
#         print(np.mean(abs_error))
#         return np.mean(abs_error)

#     def correct_z_pos(self, coords, sphere_centre):
#         def get_z_coord(coord):
#             a = 1
#             b = -2 * (sphere_centre[2])
#             c = sum([c ** 2 for c in sphere_centre]) \
#                 + (coord[0] ** 2) \
#                 + (coord[1] ** 2) \
#                 - (2 * ((coord[0] * sphere_centre[0]) + (coord[1] * sphere_centre[1]))) \
#                 - (self.radius ** 2)
#             coeffs = [a, b, c]
#             z_coords = np.roots(coeffs)
#             z_coord = min(z_coords, key=lambda x: abs(x - coord[2]))

#             return z_coord

#         z_coords = np.copy(coords[:, 2])
#         coords[:, 2] = np.apply_along_axis(get_z_coord, 1, coords)
#         if DEBUG:
#             plt.scatter(coords[:, 2], z_coords-coords[:, 2])
#             plt.show()
#         # Assert coordinates sit on sphere
#         assert sum(self.sphere_loss(sphere_centre, coords)) < 1e-5
#         return coords

#     def fetch_emitters_modelled_z(self):
#         psfs, xyz_coords = self.estimate_ground_truth()
#         z_pos = xyz_coords[:, 2]
#         lower_bound_z = z_pos.min() + self.config['coverslip_exclusion'][0]
#         upper_bound_z = z_pos.max() - self.config['coverslip_exclusion'][1]
#         keep_idx = np.where((lower_bound_z <= z_pos) & (z_pos <= upper_bound_z))[0]
#         print(f'Keeping {len(keep_idx)} points from {xyz_coords.shape[0]} after removing coverslip layers')
#         psfs = np.array(psfs)[keep_idx]
#         xyz_coords = xyz_coords[keep_idx]
#         self.csv_data = self.csv_data.iloc[keep_idx]

#         # from tifffile import imwrite
#         # for i, psf in enumerate(psfs):
#         #     imwrite(f'/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/tmp/psfs/{i}.tiff', psf, compress=6)
#         # quit()

#         psfs = np.stack(psfs)
#         if DEBUG:
#             scatter_yz(xyz_coords)
#             scatter_3d(xyz_coords)
#         sphere_centre, subset_idx = self.double_fit_sphere(xyz_coords, subset=True)
#         print(f'{len(subset_idx)} emitters after fitting sphere')

#         self.csv_data = self.csv_data.iloc[subset_idx]
#         psfs = psfs[subset_idx]
#         xyz_coords = xyz_coords[subset_idx]
#         if DEBUG:
#             # scatter_3d(xyz_coords)
#             scatter_yz(xyz_coords)
#         # plot_with_sphere(xyz_coords, sphere_centre, self.radius)

#         # self.measure_error(xyz_coords, sphere_centre)
#         xyz_coords = self.correct_z_pos(xyz_coords, sphere_centre)

#         min_max_z = psfs[0].shape[0]/2
#         z_placeholder = np.linspace(-min_max_z, min_max_z-1, psfs[0].shape[0]) * self.voxel_sizes[0]
#         # trimmed_psfs = []
#         # trimmed_zpos = []
#         # for psf, offset in zip(psfs, xyz_coords[:, 2]):
#         #     trim_psf, trim_zpos = self.trim_stack(psf, z_placeholder.copy() - offset) 
#         #     if len(trim_zpos) != 40:
#         #         continue
#         #     trimmed_psfs.append(trim_psf)
#         #     trimmed_zpos.append(trim_zpos)
#         psfs, z_pos = zip(*[self.trim_stack(psf, z_placeholder.copy() + offset) for psf, offset in zip(psfs, xyz_coords[:, 2])])

#         psfs, z_pos = self.trim_equal_length(psfs, z_pos)
#         z_pos = np.stack(z_pos)
        
#         xy_coords = xyz_coords[:, [0,1]]
#         xy_coords = xy_coords[:, np.newaxis, :]
#         z_pos = z_pos[:, :, np.newaxis]
#         xy_coords_rep = np.repeat(xy_coords, z_pos.shape[1], axis=1)

#         zxy_coords = np.concatenate((z_pos, xy_coords_rep), axis=2)
#         return [psfs, zxy_coords]

# def gen_sphere_dataset(surface_density, radius):
    
#     sphere_center = (0, 0, 0)
#     surface_area = (4/3)*np.pi*(radius**2)
#     n_points = int(surface_area * surface_density)
#     print(f'Placing {n_points} points')

#     def sample_spherical(npoints, ndim=3):
#         vec = np.random.randn(npoints, ndim)
#         vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
#         vec[:, 2] = -abs(vec[:, 2])
#         return vec

#     # Generate random vectors
#     sphere_sample = sample_spherical(n_points)
#     sphere_sample *= radius

#     idx = np.where(sphere_sample[:, 2] < (201 * 50 + sphere_sample[:, 2].min()))
#     sphere_sample = sphere_sample[idx]

#     for dim in range(sphere_sample.shape[1]):
#         sphere_sample[:, dim] += sphere_center[dim]

#     # Offset sphere to 0
#     sphere_sample[:, 2] = (sphere_sample[:, 2] - sphere_sample[:, 2].min()) + 100
#     df = pd.DataFrame(sphere_sample, columns=['x', 'y', 'z'])
#     if DEBUG:
#         scatter_3d(df.to_numpy())
#     return df.to_numpy()

if __name__=='__main__':
    from final_project.smlm_3d.config.datasets import dataset_configs

    z_range = 1000

    # dataset = 'paired_bead_stacks'
    # train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=True)
    # exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False, split_data=False)
    # print(np.stack(exp_dataset.all_psfs).shape)
    # print(exp_dataset.csv_data.shape)

    dataset = 'openframe'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=False)
    
    # sphere_dataset = SphereTrainingDataset(dataset_configs[dataset]['sphere_ground_truth'], transform_data=False)
    # dataset = 'matched_index_sphere'
    # train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=True, filter_emitters_proximity=False, split_data=True)
    # exp_dataset = SphereTrainingDataset(dataset_configs[dataset]['sphere_ground_truth_647nm'], transform_data=False, z_range=1000, split_data=True, add_noise=False, radius=5e5+50, exclude_coverslip=True)



    # dataset_name = 'other'
    # sub_name = 'sphere'
    # z_range = 1000
    # cfg = dataset_configs[dataset_name][sub_name]
    # dataset = ExperimentalDataSet(cfg, z_range, transform_data=False)
