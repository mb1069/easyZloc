from pandas.io.formats.format import return_docstring
from final_project.smlm_3d.data.datasets import ExperimentalDataSet, TrainingDataSet
from functools import partial
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from scipy.optimize import least_squares
from tifffile import imread
from scipy.spatial.distance import cdist

# from src.wavelets.wavelet_data.datasets.training_dataset import TrainingDataset
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.data.visualise import plot_with_sphere, scatter_3d, scatter_yz
from final_project.smlm_3d.config.datafiles import res_file
# from final_project.smlm_3d.workflow_v2 import load_model
from final_project.smlm_3d.experiments.deep_learning import load_model

fname = os.path.join(os.path.dirname(__file__), '..', 'tmp/animation.gif')

def sphere_loss(p, radius, fit_data=None):
    x0, y0, z0 = p
    x, y, z = fit_data.T
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - radius

def fit_sphere(df, radius, with_bounds=True, top_or_bottom='bottom'):
    fit_data = df[['x', 'y', 'z']].to_numpy()

    if top_or_bottom == 'bottom':
        invert_z = 1
    else:
        invert_z = -1

    initial_guess = [df['x'].mean(), df['y'].mean(), df['z'].min() + (invert_z * radius)]

    z_bounds = [
        df['z'].min() + (invert_z * radius * 0.75),
        df['z'].min() + (invert_z * radius * 1.25),
    ]
    if with_bounds:
        low_bounds = [
            df['x'].min() - radius,
            df['y'].min() - radius,
            min(z_bounds)
        ]
        high_bounds = [
            df['x'].max() + radius,
            df['y'].max() + radius,
            max(z_bounds)
        ]
        bounds = (low_bounds, high_bounds)
        print('low', low_bounds)
        print('initial', initial_guess)
        print('high', high_bounds)
    else:
        bounds = ([-np.inf]*3, [np.inf]*3)

    sphere_loss_fn = partial(sphere_loss, radius=radius, fit_data=fit_data)
    res = least_squares(sphere_loss_fn, initial_guess, bounds=bounds, verbose=True, ftol=1e-16,
                        xtol=1e-16, max_nfev=1000, loss='soft_l1')
    centre = res.x[0:3]
    x = ['x', 'y', 'z']
    if with_bounds:
        plt.plot(x, low_bounds, label='low')
        plt.plot(x, high_bounds, label='high')
    plt.plot(x, initial_guess, ':', label='initial')
    plt.plot(x, centre, label='centre')
    plt.legend()
    plt.show()
    plot_with_sphere(df[x].to_numpy(), centre, radius)

    return centre, res.fun

class SphereDataset:
    radius = 5e5

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        scatter_yz(self.data[['x', 'y', 'z']].to_numpy(), title='Raw data')
        # self.data = self.exclude_coverslip(self.data)


    def measure_error(self):
        centre, residuals, res_df = self.double_fit_sphere(self.data)


        fitted_sphere_df = self.map_emitters_to_sphere(res_df, centre)
        # self.data['z'] = self.translate_z(fitted_sphere_df['z'], self.data['z'])

        scatter_3d(fitted_sphere_df.to_numpy(), title='Sphere data')

        self.plot_with_sphere(res_df, centre)

        plt.show()
        errors = abs(fitted_sphere_df['z'] - res_df['z'])
        plt.title(f'MAE: {round(errors.mean(), 3)}')
        plt.boxplot(errors)
        plt.show()

        # self.create_animation(fname, self.data, centre)

    @staticmethod
    def translate_z(s1, s2):
        total_error = lambda x, z: abs(x-z).sum()
        print(f'starting error: {total_error(s1,s2)}')
        coeffs = np.polyfit(s1, s2, deg=1)
        s2 = s2 - coeffs[-1]
        print(f'final error: {total_error(s1,s2)}')
        return s2
    
    @staticmethod
    def exclude_coverslip(df):
        df = df[df['z'] > (df['z'].min() + 400)]
        scatter_yz(df[['x', 'y', 'z']].to_numpy(), title='Coverslip removed')
        return df


    def double_fit_sphere(self, df):
        centre, residuals = fit_sphere(df, self.radius)
        # self.plot_with_sphere(self.data, centre)
        residuals = abs(residuals)
        keep_idx = np.where(residuals < np.percentile(residuals, 90))

        res_df = df.iloc[keep_idx]

        centre, residuals = fit_sphere(res_df, self.radius)
        return centre, residuals, res_df


    def plot_with_sphere(self, df, centre):
        dist_centre = cdist(df[['x', 'y']].to_numpy(), np.array([centre[0:2]]))
        dist_centre_y = [(d if y>centre[1] else -d) for y, d in zip(df['y'], dist_centre)]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # YZ view
        ax1.scatter(dist_centre_y, df['z'])
        zerod_centre = centre.copy()
        zerod_centre[1] = 0.0
        ax1.add_artist(plt.Circle(zerod_centre[[1, 2]], self.radius, color='red',fill=False))
        ax1.set_xlabel('y (distance from centre)')
        ax1.set_ylabel('z')

        ax2.scatter(df['y'], df['z'])
        ax2.add_artist(plt.Circle(centre[[1, 2]], self.radius, color='red',fill=False))
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')

        return

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        n_meridians = 20
        n_circles_latitude = 100
        u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
        sphere_x = centre[0] + self.radius * np.cos(u) * np.sin(v)
        sphere_y = centre[1] + self.radius * np.sin(u) * np.sin(v)
        sphere_z = centre[2] + self.radius * np.cos(v)

        ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
        ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
        ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

        ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
        ax.scatter(*centre)

        plt.show()

    def create_animation(self, fname, df, centre):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        n_meridians = 20
        n_circles_latitude = 100
        u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
        sphere_x = centre[0] + self.radius * np.cos(u) * np.sin(v)
        sphere_y = centre[1] + self.radius * np.sin(u) * np.sin(v)
        sphere_z = centre[2] + self.radius * np.cos(v)

        ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
        ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
        ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        def init():
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

            ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
            ax.scatter(*centre)
            return fig,

        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=360, interval=20, blit=True)
        # Save
        anim.save(fname, writer='imagemagick', fps=30)

    def map_emitters_to_sphere(self, df, sphere_centre):
        df = df.copy(deep=True)

        def get_z_coord(r):
            a = 1
            b = -2 * (sphere_centre[2])
            c = sum([c ** 2 for c in sphere_centre]) \
                + (r['x'] ** 2) \
                + (r['y'] ** 2) \
                - (2 * ((r['x'] * sphere_centre[0]) + (r['y'] * sphere_centre[1]))) \
                - (self.radius ** 2)
            coeffs = [a, b, c]
            z_coords = np.roots(coeffs)
            z_coord = min(z_coords, key=lambda x: abs(x - r['z']))

            return z_coord

        df['z'] = df.apply(get_z_coord, axis=1)
        # Assert coordinates sit on sphere
        assert sum(sphere_loss(sphere_centre, df[['x', 'y', 'z']].to_numpy()), self.radius) < 1e-5
        return df

def main():

    dataset = 'other'

    # truth_file = res_file.replace('.csv', '_truth.csv')
    # cfg = dataset_configs[dataset]['sphere_ground_truth']
    # truth_dataset = ExperimentalDataSet(cfg, lazy=True)
    # truth_coords = truth_dataset.estimate_ground_truth()
    # truth_df = pd.DataFrame(data=truth_coords, columns=['x', 'y', 'z'])
    # truth_df.to_csv(truth_file, index=False)
    # target_file = truth_file



    model = load_model()
    cfg = dataset_configs[dataset]['sphere']

    # train_dataset = TrainingDataSet(cfg, transform_data=False, z_range=1000, split_data=False, add_noise=False)
    # x, z = train_dataset.data['all']
    # pred_z = model.predict(x).squeeze()
    # plt.scatter(z, pred_z)
    # plt.xlabel('True z')
    # plt.ylabel('Pred z')
    # ae = abs(pred_z - z)
    # plt.title(f'MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')
    # plt.show()
    # quit()

    est_file = res_file.replace('.csv', '_truth.csv')
    exp_dataset = ExperimentalDataSet(cfg, transform_data=False)
    pred_coords = exp_dataset.predict_dataset(model)
    pred_df = pd.DataFrame(data=pred_coords, columns=['x', 'y', 'z'])
    # pred_df = pred_df[(np.percentile(pred_df['z'], 25) < pred_df['z']) & (np.percentile(pred_df['z'], 75) > pred_df['z'])]
    pred_df.to_csv(est_file, index=False)
    target_file = est_file


    # test_file = res_file.replace('.csv', '_test.csv')
    # def sample_spherical(npoints, ndim=3):
    #     vec = np.random.randn(ndim, npoints)
    #     vec /= np.linalg.norm(vec, axis=0)
    #     return vec
    
    # sphere_sample = sample_spherical(50000)
    # for dim in range(sphere_sample.shape[0]):
    #     sphere_sample[dim] = (sphere_sample[dim] - sphere_sample[dim].min()) / (sphere_sample[dim].max() - sphere_sample[dim].min())
    # sphere_sample *= 1e6

    # sphere_sample = sphere_sample.T

    # idx = np.where(sphere_sample[:,2] < (15050 + sphere_sample[:,2].min()))
    # print(sphere_sample.shape)
    # sphere_sample = sphere_sample[idx]
    # print(sphere_sample.shape)

    # scatter_3d(sphere_sample)
    # test_df = pd.DataFrame(data=sphere_sample, columns=['x', 'y', 'z'])
    # test_df.to_csv(test_file, index=False)
    # target_file = test_file



    ds = SphereDataset(target_file)
    ds.measure_error()


if __name__ == '__main__':
    main()
