# from functools import partial
# import os
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt, animation
# from scipy.optimize import least_squares
# from tifffile import imread

# # from src.wavelets.wavelet_data.datasets.training_dataset import TrainingDataset

# from final_project.smlm_3d.data.visualise import scatter_3d
# from final_project.smlm_3d.config.datafiles import res_file

# fname = os.path.join(os.path.dirname(__file__), '..', 'tmp/animation.gif')


# class SphereDataset:
#     radius = 1e6

#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.data['z'] = self.data['z']

#     def measure_error(self):
#         centre, residuals = self.double_fit_sphere(self.data)

#         scatter_3d(self.data[['x', 'y', 'z']].to_numpy())
#         fitted_sphere_df = self.map_emitters_to_sphere(self.data, centre)
#         scatter_3d(fitted_sphere_df.to_numpy())

#         self.plot_with_sphere(self.data, centre)
#         errors = abs(fitted_sphere_df['z'] - self.data['z'])
#         plt.hist(errors)
#         plt.show()

#         # self.create_animation(fname, self.data, centre)

#     def double_fit_sphere(self, df):
#         centre, residuals = self.fit_sphere(df)
#         self.plot_with_sphere(self.data, centre)
#         residuals = abs(residuals)
#         keep_idx = np.where(residuals < np.percentile(residuals, 95))

#         res_df = df.iloc[keep_idx]

#         centre, residuals = self.fit_sphere(res_df)
#         return centre, residuals

#     def fit_sphere(self, df):
#         fit_data = df[['x', 'y', 'z']].to_numpy()

#         initial_guess = [df['x'].mean(), df['y'].mean(), df['z'].min() + self.radius]

#         low_bounds = [
#             df['x'].min(),
#             df['y'].min(),
#             df['z'].min(),
#         ]
#         high_bounds = [
#             df['x'].max(),
#             df['y'].max(),
#             df['z'].max() + (self.radius * 1.5),
#         ]
#         sphere_loss = partial(self.sphere_loss, fit_data=fit_data)
#         res = least_squares(sphere_loss, initial_guess, bounds=(low_bounds, high_bounds), verbose=True, ftol=1e-1000,
#                             xtol=1e-1000, max_nfev=1000)
#         centre = res.x[0:3]
#         print(initial_guess)
#         print(low_bounds)
#         print(high_bounds)
#         print(centre)
#         return centre, res.fun

#     def plot_with_sphere(self, df, centre):
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')

#         n_meridians = 20
#         n_circles_latitude = 100
#         u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
#         sphere_x = centre[0] + self.radius * np.cos(u) * np.sin(v)
#         sphere_y = centre[1] + self.radius * np.sin(u) * np.sin(v)
#         sphere_z = centre[2] + self.radius * np.cos(v)

#         ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
#         ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
#         ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#         ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

#         ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
#         ax.scatter(*centre)
#         plt.show()

#     def create_animation(self, fname, df, centre):
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')

#         n_meridians = 20
#         n_circles_latitude = 100
#         u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
#         sphere_x = centre[0] + self.radius * np.cos(u) * np.sin(v)
#         sphere_y = centre[1] + self.radius * np.sin(u) * np.sin(v)
#         sphere_z = centre[2] + self.radius * np.cos(v)

#         ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
#         ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
#         ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#         def init():
#             ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

#             ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
#             ax.scatter(*centre)
#             return fig,

#         def animate(i):
#             ax.view_init(elev=10., azim=i)
#             return fig,

#         # Animate
#         anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                        frames=360, interval=20, blit=True)
#         # Save
#         anim.save(fname, writer='imagemagick', fps=30)

#     def map_emitters_to_sphere(self, df, sphere_centre):
#         df = df.copy(deep=True)

#         def get_z_coord(r):
#             a = 1
#             b = -2 * (sphere_centre[2])
#             c = sum([c ** 2 for c in sphere_centre]) \
#                 + (r['x'] ** 2) \
#                 + (r['y'] ** 2) \
#                 - (2 * ((r['x'] * sphere_centre[0]) + (r['y'] * sphere_centre[1]))) \
#                 - (self.radius ** 2)
#             coeffs = [a, b, c]
#             z_coords = np.roots(coeffs)
#             z_coord = min(z_coords, key=lambda x: abs(x - r['z']))

#             return z_coord

#         df['z'] = df.apply(get_z_coord, axis=1)
#         # Assert coordinates sit on sphere
#         assert sum(self.sphere_loss(sphere_centre, df[['x', 'y', 'z']].to_numpy())) < 1e-5
#         return df

#     def sphere_loss(self, p, fit_data=None):
#         x0, y0, z0 = p
#         x, y, z = fit_data.T
#         return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) - self.radius


# # def load_data_as_3d_localisation_table():
# #     dpath = get_base_data_path() / 'experimental' / 'other' / '1mm_bead_june6_' / '635_red_stack_5_1'
# #     ds = SphereDataset(dpath, '635_red_stack_5_1_MMStack_Pos0.ome.tif', '635_red_stack_5_1_MMStack_Pos0.csv')
# #     df = ds.get_3d_localisation_table()
# #
# #     # # Remove emitters on coverslip
# #
# #     return df
# #
# #
# # def load_11mm_bead_stack(max_psfs=100000000, z_range=1000):
# #     dpath = get_base_data_path() / 'experimental' / 'other' / '1mm_bead_june6_' / '635_red_stack_5_1'
# #     ds = SphereDataset(dpath, '635_red_stack_5_1_MMStack_Pos0.ome.tif', '635_red_stack_5_1_MMStack_Pos0.csv')
# #     x, y = ds.prepare_data(max_psfs, z_range)
# #     return split_for_training(x, y)

# if __name__ == '__main__':
#     # ds = SphereDataset('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/experiments/sphere_check/olympus_truth.csv')
#     ds = SphereDataset(res_file)

#     ds.measure_error()
