import numpy as np
from data.visualise import scatter_3d
from config.datasets import dataset_configs
import pandas as pd
import os

sphere_center = [5e5, 5e5, 0]
radius = 5e5
surface_density_nm2 = 5e-8



def main(cfg, surface_density):
    surface_area = (4/3)*np.pi*(radius**2)
    print(surface_density)
    n_points = int(surface_area * surface_density)
    print(f'Placing {n_points} points')

    csv_path = os.path.join(cfg['bpath'], cfg['simulation_csv'])

    def sample_spherical(npoints, ndim=3):
        vec = np.random.randn(npoints, ndim)
        vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
        vec[:, 2] = -abs(vec[:, 2])
        return vec

    # Generate random vectors
    sphere_sample = sample_spherical(n_points)
    sphere_sample *= radius

    idx = np.where(sphere_sample[:, 2] < (201 * cfg['voxel_sizes'][0] + sphere_sample[:, 2].min()))
    sphere_sample = sphere_sample[idx]

    for dim in range(sphere_sample.shape[1]):
        sphere_sample[:, dim] += sphere_center[dim]

    # scatter_3d(sphere_sample)

    # Offset sphere to 0
    sphere_sample[:, 2] = (sphere_sample[:, 2] - sphere_sample[:, 2].min()) + 100
    df = pd.DataFrame(sphere_sample, columns=['x', 'y', 'z'])
    df.to_csv(csv_path, index=False)

    print(csv_path)


if __name__ == '__main__':
    cfg = dataset_configs['simulated_ideal_psf']['sphere_ground_truth']
    main(cfg, surface_density=surface_density_nm2)
