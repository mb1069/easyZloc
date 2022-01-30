from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.data.visualise import scatter_3d, show_psf_axial
import numpy as np
import matplotlib.pyplot as plt
import dill
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d

def fit_plane(coords, subset=True, subset_idx=None):
    points = Points(coords)
    plane = Plane.best_fit(points)
    # plot_3d(
    #     points.plotter(c='k', s=75, alpha=0.2, depthshade=False),
    #     plane.plotter(alpha=0.8, lims_x=(-50000, 50000), lims_y=(-50000, 50000)),
    # )
    # plt.xlabel('x (nm)')
    # plt.ylabel('y (nm)')
    # plt.show()

    dists_to_plane = np.array([abs(plane.distance_point_signed(p)) for p in points])
    print(f'{np.mean(dists_to_plane):.03} {np.std(dists_to_plane):.03}')

    if not subset:
        return dists_to_plane, points, subset_idx

    # percentile_threshold = 50
    # cutoff = np.percentile(dists_to_plane, percentile_threshold)
    points_idx = np.where(dists_to_plane < 100)[0]

    points_subset = coords[points_idx]
    return fit_plane(points_subset, subset=False, subset_idx=points_idx)


if __name__=='__main__':
    from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
    from final_project.smlm_3d.workflow_v2 import eval_model
    from final_project.smlm_3d.experiments.deep_learning import load_model
    z_range = 1000
    model_file = 'tmp.p'
    dataset = 'paired_bead_stacks'

    sub_dataset = 'training'
    exp_dataset = TrainingDataSet(dataset_configs[dataset][sub_dataset], transform_data=False, lazy=True, z_range=1000)
    model = load_model()

    coords = exp_dataset.estimate_ground_truth()
    # scatter_3d(coords)
    with open(model_file, 'wb') as f:
        dill.dump(coords, f)

    with open(model_file, 'rb') as f:
        coords = dill.load(f)

    # import pandas as pd
    # df = pd.read_csv('~/Downloads/100nm_Tetraspeck_beads_zstack_4um_10nm_647nm_300ms_1_MMStack_Default.csv')
    # coords = df[['x0 (um)', 'y0 (um)', 'z0 (um)']].to_numpy()
    # coords = coords*1000

    fit_plane(coords, subset=True)


# Peak pixel value
# 69.0 57.8
# 24.7 15.0

# Error with PSFj
# 84.4 / 76.7
# 34.4 / 19.5

    scatter_3d(coords)
    quit()