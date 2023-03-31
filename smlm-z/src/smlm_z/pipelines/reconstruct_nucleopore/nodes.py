"""
This is a boilerplate pipeline 'reconstruct_nucleopore'
generated using Kedro 0.18.4
"""
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from ..train_model.nodes import create_boxplot


def cluster_locs(df, parameters):
    np.random.seed(parameters['random_seed'])
    cluster_params = parameters['cluster_parameters']
    coords = df[['x', 'y']].to_numpy() * parameters['images']['xy_res']
    cluster_ids = DBSCAN(
        eps=cluster_params['eps'], min_samples=cluster_params['min_count']).fit_predict(coords)
    df['cluster_id'] = cluster_ids.astype(str)
    ax = sns.scatterplot(data=df, x='x', y='y', hue='cluster_id')
    # df.to_csv('tmp.csv', index=False)
    # import os
    # print(os.getcwd() + '/tmp.csv')
    return df, ax.get_figure()


def norm_exp_coordinates(locs: pd.DataFrame, parameters: np.array) -> pd.DataFrame:
    img_y, img_x = parameters['images']['exp']['shape']
    locs['y'] = locs['y'] - (img_y/2)
    locs['x'] = locs['x'] - (img_x/2)
    train_img_y, train_img_x = parameters['images']['train']['shape'][1:]
    locs['y'] /= train_img_y
    locs['x'] /= train_img_x
    return locs


def predict_z(model, spots, df):
    coords = df[['x', 'y']].to_numpy()
    coords[:] = 0
    z_pos = model.predict((spots, coords)).squeeze()
    return z_pos


def scatter_3d(xyz_coords, title=None):
    xyz_coords = xyz_coords.astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs, ys, zs = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]
    cmap = plt.get_cmap("coolwarm")
    ax.scatter(xs, ys, zs, c=zs, cmap=cmap)

    if title:
        ax.set_title(title)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')

    return fig


def gm_min_bic(data):
    gm_df = pd.DataFrame({'pred': data.squeeze()},
                         index=np.arange(0, data.squeeze().shape[0]))

    best_gm = None
    min_bic = np.inf
    bics = []
    cov_type = 'tied'
    stdevs = []
    idx = np.argsort(data.squeeze())

    fig, axes = plt.subplots(1, 6)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    for n in range(1, 7):
        gm = GaussianMixture(n_components=n, n_init=20,
                             covariance_type=cov_type).fit(data)
        bic = gm.bic(data)

        bics.append(round(bic, 3))
        weight_range = gm.weights_.min() / gm.weights_.max()
        if bic < min_bic and weight_range > 0.3:
            min_bic = bic
            best_gm = gm

        labels = gm.predict(data).squeeze()

        gm_df['cluster_id'] = labels.astype(str)

        ax = axes[n-1]
        sns.histplot(data=gm_df, x='pred', hue='cluster_id',
                     stat='density', alpha=0.2, bins=20, ax=ax)

        # create necessary things to plot
        x_axis = np.linspace(data.min(), data.max(), 50)
        sub_df2 = pd.DataFrame.from_dict({'x': x_axis})
        for i in range(0, gm.n_components):
            if cov_type == 'tied':
                cov = gm.covariances_.squeeze()
            elif cov_type == 'full' or cov_type == None:
                cov = gm.covariances_[i][0][0]
            elif cov_type == 'spherical':
                cov = gm.covariances_[i]
            elif cov_type == 'diag':
                cov_type = gm.covariances_[i]

            sub_df2[f'y_{i}'] = norm.pdf(x_axis, float(
                gm.means_[i][0]), np.sqrt(cov))*gm.weights_[i]
            sns.lineplot(data=sub_df2, x='x', y=f'y_{i}', ax=ax)

    print(bics)
    print(f'Best EM: {best_gm.n_components}')
    plt.title(f'Best EM: {best_gm.n_components}')

    return best_gm.means_[:, 0], fig


def apply_gm(data):
    data = data.reshape(-1, 1)
    return gm_min_bic(data)

def recreate_sample(all_z_pos, locs, parameters):
    n_emitters = []
    dists_within_clusters = []

    all_coords = []
    all_fit_figs = []

    for cid in sorted(set(locs['cluster_id'])):
        if cid == '-1':
            continue
        idx = np.argwhere(locs['cluster_id'].to_numpy() == cid).squeeze()
        coords = np.zeros((idx.shape[0], 2))

        preds = all_z_pos[idx]
        # Useful to remove skew from gaussians
    #     preds -= preds.min()
    #     preds += 0.00000001
    #     preds = np.sqrt(preds)
        cluster_z_pos, fig = apply_gm(preds)
        all_fit_figs.append(fig)
        diffs = np.diff(sorted(cluster_z_pos))
        n_emitters.extend([len(cluster_z_pos)]*len(diffs))
        dists_within_clusters.extend(diffs)

        x, y = locs.iloc[idx][['x', 'y']].mean(axis=0).to_numpy() * parameters['images']['xy_res']
        coords = [[x, y, z, int(cid)] for z in cluster_z_pos]
        all_coords.extend(coords)

    all_coords = np.array(all_coords)

    res = pd.DataFrame.from_dict({
        k: all_coords[:, i] for k, i in zip(['x', 'y', 'z', 'cluster_id'], [0, 1, 2, 3])
    })

    fig_3d = scatter_3d(res[['x', 'y', 'z']].to_numpy())

    cluster_stats_fig = plt.figure()
    plt.scatter(x=n_emitters, y=dists_within_clusters)
    return fig_3d, all_fit_figs, cluster_stats_fig


def check_exp_data(psfs, df):
    xy_coords = df[['x', 'y']].to_numpy()
    X = (psfs, xy_coords)
    figs = []
    pixel_data = {
        'exp': X[0].mean(axis=(1, 2, 3)),
    }
    figs.append(create_boxplot(pixel_data, 'mean pixel val'))

    pixel_data = {
        'exp': X[0].max(axis=(1, 2, 3)),
    }
    figs.append(create_boxplot(pixel_data, 'max pixel val'))

    pixel_data = {
        'exp': X[0].min(axis=(1, 2, 3)),
    }
    figs.append(create_boxplot(pixel_data, 'min pixel val'))

    # Coord values
    x_data = {
        'exp': X[1][:, 0].flatten(),
    }
    figs.append(create_boxplot(x_data, 'x'))

    # Coord values
    y_data = {
        'exp': X[1][:, 1].flatten(),
    }
    figs.append(create_boxplot(y_data, 'y'))

    return figs
