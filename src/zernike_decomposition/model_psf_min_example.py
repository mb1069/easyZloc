import glob
import os

from matplotlib import cm
from pyotf.otf import HanserPSF, apply_aberration
from pyotf.phaseretrieval import retrieve_phase
from pyotf.utils import prep_data_for_PR, center_data
from tifffile import imread, imshow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2).sum()


def gen_csv():
    mses = []

    all_mcoefs = []
    all_pcoefs = []
    imgs = []

    n_zerns = 120
    for psf_path in list(glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif')):
        imgs.append(os.path.basename(psf_path))
        target_psf = imread(psf_path)

        model_kwargs = dict(
            wl=660,
            na=1.3,
            ni=1.51,
            res=85,
            size=target_psf.shape[1],
            zsize=target_psf.shape[0],
            zres=100,
            vec_corr="none",
            condition="none",
        )


        # Normalise PSF and cast to uint8
        target_psf = target_psf / target_psf.max()
        target_psf *= 255
        target_psf = target_psf.astype(np.uint8)

        target_psf = prep_data_for_PR(target_psf, multiplier=1.01)

        # Retrieve phase for experimental PSF
        PR_result = retrieve_phase(
            target_psf, model_kwargs, max_iters=100, pupil_tol=0, mse_tol=0,
            phase_only=False,
        )

        PR_result.fit_to_zernikes(n_zerns)

        pcoefs = PR_result.zd_result.pcoefs
        mcoefs = PR_result.zd_result.mcoefs

        all_mcoefs.append(mcoefs)
        all_pcoefs.append(pcoefs)
        # Simulate HanserPSF with parameters

        result_psf = HanserPSF(**model_kwargs)
        result_psf = apply_aberration(result_psf, PR_result.zd_result.mcoefs, PR_result.zd_result.pcoefs)

        # Render side-by-side axial slices

        result_psf = result_psf.PSFi

        target_psf = target_psf / target_psf.max()
        result_psf = result_psf / result_psf.max()
        diff = abs(target_psf - result_psf)
        print(os.path.basename(psf_path), mse(target_psf, result_psf))
        mses.append(mse(target_psf, result_psf))
        comb_psf = np.concatenate((target_psf, diff, result_psf), axis=2)
        comb_psf = comb_psf[slice(5, 40, 3)]
        comb_psf = np.concatenate(comb_psf, axis=0)
        # imshow(comb_psf)
        # PR_result.plot_convergence()
        # plt.show()

    all_pcoefs = np.stack(all_pcoefs).T
    all_mcoefs = np.stack(all_mcoefs).T
    imgs = np.stack(imgs)[:, np.newaxis].T
    mses = np.stack(mses)[:, np.newaxis].T
    res = np.concatenate((imgs, mses, all_pcoefs, all_mcoefs), axis=0).T

    df = pd.DataFrame(data=res, columns=['img_name', 'mse', *[f'pcoef_{i}' for i in range(n_zerns)], *[f'mcoef_{i}' for i in range(n_zerns)]])
    df.to_csv('./psf_models.csv')


def read_csv():
    coords = pd.read_csv('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/coords.csv')
    coords = coords[['x', 'y']]
    print(coords.shape)

    df = pd.read_csv('./psf_models.csv')
    print(df.shape)

    df = pd.concat((coords, df), axis=1)
    print(df.shape)
    from scipy import stats
    cmap = cm.get_cmap('Spectral')
    # for c in [f'pcoef_{i}' for i in range(32)]:
    #     outliers = df[np.abs(df[c]-df[c].mean()) >= (2*df[c].std())]['img_name'].tolist()
    #     for o in outliers:
    #         print(f'open /Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/{o}')
    #     print(outliers)
    #     # df.plot.scatter('x', 'y', c=c, cmap=cmap)
    #     # plt.show()
    df.boxplot(column=[f'pcoef_{i}' for i in range(32)])
    plt.xticks(rotation=45)
    plt.show()
    df.boxplot(column=[f'mcoef_{i}' for i in range(32)])
    plt.xticks(rotation=45)
    plt.show()
    df.boxplot(column='mse')
    plt.xticks(rotation=45)
    plt.show()

if __name__=='__main__':
    # gen_csv()
    read_csv()