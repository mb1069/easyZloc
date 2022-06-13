import copy
import glob
import random
from functools import partial
from multiprocessing.pool import Pool

from pyotf.utils import prep_data_for_PR
from pyotf.zernike import name2noll
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
# model kwargs
from tqdm import tqdm

from src.config.datafiles import psf_modelling_file, data_dir
from src.config.optics import model_kwargs
from pyotf.otf import HanserPSF, apply_aberration
from pyotf.phaseretrieval import retrieve_phase
import pandas as pd

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from src.data.visualise import show_psf_axial
from src.data.evaluate import mse
from src.zernike_decomposition.gen_psf import gen_psf_named_params, gen_psf_modelled_param


# def get_psfs():
#     ds = JonnyDataSource(jonny_data_dir)
#     return ds.get_all_emitter_stacks()


def normalise_to_uint16(psf):
    psf = psf / psf.max()
    psf *= 65535
    psf = psf.astype(np.uint16)
    return psf


def normalise_to_float(psf):
    psf = psf.astype(np.float)
    psf = psf / psf.max()
    return psf


def fit_psf_lsquares(n_zerns, psf):
    _model_kwargs = model_kwargs.copy()
    _model_kwargs['zsize'] = psf.shape[0]
    _model_kwargs['size'] = psf.shape[1]

    original_psf = copy.deepcopy(psf)
    # Normalise PSF and cast to uint16
    psf = normalise_to_uint16(psf)


    psf = prep_data_for_PR(psf, multiplier=1.01)

    PR_result = retrieve_phase(
        psf, _model_kwargs, max_iters=1000, pupil_tol=0, mse_tol=0, phase_only=True,
    )
    PR_result.fit_to_zernikes(n_zerns)
    # PR_result.plot()
    # PR_result.plot_convergence()
    mcoefs = PR_result.zd_result.mcoefs
    pcoefs = PR_result.zd_result.pcoefs

    result_psf = gen_psf_modelled_param(mcoefs, pcoefs)

    # show_all_psfs((psf, result_psf))

    res = {
        'mse': mse(normalise_to_float(original_psf), normalise_to_float(result_psf))
    }
    for i in range(n_zerns):
        res[f'pcoef_{i}'] = pcoefs[i]
        res[f'mcoef_{i}'] = mcoefs[i]
    return res


def fit_multiple_psfs(psfs, n_zerns):
    pfunc = partial(fit_psf_lsquares, n_zerns)
    with Pool(4) as p:
        results = list(tqdm(p.imap(pfunc, psfs), total=len(psfs)))

    df = pd.DataFrame.from_records(results)

    df.to_csv(psf_modelling_file, index=False)
    avg_df = df.mean(axis=0)

    avg_mcoefs = avg_df[[f'mcoef_{i}' for i in range(n_zerns)]].to_numpy()
    avg_pcoefs = avg_df[[f'pcoef_{i}' for i in range(n_zerns)]].to_numpy()

    avg_psf = gen_psf_modelled_param(avg_mcoefs, avg_pcoefs)
    avg_psf = avg_psf / avg_psf.max()
    print(avg_psf.shape)

    # psfs = [avg_psf] + list(random.sample(psfs, 5))
    # psfs = [((psf / psf.max()) * 65535).astype(np.uint16) for psf in psfs]
    # psfs = [prep_data_for_PR(psf, multiplier=1.01) for psf in psfs]
    # all_psfs = np.concatenate(psfs, axis=2)
    # print(all_psfs.shape)
    # show_psf_axial(all_psfs)


def main():
    # psfs, coordinates = get_psfs()
    # all_mcoefs = []
    # all_pcoefs = []
    # for psf in psfs:
    #     mcoefs, pcoefs = fit_psf_lsquares(psf)
    #     all_mcoefs.append(mcoefs)
    #     all_pcoefs.append(pcoefs)
    #
    # coordinates = [list(c) for c in coordinates]
    # coords = np.array(coordinates)
    # all_mcoefs = np.stack(all_mcoefs)
    # all_pcoefs = np.stack(all_pcoefs)
    #
    # data = np.hstack((coords, all_mcoefs, all_pcoefs))
    # labels = ['x', 'y'] + [f'mcoef_{i}' for i in range(all_mcoefs.shape[1])] + [f'pcoef_{i}' for i in
    #                                                                             range(all_mcoefs.shape[1])]
    # df = pd.DataFrame(columns=labels, data=data)
    #
    # df.to_csv('models.csv')

    df = pd.read_csv('models.csv')

    coef_cols = [c for c in list(df) if 'pcoef' in c]

    df = df[coef_cols]

    plt.axis('on')
    df.boxplot()
    plt.xticks(rotation=45)

    plt.show()


def named_aberration_to_pcoefs(aberration, magnitude):
    try:
        noll = name2noll[aberration]
    except KeyError as e:
        raise KeyError(
            f"Aberration '{aberration}' unknown, choose from: '"
            + "', '".join(name2noll.keys())
            + "'"
        )
    pcoefs = np.zeros(len(name2noll))
    pcoefs[noll - 1] = magnitude
    return pcoefs


if __name__ == '__main__':
    psfs = []
    for psf_path in list(
            glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif')):
        psfs.append(imread(psf_path))
    # fit_multiple_psfs(psfs, 32)

    pcoefs = named_aberration_to_pcoefs('oblique astigmatism', 1)
    # pcoefs2 = named_aberration_to_pcoefs('Vertical astigmatism', 1)
    # pcoefs = np.add(pcoefs, pcoefs2)

    params = {
        'oblique astigmatism': 1,
        # 'vertical astigmatism': 1
    }

    target_psf = gen_psf_named_params(params)

    target_psf_path = '/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/2.tif'
    target_psf = imread(target_psf_path)

    target_psf = target_psf / target_psf.max()
    target_psf *= 255
    target_psf = target_psf.astype(np.uint8)

    # pcoefs = np.random.uniform(0, 1, 15)
    # target_psf = gen_psf_modelled_param(None, pcoefs)
    # print(center_of_mass(target_psf))

    model_kwargs['size'] = target_psf.shape[1]
    model_kwargs['zsize'] = target_psf.shape[0]

    # base: 0.0015392293
    # target_psf = target_psf.astype(np.float32)
    # target_psf /= target_psf.max()
    # show_psf_axial(target_psf)

    res = fit_psf_lsquares(32, target_psf)

    result_psf = gen_psf_modelled_param(res['mcoefs'], res['pcoefs'])
    result_psf = result_psf / result_psf.max()

    target_psf = result_psf / result_psf.max()

    diff = target_psf - result_psf

    print('Diff', diff.min(), diff.max())
    comparison = np.concatenate((target_psf, diff, result_psf), axis=2)
    show_psf_axial(comparison)

    print([round(r, 3) for r in pcoefs])
    print([r for r in res['pcoefs']])

    print(mse(result_psf, target_psf))
    # target_psf = imread(
    #     '/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/data/jonny_psf_emitters/15.tif').astype(np.float32)

    target_psf /= target_psf.max()

    target_psf_shape = target_psf.shape

    model = HanserPSF(**model_kwargs)

    PR_result = retrieve_phase(
        target_psf, model_kwargs, max_iters=10000, pupil_tol=0e-100, mse_tol=0e-100, phase_only=False,
    )

    print(PR_result)
    print(PR_result.mse)
    PR_result.plot()
    PR_result.plot_convergence()
    PR_result.fit_to_zernikes(8)
    PR_result.zd_result.plot()
    PR_result.zd_result.plot_named_coefs()
    plt.show()

    mcoefs = PR_result.zd_result.mcoefs
    pcoefs = PR_result.zd_result.pcoefs
    print(mcoefs.shape)
    print(pcoefs.shape)
    # make the model
    model = HanserPSF(**model_kwargs)

    model = apply_aberration(model, mcoefs, pcoefs)

    model_psf = model.PSFi
    model_psf /= model_psf.max()
    model_psf = model_psf.astype(np.float32)

    both_psf = np.concatenate((target_psf, model_psf), axis=2)
    print(both_psf.shape)
    plt.imshow(np.concatenate(both_psf[slice(0, 41, 5)], axis=0))
    plt.axis('off')
    plt.show()
    quit()
    # imwrite('/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/data/test_out.tif', both_psf.astype(np.float32))
    #
    # quit()
    #
    # # extract kr
    # model._gen_kr()
    # kr = model._kr
    # theta = model._phi
    # # make zernikes (need to convert kr to r where r = 1 when kr is at
    # # diffraction limit)
    # r = kr * model.wl / model.na
    # mask = r <= 1
    # zerns = zernike(r, theta, np.arange(5, 16))
    # # make fake phase and magnitude coefs
    # np.random.seed(12345)
    # pcoefs = np.random.randn(zerns.shape[0])
    # mcoefs = np.random.rand(zerns.shape[0])
    # pupil_phase = (zerns * pcoefs[:, np.newaxis, np.newaxis]).sum(0)
    # pupil_mag = (zerns * mcoefs[:, np.newaxis, np.newaxis]).sum(0)
    # pupil_mag = pupil_mag + model._gen_pupil() * (2.0 - pupil_mag.min())
    # # phase only test
    # # model.apply_pupil(pupil_mag * np.exp(1j * pupil_phase) * model._gen_pupil())
    # PSFi = np.random.poisson(model.PSFi / model.PSFi.max() * 1000) + np.random.randn(*model.PSFi.shape)
    # # we have to converge really close for this to work.
    #
    # PSFi = imread('/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/data/test_otf.tif')
    # PR_result = retrieve_phase(
    #     PSFi, model_kwargs, max_iters=1000, pupil_tol=0, mse_tol=0, phase_only=False
    # )
    #
    # ga_fit_psf = PR_result.generate_psf(size=128, zsize=40, zrange=list(range(40)))
    #
    # print(ga_fit_psf.shape)
