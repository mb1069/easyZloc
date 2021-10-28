from functools import partial
import jax
from jax.experimental import optimizers
from pyotf.otf import HanserPSF, apply_aberration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyotf.utils import prep_data_for_PR
from tqdm import trange, tqdm

import pandas as pd
import matplotlib.pyplot as plt
from final_project.smlm_3d.data.datasets import ExperimentalDataSet
from final_project.smlm_3d.workflow_v2 import load_model
from final_project.smlm_3d.config.datasets import dataset_configs
import numpy as np
from ggplot import *
from tqdm import tqdm
from final_project.smlm_3d.config.optics import model_kwargs
from src.data.evaluate import mse
from src.data.visualise import show_psf_axial


def view_psf(psf):
    show_psf_axial(psf)
    plt.show()


def model_psf(model_kwargs, coefs):
    pcoefs, mcoefs = coefs
    if mcoefs is None:
        mcoefs = np.zeros(pcoefs.shape)
    model = HanserPSF(**model_kwargs)
    psf = apply_aberration(model, mcoefs, pcoefs)
    return psf


def error_func(target_psf, psf):
    model = psf.PSFi
    model = model / model.max()
    return mse(target_psf, model)


def diff_func(kwargs, target_psf, coefs):
    pcoefs, mcoefs = coefs
    psf = model_psf(kwargs, (pcoefs, mcoefs))
    return error_func(target_psf, psf)


def optimize_model(config, target_psf, n_coefs, target_pcoefs=None, plot=True):
    target_psf = (target_psf * 255 * 255).astype(np.int64)
    
    target_psf = prep_data_for_PR(target_psf, multiplier=1.1)
    target_psf = target_psf / target_psf.max()

    kwargs = {k:v for k, v in model_kwargs.items()}
    kwargs['zres'] = config['voxel_sizes'][0]
    kwargs['res'] = config['voxel_sizes'][1]  
    kwargs['wl'] = config['wl']
    kwargs['na'] = config['na']
    kwargs['zsize'] = target_psf.shape[0]
    kwargs['size'] = target_psf.shape[1]

    initial_pcoefs = np.random.uniform(0, 1, (n_coefs,))

    initial_pcoefs[:] = 0

    initial_mcoefs = np.random.uniform(0, 1, (n_coefs,))

    opt_init, opt_update, get_params = optimizers.sgd(20)
    opt_state = opt_init((initial_pcoefs.copy(), initial_mcoefs.copy()))

    func = partial(diff_func, kwargs, target_psf)
    def step(step, opt_state):
        value, grads = jax.value_and_grad(func)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    patience = 100
    early_stop_count = 0
    all_data = []
    tol = 1e-7

    best_mse = np.inf
    best_iter = None
    ITER = 1000
    for i in trange(ITER, position=0, leave=True):
        value, opt_state = step(i, opt_state)
        log_data = {
            'epoch': i,
            'mse': float(value)
        }
        for pcoef, val in enumerate(opt_state.packed_state[0][0].real._value):
            log_data[f'pcoef_{pcoef}'] = val

        for mcoef, val in enumerate(opt_state.packed_state[1][0].real._value):
            log_data[f'mcoef_{mcoef}'] = val

        all_data.append(log_data)
        mse_val = float(value)
        if best_mse - mse_val < tol:
            early_stop_count += 1
            if early_stop_count == patience:
                print('Early stopping')
                break
        else:
            early_stop_count = 0

        tqdm.write(f'{i} {"%.4E" % round(value, 20)} - {"%.4E" % round(best_mse, 20)} - {early_stop_count} - {best_mse - mse_val}')


        if mse_val < best_mse:
            best_iter = log_data
            best_mse = mse_val



    final_coefs = tuple([opt_state.packed_state[i][0].real._value for i in range(2)])

    if plot:
        final_psf = model_psf(kwargs, final_coefs).PSFi

        final_psf = final_psf / final_psf.max()

        initial_psf = model_psf(kwargs, (initial_pcoefs, None)).PSFi
        initial_psf = initial_psf / initial_psf.max()
        target_psf = target_psf / target_psf.max()

        comparison = np.concatenate((initial_psf, final_psf, target_psf), axis=2)
        plt.title('Initial - Final - Target')
        view_psf(comparison)

        df = pd.DataFrame.from_records(all_data, index='epoch')
        df['mse'] = df['mse'] / df['mse'].max()

        ax = df.plot()

        if target_pcoefs is not None:
            for i, val in enumerate(target_pcoefs):
                plt.axhline(y=val, color=ax.lines[i + 1].get_color(), linestyle='dotted', label='_Hidden')
        plt.show()

    return best_iter, all_data

train_config = dataset_configs['openframe']['training']
train_dataset = ExperimentalDataSet(train_config, lazy=True)
train_dataset.prepare_debug()

sphere_config = dataset_configs['openframe']['sphere_ground_truth']
sphere_dataset = ExperimentalDataSet(sphere_config, lazy=True)
sphere_dataset.prepare_debug()

n_coefs = 16
records = []
n_emitters = min(10, train_dataset.total_emitters)
for emitter_id in trange(n_emitters):
    try:
        psf, dwt, coords, z, record = train_dataset.debug_emitter(emitter_id, z_range=1000)
        coeffs = optimize_model(train_config, psf, n_coefs, plot=False)[0]
        coeffs = {k: v for k, v in coeffs.items() if 'pcoef' in k or k == 'mse'}
        coeffs['type'] = 'train'
        records.append(coeffs)
    except RuntimeError:
        pass

    # try:
    #     psf, dwt, coords, z, record = sphere_dataset.debug_emitter(emitter_id, z_range=1000)
    #     coeffs = optimize_model(sphere_config, psf, n_coefs, plot=False)[0]
    #     coeffs = {k: v for k, v in coeffs.items() if 'pcoef' in k or k == 'mse'}
    #     coeffs['type'] = 'sphere'
    #     records.append(coeffs)
    # except RuntimeError:
    #     pass

df = pd.DataFrame.from_records(records)
df.to_csv('tmp.csv', index=False)
df = pd.read_csv('tmp.csv')
for col in list(df):
    if col =='type':
        continue
    df.boxplot(col, by='type')
    plt.show()
quit()


exp_dataset.prepare_debug()
psf, dwt, coords, z = exp_dataset.debug_emitter(emitter_id, z_range=1000)

coeffs2 = optimize_model(config, psf, n_coefs, plot=True)
pcoefs2, mcoefs2 = coeffs2[0]

plt.scatter(pcoefs, pcoefs2, label='pcoefs')
# plt.scatter(mcoefs, mcoefs2, label='mcoefs')
max_val = int(max(pcoefs.max(), pcoefs2.max(), mcoefs.max(), mcoefs2.max()))
plt.plot(list(range(0, max_val+1)), list(range(0, max_val+1)), ':')
plt.legend()
plt.show()
