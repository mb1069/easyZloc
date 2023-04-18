from pyotf.phaseretrieval import prep_data_for_PR
import jax.numpy as jnp
import jaxopt
from functools import partial
import matplotlib.pyplot as plt

from ..preprocessing.align_psfs import show_psf_axial
from ..preprocessing.nodes import norm_zero_one
from .psf_simulator import Simulator

def diff_func(params, simulator, target_psf, l1reg=0, l2reg=0):
    psf = simulator.get_scalar_psf(zern_coefs=params)
    psf = norm_zero_one(psf)
    error = jnp.mean((psf-target_psf)**2) 
    return error + (l1reg * jnp.sum(jnp.abs(params))) + (l2reg * jnp.sum(params**2))



def model_psf(simulator, target_psf, n_coefs, l1reg, l2reg):
    
    target_psf = prep_data_for_PR(target_psf, multiplier=1.1)


    target_psf = norm_zero_one(target_psf)
    x0 = jnp.array([0] * n_coefs).astype(jnp.float32)
    # res = minimize(fun=diff_func, method='BFGS', x0=x0, tol=1e-7, args=(simulator, target_psf))

    func = partial(diff_func, simulator=simulator, target_psf=target_psf, l1reg=l1reg, l2reg=l2reg)

    opt = jaxopt.GradientDescent(func, tol=1e-6, maxiter=2000, )
    res = opt.run(init_params=x0)

    params, state = res
    
    
    error = diff_func(params, simulator, target_psf)

    print(f'Final error: {error:.4E}')
    print(f'Num iters:', state.iter_num)

    initial_psf = simulator.get_scalar_psf(zern_coefs=x0)

    final_psf = simulator.get_scalar_psf(zern_coefs=params)

    initial_psf = simulator.get_scalar_psf()
    final_psf = final_psf / final_psf.max()
    initial_psf = initial_psf / initial_psf.max()
    target_psf = target_psf / target_psf.max()

    comparison = jnp.concatenate((initial_psf, final_psf, target_psf), axis=2)
    plt.title('Initial - Final - Target')
    show_psf_axial(comparison, '', 14)

    return params, error

def model_and_sim_beadstacks(bead_stacks, modelling_params):
    simulator = Simulator(n_coefs=modelling_params['n_zern_coefs'], **modelling_params['optical_params'])
