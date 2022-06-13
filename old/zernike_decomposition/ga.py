import copy
import os
import pickle
from functools import partial

import numpy as np
from deap import base, algorithms
from deap import creator
from deap import tools
from scipy.ndimage import center_of_mass
from tifffile import imread
import matplotlib.pyplot as plt
import multiprocessing
import random

from data.pyOTF.pyotf.otf import HanserPSF, apply_aberration
from src.config.optics import model_kwargs as _model_kwargs
from src.data.evaluate import mse
from src.data.visualise import show_psf_axial

model_kwargs = copy.deepcopy(_model_kwargs)


def center_axial(psf):
    axial_intensity = psf.max(axis=(1, 2))
    peak = np.argmax(axial_intensity)
    if peak < psf.shape[0] / 2:
        centered_psf = psf[0:2 * peak]
    else:
        start = peak - (psf.shape[0] - peak)
        centered_psf = psf[start:]
    return centered_psf


def center_lateral(psf):
    lateral_intensity = psf.max(axis=0)
    peak = np.unravel_index(np.argmax(lateral_intensity), psf.shape)
    print(2, peak)


def center_peak(psf):
    # x, y, z = ndimage.measurements.center_of_mass(psf)
    psf = center_axial(psf)
    # psf = center_lateral(psf)

    return psf


def plot_axial_intensity(psf, label):
    psf = psf.astype(np.float)
    psf = psf / psf.max()
    axial_intensity = psf.max(axis=(1, 2))
    y = axial_intensity
    x = list(range(len(y)))
    plt.plot(x, y, label=f'{label}_real')
    plt.legend()


# target_psf = HanserPSF(**model_kwargs)
# cfg = {
#     "oblique astigmatism": random.uniform(-4, 4),
#     "vertical astigmatism": random.uniform(-4, 4),
#     "vertical coma": random.uniform(-4, 4),
#     "horizontal coma": random.uniform(-4, 4),
#     "vertical trefoil": random.uniform(-4, 4),
#     "oblique trefoil": random.uniform(-4, 4)
# }
# for c, v in cfg.items():
#     target_psf = apply_named_aberration(target_psf, c, v)
#
# target_psf = target_psf.PSFi
# target_psf = target_psf / target_psf.max()
N_ZERNS = 16

POP = 1000
NGEN = 100
CXPB = 0.4
MUTPB = 0.5


def sim_psf(model_kwargs, pcoefs):
    model = HanserPSF(**model_kwargs)
    model = apply_aberration(model, None, pcoefs)

    psf = model.PSFi
    psf /= psf.max()
    psf = psf.astype(np.float32)
    return psf


def evaluate(target_psf, model_kwargs, individual, plot=False):
    # mcoefs = individual.mcoefs
    pcoefs = individual.pcoefs
    model_psf = sim_psf(model_kwargs, pcoefs)

    if plot:
        diff = target_psf - model_psf
        _psfs = np.concatenate((target_psf, model_psf, diff), axis=2)
        show_psf_axial(_psfs)
        plot_axial_intensity(target_psf, 'target_psf')
        plot_axial_intensity(model_psf, 'ga_fit_psf')
        plt.show()

    return mse(model_psf, target_psf),


def mutate(ind, mu, sigma, indpb):
    # for i in range(0, len(ind.mcoefs)):
    #     if random.random() < indpb:
    #         ind.mcoefs[i] += random.gauss(mu, sigma)

    for i in range(0, len(ind.pcoefs)):
        if random.random() < indpb:
            ind.pcoefs[i] += random.gauss(mu, sigma)
    return ind,


def mate(ind1, ind2):
    crossover = random.randint(1, len(ind1.pcoefs))

    # ind1_mcoefs = np.hstack((ind1.mcoefs[0:crossover][:], ind2.mcoefs[crossover:][:]))
    # ind2_mcoefs = np.hstack((ind2.mcoefs[0:crossover][:], ind1.mcoefs[crossover:][:]))

    ind1_pcoefs = np.hstack((ind1.pcoefs[0:crossover][:], ind2.pcoefs[crossover:][:]))
    ind2_pcoefs = np.hstack((ind2.pcoefs[0:crossover][:], ind1.pcoefs[crossover:][:]))

    # ind1.mcoefs = ind1_mcoefs
    ind1.pcoefs = ind1_pcoefs

    # ind2.mcoefs = ind2_mcoefs
    ind2.pcoefs = ind2_pcoefs
    return ind1, ind2


class Individual:
    def __init__(self, *args, **kargs):
        # self.mcoefs = np.random.uniform(-1, 1, N_ZERNS)
        self.pcoefs = np.random.uniform(-1, 1, N_ZERNS)


toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", Individual, fitness=creator.FitnessMin)

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, Individual, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", mate)
toolbox.register("mutate", mutate, mu=0, sigma=2, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=5)


def ga_fit_psf(target_psf, n_zerns=N_ZERNS, save=True, plot_best=True):
    global POP
    global NGEN
    global N_ZERNS

    N_ZERNS = n_zerns
    DEBUG = False

    if DEBUG:
        POP = 100
        NGEN = 100
    else:
        pool = multiprocessing.Pool(4)
        toolbox.register("map", pool.map)

    toolbox.register("evaluate", partial(evaluate, target_psf, model_kwargs))

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    hof = tools.HallOfFame(10)

    pop = toolbox.population(n=POP)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof,
                                   verbose=True)

    best = hof.items[0]
    best_mse = evaluate(target_psf, model_kwargs, best, plot_best)[0]
    ppath = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model.p')

    if save:
        best_ind = Individual()
        # best_ind.mcoefs = best.mcoefs
        best_ind.pcoefs = best.pcoefs
        with open(ppath, 'wb') as f:
            pickle.dump(best, f)

    return {
        # 'mcoefs': best.mcoefs,
        'pcoefs': best.pcoefs,
        'mse': best_mse
    }


def update_kwargs(target_psf):
    model_kwargs['size'] = target_psf.shape[1]
    model_kwargs['zsize'] = target_psf.shape[0]


def prepare_target_psf(target_psf, is_simulated_psf=False):
    target_psf = target_psf.astype(np.float32)
    if not is_simulated_psf:
        target_psf = center_peak(target_psf)
    target_psf = target_psf / target_psf.max()
    return target_psf


def prepare_and_model(target_psf, is_simulated_psf):
    target_psf = prepare_target_psf(target_psf, is_simulated_psf)
    update_kwargs(target_psf)
    ga_fit_psf(target_psf)


def main():
    target_psf_path = '/home/miguel/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/1.tif'
    target_psf = imread(target_psf_path)
    # pcoefs = named_aberration_to_pcoefs('Oblique astigmatism', 1)
    # pcoefs2 = named_aberration_to_pcoefs('Vertical astigmatism', 1)

    # pcoefs = np.add(pcoefs, pcoefs2)
    # pcoefs = np.random.uniform(0, 1, N_ZERNS)
    # target_psf = gen_psf_modelled_param(None, pcoefs)
    show_psf_axial(target_psf)

    prepare_and_model(target_psf, is_simulated_psf=False)


if __name__ == '__main__':
    main()
