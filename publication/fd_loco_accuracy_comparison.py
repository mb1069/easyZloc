from argparse import ArgumentParser
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import optimize as opt
from sklearn.metrics import root_mean_squared_error
import os
import matplotlib.pyplot as plt

from util.util import load_dataset, load_model


def bestfit_error(z_true, z_pred):
    def linfit(x, c):
        return x + c

    x = z_true
    y = z_pred
    popt, _ = opt.curve_fit(linfit, x, y, p0=[0])

    x = np.linspace(z_true.min(), z_true.max(), len(y))
    y_fit = linfit(x, popt[0])
    error = root_mean_squared_error(y_fit, y)
    return error, popt[0], y_fit, abs(y_fit-y)

def remove_constant_error(xys, zs, pred_zs):
    coords2 = ['_'.join(x.astype(str)) for x in xys]
    all_errors = []
    for c in set(coords2):
        idx = [i for i, val in enumerate(coords2) if val==c]
        _, _, _, errors = bestfit_error(zs[idx], pred_zs[idx])
        all_errors.append(errors)
    all_errors = np.concatenate(all_errors)
    return all_errors

def get_z_coordinates(dataset):
    zs = []
    xys = []
    for (_, xy), z in dataset.as_numpy_iterator():
        zs.append(z)
        xys.append(xy)

    return np.concatenate(xys).squeeze(), np.concatenate(zs).squeeze()


def get_model_performance(args):
    model = load_model(args)

    results = {}

    for name in ['test']:
        dataset = load_dataset(name, args)
        xys, zs = get_z_coordinates(dataset)
        # coords2 = np.array(['_'.join(x.astype(str)) for x in xys])
        pred_zs = model.predict(dataset, batch_size=4096).squeeze()

        errors = remove_constant_error(xys, zs, pred_zs)
        results[name] = (zs, errors)
    return results

def load_ries_data(args):
    ries_data = pd.read_csv(args['ries_data'])
    cols = list(ries_data)
    ries_deeploc = ries_data[[c for c in cols if ('DeepLoc' in c) or ('z(nm)' in c)]].dropna().set_index('z(nm)')
    crlb_deeploc = ries_data[[c for c in cols if ('CRLB' in c) or ('z(nm)' in c)]].dropna().set_index('z(nm)')
    return ries_deeploc, crlb_deeploc
    

def plot_comparison(model_perf, ries_data, args, fname):

    outdir = os.path.join(args['outdir'], 'fdloco_comparison')
    os.makedirs(outdir, exist_ok=True)
    
    for dataset, (z, error) in model_perf.items():
        plt.title(f'{dataset} RMSE compared to FD-Loco')
        sns.regplot(x=z, y=error, scatter=True, ci=95, order=25, x_bins=np.arange(-1000, 1000, 50), label='Our method')
        sns.lineplot(data=ries_data)
        outpath = os.path.join(outdir, f'{fname}_comparison.png')
        plt.ylabel('Localisation RMSE')
        plt.xlabel('z [nm]')
        plt.savefig(outpath)
        plt.close()
        print(f'Wrote {outpath}')


    
def main(args):
    model_perf = get_model_performance(args)
    ries_data, crlb_data = load_ries_data(args)

    plot_comparison(model_perf, ries_data, args, fname='fd-loco')
    plot_comparison(model_perf, crlb_data, args, fname='crlb')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--ries_data', default='/home/miguel/Projects/smlm_z/publication/ries_comparison_data.csv')
    return vars(parser.parse_args())

if __name__ == '__main__':
    main(parse_args())