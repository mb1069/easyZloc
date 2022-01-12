from final_project.smlm_3d.data.visualise import scatter_3d, show_psf_axial
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from scipy.optimize import curve_fit


from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.util import get_base_data_path, dwt_inverse_transform, chunks
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.debug_tools.view_dataset_psfs import grid_psfs
from tifffile import imwrite

DISABLE_LOAD_SAVED = True
FORCE_LOAD_SAVED =  False
DEBUG = False

USE_GPU = True
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src/wavelets/wavelet_data/output')

model_path = os.path.join(os.path.dirname(__file__), 'tmp/model.json')


model_options = {
    'n_estimators': 2000 if not DEBUG else 5,
    'gamma': 0,
    'max_bin': 2048,
    'max_depth': 8,
    'min_child_weight': 1,
    'verbosity': 1,
    'tree_method': 'gpu_hist',
    'n_jobs': 4,
    'nthread': 4
}

def load_model():
    model = define_model()
    model.load_model(model_path)
    return model


def define_model():
        # XGBoost
    model = XGBRegressor(**model_options, learning_rate=0.1)

    return model

def concat_dataset_features(dataset):
    for k, v in dataset.items():
        dataset[k][0] = np.hstack(dataset[k][0])
        print(k, v[0].shape, v[1].shape)

def train_model(dataset, val_dataset=None):
    if val_dataset:
        # Swap existing val_dataset for specified val_dataset
        dataset['val'] = val_dataset

    # # # LightGBM

    # import lightgbm as lgb
    # param = {'num_leaves': 2000,
    #          'boosting_type': 'gbdt',
    #          'task': 'train',
    #          'objective': 'regression',
    #          'metric': 'mae',
    #          'device': 'gpu' if USE_GPU else 'cpu',
    #          'min_data_in_leaf': 1
    #          }
    
    # model = lgb.LGBMRegressor(**param)
    
    # num_round = 500
    
    # model = model.fit(*dataset['train'], num_round, eval_set=[dataset['val']], early_stopping_rounds=3,
    #                   init_model=pretrained_model)
    

    for i in range(2):
        try:
            model = None
            if os.path.exists(model_path):
                if FORCE_LOAD_SAVED or (not DISABLE_LOAD_SAVED) and ('y' in input('Load saved model?')):
                    return load_model()
            if model is None:
                model = define_model()
            model.fit(*dataset['train'],
                      eval_metric='mae',
                      eval_set=[dataset['val']],
                      early_stopping_rounds=3,
                      verbose=True,
                      )
            model.save_model(model_path)

        except xgboost.core.XGBoostError as e:
            print(e)
            del model_options['tree_method']
            # del model_options['num_parallel_tree']
            continue
        break

    return model
    # num_round = 500
    # model = model.fit(*train_dataset, num_round, eval_set=[val_dataset], early_stopping_rounds=5, verbose=True)

def measure_error(model, test_dataset):
    x, y = test_dataset
    y_pred = model.predict(x).squeeze()
    ae = abs(y_pred - y.squeeze())
    return ae

def shift_correction(y, y_pred):
    f = lambda x, m, c: (x*m) + c 
    popt, pcov = curve_fit(f, y, y_pred, p0=[1, 0])
    shift = popt[1]
    y_pred -= shift
    # print('Pred shift', np.mean(abs(y_pred-y)))
    # print('Post shift', np.mean(abs(y_pred-y)))
    return y_pred

def chunk_stacks(y_pred, y):
    chunked_y = []
    chunked_ypred = []
    chunk_limits = [0] + [i+1 for i in range(len(y)-1) if y[i] > y[i+1]]
    for i in range(len(chunk_limits)-1):
        start = chunk_limits[i]
        end = chunk_limits[i+1]

        chunked_y.append(y[start:end])
        chunked_ypred.append(y_pred[start:end])
    return chunked_y, chunked_ypred

def eval_model(model, test_dataset, title, w_shift_correction=False):
    x, y = test_dataset
    y_pred = model.predict(x).squeeze()
    y = y.squeeze()
    ae = abs(y_pred - y)
    if w_shift_correction:
        # shifts preds
        y_pred = shift_correction(y, y_pred)

    ae = abs(y_pred - y)
    # idx = np.where(ae < np.percentile(ae, 75))
    # ae = ae[idx]
    # y = y[idx]
    # y_pred = y_pred[idx]
    plt.boxplot(ae)
    plt.title(f'{title} MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')
    plt.ylabel('Absolute error (nm)')
    plt.show()
    plt.title(f'{title} MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')

    chunked_y, chunked_ypred = chunk_stacks(y_pred, y)

    for stack_y, stack_y_pred in zip(chunked_y, chunked_ypred):
        plt.plot(stack_y, stack_y_pred)

    # plt.scatter(y, y_pred)
    plt.ylabel('pred')
    plt.xlabel('truth')
    plt.show()

    # df = pd.DataFrame.from_dict({'z_pos': y, 'error': ae})
    # df.to_csv('/home/miguel/Projects/uni/phd/smlm_z/final_project/tmp/tmp.csv')

    print('MAE:', round(ae.mean(), 4))
    return ae.mean(), ae.std()

def inspect_large_errors(model, dataset):
    x, y = dataset.data['all']
    y_pred = model.predict(x).squeeze()
    y = y.squeeze()
    ae = abs(y_pred - y)
    print(ae.mean())
    print(np.median(ae))
    plt.scatter(y, y_pred)
    plt.show()

    y_pred = shift_correction(y, y_pred)
    psfs = np.stack(dataset.all_psfs)
    coords = np.stack(dataset.all_coords)
    ae = np.stack(list(chunks(ae, psfs.shape[1])))

    y_pred = np.stack(list(chunks(y_pred, psfs.shape[1])))
    y = np.stack(list(chunks(y, psfs.shape[1])))
    

    stack_coords = np.stack([n[0] for n in coords])
    
    df = {
        'errors': ae.mean(axis=1),
        'x': stack_coords[:, 0],
        'y': stack_coords[:, 1]
    }

    df = pd.DataFrame.from_dict(df)
    # df.plot.scatter('x', 'y', c='errors')
    # plt.show()
    dataset.csv_data['errors'] = ae.mean(axis=1)

    dataset.csv_data.to_csv('error_coords.csv')
    bad_stacks = []
    good_stacks = []
    stack_errors = (abs(y - y_pred)).flatten()
    for i in range(len(y)):
        # show_psf_axial(psfs[i])
        plt.plot(y[i], y_pred[i])
        plt.xlabel('Truth')
        plt.ylabel('Pred')
    plt.title(f'Mean error: {str(round(np.mean(stack_errors), 3))} nm, std: {str(round(np.std(stack_errors), 3))}')

    plt.show()
    print('done')
    quit()

    
    stack_errors = [np.mean(abs(y[i]-y_pred[i])) for i in max_errors]
    low_perc = np.percentile(stack_errors, 25)
    high_perc = np.percentile(stack_errors, 75)

    good_stacks = psfs[np.where(stack_errors < low_perc)]

    bad_stacks = psfs[np.where(stack_errors >= high_perc)]

    bad_imgs = grid_psfs(bad_stacks)
    good_imgs = grid_psfs(good_stacks)

    imwrite('bad_localisations.tiff', bad_imgs, compress=6)
    imwrite('good_localisations.tiff', good_imgs, compress=6)


def predict(model, exp_dataset, fname):
    z = model.predict(exp_dataset['wavelets']).squeeze()
    df = exp_dataset['df']
    df['z [nm]'] = z

    outpath = os.path.join(RESULTS_DIR, fname)
    df.to_csv(outpath)
    return df


def visualise_dataset(fname):
    outpath = os.path.join(RESULTS_DIR, fname)

    df = pd.read_csv(outpath)
    print(df.shape)
    n_samples = 5000000
    if df.shape[0] > n_samples:
        df = df.sample(n_samples)
    fig = plt.figure()
    plt.set_cmap('brg')

    ax = fig.add_subplot(projection='3d')

    x = df['x [nm]']
    y = df['y [nm]']
    z = df['z [nm]']
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z) * 100))

    ax.scatter(x, y, z, marker='.', s=3, c=z)
    plt.show()


def main():
    # # # Get jonny bead stacks
    # jonny_dataset = load_jonny_dataset(50000, 1000)
    # # Train model
    # model = train_model(jonny_dataset)

    #
    # print('Multiwell trained')
    # eval(model, test_bead_stack['test'])
    #
    # model = train_model(test_bead_stack, model)
    # print('Retrained')
    # eval(model, test_bead_stack['test'])

    # TODO LUPUS NEPHRITIS - Subsampling test
    # Red stack
    # dwt_level = [9]
    # results = []
    # for d in dwt_level:
    #     test_bead_stack = load_red_lupus_nephritis_bead_stack(level=d)
    #     model = train_model(test_bead_stack, None)
    #     mae, std = eval(model, test_bead_stack['test'], 'Lupus Nephritis red channel')
    #     results.append({
    #         'dwt_level': d,
    #         'error': mae,
    #         'std': std
    #     })
    #     del model
    #     gc.collect()
    # df = pd.DataFrame.from_records(results)
    # df.plot(x='dwt_level', y='error', yerr='std', capsize=4)
    # plt.ylabel('Axial localisation error (nm)')
    # plt.xlabel('Wavelet decomposition level')
    # plt.show()
    # print(df)

    # LUPUS NEPHRITIS
    # Red stack
    # red_bead_stack = load_red_lupus_nephritis_bead_stack()
    # model = train_model(red_bead_stack, None)
    # eval(model, red_bead_stack['test'], 'Lupus Nephritis red channel')
    # red_exp_data = load_red_lupus_nephritis_experimental_data()
    # predict(model, red_exp_data, 'lupus_red.csv')

    # Green stack
    # green_bead_stack = load_green_lupus_nephritis_bead_stack()
    # model = train_model(green_bead_stack, None)
    # eval(model, green_bead_stack['test'], 'Lupus Nephritis green channel')
    # green_exp_data = load_green_lupus_nephritis_experimental_data()
    # predict(model, green_exp_data, 'lupus_green.csv')

    # membranous_glomerulonephritis
    # Red stack
    # red_bead_stack = load_red_membranous_glomerulonephritis_bead_stack()
    # model = train_model(red_bead_stack, None)
    # eval(model, red_bead_stack['test'], 'Membranous glomerulonephritis red channel')
    # red_exp_data = load_red_membranous_glomerulonephritis_experimental_data()
    # predict(model, red_exp_data, 'membranous_glomerulonephritis_red.csv')
    # visualise_dataset('membranous_glomerulonephritis_red.csv')

    # Green stack
    # green_bead_stack = load_green_membranous_glomerulonephritis_bead_stack()
    # model = train_model(green_bead_stack, None)
    # eval(model, green_bead_stack['test'], 'Membranous glomerulonephritis green channel')
    # green_exp_data = load_green_membranous_glomerulonephritis_experimental_data()
    # predict(model, green_exp_data, 'membranous_glomerulonephritis_green.csv')
    # visualise_dataset('membranous_glomerulonephritis_green.csv')
    #
    # bead_stack = load_olympus_3d_bead_stack()
    # model = train_model(bead_stack, None)
    # eval(model, bead_stack['test'], 'Olympus 3d')

    # 11nm bead stack
    # mm11_bead_stack = load_11mm_bead_stack()
    # model = train_model(mm11_bead_stack, None)
    # eval(model, mm11_bead_stack['test'], 'MM11 bead stack')
    # dfs = []
    # for slc in range(10, 160, 10):
    #     print(f'Slice {slc}')
    #     try:
    #         mm11_exp_data = load_11mm_experimental_data(slc)
    #         df = predict(model, mm11_exp_data, 'mm11_bead_stack.csv')
    #         dfs.append(df)
    #     except IndexError:
    #         break
    # df = pd.concat(dfs, ignore_index=True)
    # df.to_csv('mm11_bead_stack.csv')
    # visualise_dataset('mm11_bead_stack.csv')

    # Retrained MAE: 122nm
    """
    LGB
        60nm
        123 retrained
        78 from scratc
    
    XGBoost
        62nm
        73nm retrained
        73nm from scratch
        
    NN
        63nm
        411 retrained
        77.9 from scratch
    """

    # Train on bead stack
    # Eval model on exp bead stack

    # Run on exp data
    z_range = 1000


    dataset = 'paired_bead_stacks'


    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=True, add_noise=False)

    exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=True, add_noise=False, split_data=False)


    concat_dataset_features(train_dataset.data)
    concat_dataset_features(exp_dataset.data)

    # model = train_model(train_dataset.data) 

    model = load_model()
    eval_model(model, train_dataset.data['test'], 'Bead test (bead stack training)')
    eval_model(model, exp_dataset.data['all'], 'Sphere (bead stack training)', w_shift_correction=True)

    inspect_large_errors(model, exp_dataset)

if __name__ == '__main__':
    main()
