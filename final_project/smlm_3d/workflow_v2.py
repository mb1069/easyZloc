from final_project.smlm_3d.data.visualise import scatter_3d
from operator import truth
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.util import get_base_data_path
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs

DISABLE_LOAD_SAVED = False
FORCE_LOAD_SAVED = False
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
    'n_jobs': 8,
    'nthread': 8
}

def load_model():
    model = define_model()
    model.load_model(model_path)
    return model


def define_model():
        # XGBoost
    model = XGBRegressor(**model_options, learning_rate=0.1)

    return model

def train_model(dataset, pretrained_model=None):
    for k, v in dataset.items():
        print(k, v[0].shape, v[1].shape)

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

        except xgboost.core.XGBoostError:
            del model_options['tree_method']
            # del model_options['num_parallel_tree']
            continue
        break

    return model
    # num_round = 500
    # model = model.fit(*train_dataset, num_round, eval_set=[val_dataset], early_stopping_rounds=5, verbose=True)

def measure_error(model, test_dataset):
    x, y = test_dataset
    y_pred = model.predict(x)
    ae = abs(y_pred - y.squeeze())
    return ae

def eval_model(model, test_dataset, title):
    x, y = test_dataset
    y_pred = model.predict(x).squeeze()
    y = y.squeeze()
    ae = abs(y_pred - y)
    plt.boxplot(ae)
    plt.title(f'{title} MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')
    plt.ylabel('Absolute error (nm)')
    plt.show()
    plt.title(f'{title} MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')
    plt.scatter(y, y_pred)
    plt.ylabel('pred')
    plt.xlabel('truth')
    plt.show()
    
    df = pd.DataFrame.from_dict({'z_pos': y, 'error': ae})
    df.to_csv('/home/miguel/Projects/uni/phd/smlm_z/final_project/tmp/tmp.csv')



    print('MAE:', round(ae.mean(), 4))
    return ae.mean(), ae.std()


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

    dataset = 'openframe'
    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range)
    model = train_model(train_dataset.data)
    # exp_dataset = TrainingDataSet(dataset_configs[dataset]['sphere_ground_truth'], z_range, add_noise=False)
    # model2 = train_model(exp_dataset.data)


    eval_model(model, train_dataset.data['test'], 'Bead test (bead stack training)')
    # eval_model(model, exp_dataset.data['test'], 'Sphere (bead stack training)')




if __name__ == '__main__':
    main()
