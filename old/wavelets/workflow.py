import os
import pickle
from pathlib import Path
from tkinter import TclError

import xgboost
# from sklearn.model_selection import HalvingGridSearchCV
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import lightgbm as lgb

from src.data.data_processing import load_experimental_datasets, load_jonny_datasource
from src.wavelets.wavelet_data.util import limit_data_range, dwt_dataset
from src.wavelets.util import split_dataset, mae_dataset, view_dataset

modelpath = Path(__file__).parent / 'tmp/xgboost.model'
dpath = Path(__file__).parent / 'tmp/wavelets.p'


def get_datasets(z_range=1000, wavelet='sym4'):
    if not os.path.exists(dpath):
        # Experimental datasets
        # X, y = load_jonny_datasource(z_type='synth', max_psfs=1000)
        # X, y = limit_data_range(X, y, z_range=1000)
        #
        # if radius is not None:
        #     X = np.array([filter_freqs(x, radius) for x in X])
        # # print(X.shape, y.shape)
        # # dataset = load_corrected_datasets(0)
        # # X, y = dataset.x, dataset.y
        # # X, y = limit_data_range(X, y, z_range=1000)
        #
        # train_dataset, test_dataset, val_dataset, scalers = split_dataset(X, y)

        # Even/Odd scheme
        train_imgs = [f'{letter}{number}' for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] for number in range(1, 13, 2)]
        test_imgs = [f'{letter}{number}' for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] for number in range(2, 13, 2)]

        X, y = load_jonny_datasource(img_names=train_imgs, max_psfs=5000)
        X, y = limit_data_range(X, y, z_range=z_range)

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)

        train_dataset = [X_train, y_train]
        val_dataset = [X_val, y_val]

        X_test, y_test = load_jonny_datasource(img_names=test_imgs, max_psfs=5000)
        X_test, y_test = limit_data_range(X_test, y_test, z_range=z_range)

        test_dataset = [X_test, y_test]

        dataset = load_experimental_datasets('bead_stack')
        # X, y = dataset.emitters, dataset.z_pos
        # train_dataset, test_dataset, val_dataset, scalers = split_dataset(X, y)

        print(f'Train {[d.shape for d in train_dataset]}')
        print(f'Val {[d.shape for d in val_dataset]}')
        print(f'Test {[d.shape for d in test_dataset]}')

        for d in (train_dataset, test_dataset, val_dataset):
            d[0] = dwt_dataset(d[0])

        with open(dpath, 'wb') as f:
            pickle.dump((train_dataset, test_dataset, val_dataset), f)
    else:
        with open(dpath, 'rb') as f:
            (train_dataset, test_dataset, val_dataset) = pickle.load(f)

    return train_dataset, test_dataset, val_dataset


def train_model(train_dataset, val_dataset, test_dataset, pretrained_model=None):
    X_train, y_train = train_dataset
    X_val, y_val = val_dataset
    X_test, y_test = test_dataset


    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)
    param = {'num_leaves': 200,
             'boosting_type': 'gbdt',
             'task': 'train',
             'objective': 'regression',
             'metric': 'mae',
             'device': 'gpu'}

    if pretrained_model:
        param['task'] = 'refit'
    num_round = 500
    model = lgb.train(param, train_data, num_round, valid_sets=[val_data], early_stopping_rounds=5)


    model_options = {
        'n_estimators': 10000,
        'verbosity': 1,
        'tree_method': 'gpu_hist',
        'num_parallel_tree': 3
    }

    model_options.update({'gamma': 3, 'max_bin': 1024, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 10000})

    for i in range(2):
        try:
            if pretrained_model is None:
                model = XGBRegressor(**model_options, learning_rate=0.1)
            else:
                model = pretrained_model
            model.fit(X_train, y_train,
                      eval_metric='mae',
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=5,
                      verbose=True)

        except xgboost.core.XGBoostError:
            del model_options['tree_method']
            del model_options['num_parallel_tree']
            continue
        break

    training_mae = mae_dataset(model, X_train, y_train)
    print(f'Train mae: {round(training_mae, 3)}')

    val_mae = mae_dataset(model, X_val, y_val)
    print(f'Val mae: {round(val_mae, 3)}')

    test_mae = mae_dataset(model, X_test, y_test)
    print(f'Test mae: {round(test_mae, 3)}')

    return model


def predict_experimental_data(model):
    dataset = load_experimental_datasets('sample')
    df = dataset.csv_df

    X = dwt_dataset(dataset.emitters)

    pred = model.predict(X)

    df['z [nm]'] = pred
    return df


def retrain_test():
    if not os.path.exists(dpath):
        X, y = load_jonny_datasource(max_psfs=5000)
        X, y = limit_data_range(X, y, z_range=1000)
        X = dwt_dataset(X)

        train_dataset, val_dataset, test_dataset = split_dataset(X, y)
        with open(dpath, 'wb') as f:
            pickle.dump((train_dataset, test_dataset, val_dataset), f)
    else:
        with open(dpath, 'rb') as f:
            (train_dataset, test_dataset, val_dataset) = pickle.load(f)

    print('Multi well training')
    beadstack_model = train_model(train_dataset, val_dataset, test_dataset)

    beadstack_df = predict_experimental_data(beadstack_model)
    view_dataset(beadstack_df)

    dataset = load_experimental_datasets('bead_stack')
    X = dataset.emitters
    y = dataset.z_pos
    X = dwt_dataset(X)
    train_dataset, val_dataset, test_dataset = split_dataset(X, y)

    print('Multi well + bead stack training')
    retrained_model = train_model(train_dataset, val_dataset, test_dataset, pretrained_model=beadstack_model)

    retrained_df = predict_experimental_data(retrained_model)
    view_dataset(retrained_df)

    print('Bead stack training')
    retrained_model = train_model(train_dataset, val_dataset, test_dataset)

    retrained_df = predict_experimental_data(retrained_model)
    view_dataset(retrained_df)


if __name__ == '__main__':
    # retrain_test()
    # quit()
    res = []
    # for wavelet in ['sym2', 'sym3', 'sym4', 'sym5']:
    #     train_dataset, test_dataset, val_dataset, scalers = get_datasets(z_range=1000, wavelet=wavelet)
    #     # optimise_model(train_dataset)
    #     model = train_model(train_dataset, val_dataset)
    #
    #     mae = disp_jonny_datasets(model, test_dataset)
    #     res.append((wavelet, mae))
    # for r, mae in res:
    #     print(r, mae)
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'output.csv')

    df = None
    if os.path.exists(filename):
        if 'y' in input('Regen?: '):
            os.remove(filename)
        else:
            df = pd.read_csv(filename)

    if df is None:
        train_dataset, test_dataset, val_dataset = get_datasets()
        model = train_model(train_dataset, val_dataset, test_dataset)

        df = predict_experimental_data(model)

        df.to_csv(filename)
    try:
        view_dataset(df)
    except TclError:
        print('Failed to display dataset')
