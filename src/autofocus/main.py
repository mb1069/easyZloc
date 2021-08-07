import pandas as pd
from xgboost import XGBRegressor
import xgboost
import matplotlib.pyplot as plt
from src.autofocus.data import gather_data, split_dataset, remove_low_variance_features
from src.autofocus.config import cfgs, get_cfg_images, dwt_level
import numpy as np



def eval(model, test_dataset):
    x, y = test_dataset
    x = x.astype(np.float32)
    y_pred = model.predict(x)

    ae = y_pred - y.squeeze()

    minbin = round(y.min() / 2) * 2
    maxbin = round(y.max() / 2) * 2
    r = maxbin - minbin
    if r > 50000:
        bins = np.linspace(-80, 80, 9)
        title = 'Long range autofocus error'
    else:
        bins = np.linspace(-10, 10, 11)
        title = 'Short range autofocus error'
    df = pd.DataFrame.from_dict({'defocus': y / 1000, 'ae': ae / 1000})
    df['defocus_bin'] = pd.cut(df['defocus'], bins)
    df.boxplot(by='defocus_bin', column='ae', rot=45)
    plt.title(title)
    plt.suptitle("")
    plt.ylabel('Mean prediction error (um)')
    plt.xlabel('Defocus (um)')
    plt.tight_layout()
    plt.show()

    df.boxplot(by='defocus_bin', column='ae', rot=45)
    plt.title(title)
    plt.suptitle("")
    plt.ylabel('Mean prediction error (um)')
    plt.xlabel('Defocus (um)')
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.show()

    # plt.boxplot(ae)
    # plt.title(f'{title} MAE: {round(ae.mean(), 4)} STDev: {round(ae.std(), 4)}')
    # plt.yscale('log')
    # plt.ylabel('Absolute error (nm)')
    # plt.show()
    # plt.scatter(y.squeeze(), y_pred.squeeze())
    # plt.xlabel('Ground truth (nm)')
    # plt.ylabel('Predicted position (nm)')
    # plt.show()

    print('MAE:', round(abs(ae).mean(), 4))


def train_model(dataset):
    for k, v in dataset.items():
        print(k, v[0].shape, v[1].shape)

    # XGBoost
    model_options = {
        'n_estimators': 2000,
        'gamma': 0,
        'max_bin': 1024,
        'max_depth': 8,
        'min_child_weight': 1,
        'verbosity': 1,
        'tree_method': 'gpu_hist',
        'n_jobs': 4,
        'nthread': 4,
        # 'subsample': 0.1,
        # 'sampling_method': 'gradient_based',
    }

    # Loop used to adjust parameters if GPU not available
    for i in range(2):
        try:
            model = XGBRegressor(**model_options, learning_rate=0.05)
            model.fit(*dataset['train'],
                      eval_metric='mae',
                      eval_set=[dataset['val']],
                      early_stopping_rounds=3,
                      verbose=True,
                      )

        except xgboost.core.XGBoostError as e:
            print(e)
            del model_options['tree_method']
            continue
        break

    eval(model, dataset['test'])
    return model


def main():

    cfg_key = 'slit_50nm'
    # Select cfg from config.py
    cfg = cfgs[cfg_key]    

    imgs = get_cfg_images(cfg)


    if 'cylindrical' in cfg_key:
        # Cylindrical data only has 1 or two stacks, so train/val/test sets are not from individual stacks
        xs, ys = gather_data(imgs, slice(0, len(imgs)+1), dwt_level=dwt_level)
        xs, _ = remove_low_variance_features(xs)
        dataset = split_dataset(xs, ys)
    else:
        # Slit datasets contain more stacks, so 20% of stacks are kept as a separate test
        # Use 0:train_test_split for training/validation and train_test_split:end for test
        train_test_split = int(len(imgs) * 0.8)
        xs, ys = gather_data(imgs, slice(0, train_test_split), dwt_level=dwt_level)
        xs, cols = remove_low_variance_features(xs)
        dataset = split_dataset(xs, ys)

        val_xs, val_ys = gather_data(imgs, slice(train_test_split, len(imgs)), dwt_level=dwt_level)

        val_xs = val_xs[:, cols]

        validation_stacks = (val_xs, val_ys)
    
    model = train_model(dataset)

    # Extra validation step on withheld stacks 
    if 'slit' in cfg_key:
        eval(model, validation_stacks)


if __name__ == '__main__':
    main()
