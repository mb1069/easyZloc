import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost
import matplotlib.pyplot as plt
from src.autofocus.data import gather_data, imgs
import numpy as np


def split_dataset(xs, ys):
    x_train, x_other, y_train, y_other = train_test_split(xs, ys, train_size=0.8)
    x_val, x_test, y_val, y_test = train_test_split(x_other, y_other, train_size=0.5)

    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }


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
    dwt_level = 6
    # Use 0:cutoff for training/validation and cutoff:end for test

    cutoff = int(len(imgs) * 0.8)
    xs, ys = gather_data(slice(0, cutoff), dwt_level=dwt_level)

    xs_vars = np.var(xs, axis=0)

    cols = np.where(xs_vars > 0.1)[0]
    print(f'Removing {xs.shape[1] - len(cols)} features with low variance.')
    xs = xs[:, cols]

    print(f'X: {round(xs.nbytes / (10 ** 9), 3)} GB')
    print(f'Y: {round(ys.nbytes / (10 ** 9), 3)} GB')

    # vt = VarianceThreshold(threshold=0.1)
    # xs = vt.fit_transform(xs)

    dataset = split_dataset(xs, ys)
    model = train_model(dataset)

    val_xs, val_ys = gather_data(slice(cutoff, len(imgs)), dwt_level=dwt_level)
    # val_xs = vt.transform(val_xs)

    val_xs = val_xs[:, cols]

    validation_stacks = (val_xs, val_ys)
    # eval(model, validation_stacks)


if __name__ == '__main__':
    main()
