import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import math


def split_dataset(X, y):
    train_size = 0.7
    val_size = 0.2
    test_size = 0.1
    assert math.isclose(train_size + val_size + test_size, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size / (train_size + val_size),
                                                      shuffle=True)

    train_dataset = [X_train, y_train]
    test_dataset = [X_test, y_test]
    val_dataset = [X_val, y_val]

    return train_dataset, test_dataset, val_dataset


def mae_dataset(model, X, y):
    y_pred = model.predict(X)
    training_mae = mean_absolute_error(y_pred, y)
    return training_mae


def disp_jonny_datasets(model, dataset):
    X, y_true = dataset

    # X, y_true = process_jonny_datadir(jonny_data_dir, datasets=list(range(20)), bound=bounds)

    # axial_max = X.max(axis=(1, 2))
    # X = X / axial_max[:, None, None]
    # X = min_max_norm(X)

    X = X.squeeze()
    _true = y_true.squeeze()

    y_pred = model.predict(X).squeeze()

    training_mae = mean_absolute_error(y_pred, y_true)
    print(f'Train mae: {round(training_mae, 3)}')
    # y_pred = y_pred / (y_pred.max())
    # y_true = y_true / (y_true.max())

    # zero = min(y_pred.min(), y_true.min())
    #
    # y_pred -= zero
    # y_true -= zero

    plt.scatter(y_true, y_pred, marker='x')
    fit_line = np.linspace(y_true.min(), y_true.max(), 100)
    plt.plot(fit_line, fit_line, label='y=x', color='red')

    # plt.xlabel('True axial position (nm)')
    # plt.ylabel('Predicted axial position (nm)')
    # plt.title(f'Mean absolute error: {round(mean_absolute_error(y_pred, y_true), 3)}')
    # # plt.xlim((y_true.min(), y_true.max()))
    # # plt.ylim((y_mse.min(), y_mse.max()))
    # plt.show()
    # Center means
    # eval_results(y_true, y_pred)
    # plt.hist(y_true)
    # plt.show()
    #
    # plt.hist(y_pred)
    # plt.show()
    #
    # plt.hist(y_true - y_pred)
    # plt.show()
    return training_mae


def view_dataset(df):
    df = df.sample(n=10000)
    print(f'{df.shape[0]} samples')
    x = df['x [nm]'].to_numpy()
    y = df['y [nm]'].to_numpy()
    z = df['z [nm]'].to_numpy()

    plt.scatter(x, y, marker='x')
    plt.show()

    fig = plt.figure()
    plt.set_cmap('afmhot')

    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, marker='o', s=3, c=z)
    plt.show()


def optimise_model(train_dataset):
    model = XGBRegressor(tree_method='gpu_hist')
    param_grid = {
        'n_estimators': list(map(int, np.linspace(100, 1000, 5))),
        'max_depth': [6, 7, 8],
        'min_child_weight': [0, 1, 2, 3],
        'gamma': [0, 1, 2, 3],
        'max_bin': [128, 256, 512],
    }

    from sklearn.metrics import make_scorer
    # define your own mse and set greater_is_better=False
    mse = make_scorer(mean_squared_error, greater_is_better=False)

    gsh = HalvingGridSearchCV(estimator=model, param_grid=param_grid, factor=2, verbose=2, cv=3, n_jobs=3, scoring=mse)
    gsh.fit(*train_dataset)
    print(gsh.best_params_)
