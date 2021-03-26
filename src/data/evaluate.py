import numpy as np


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2).sum()


def avg_rmse(y, y_pred):
    return np.sqrt(np.mean(((y.squeeze() - y_pred.squeeze()) ** 2)))
