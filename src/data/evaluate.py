import numpy as np


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2).sum()