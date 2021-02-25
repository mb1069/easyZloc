from torch.utils.data import Dataset
import numpy as np

dtype = np.float32


class TaskDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]

        self.x = x.astype(dtype)
        self.y = y.astype(dtype)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]
