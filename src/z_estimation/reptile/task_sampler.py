import torch
from torch.utils.data import RandomSampler


class TaskSampler(RandomSampler):
    def __init__(self, data_source, num_samples):
        super().__init__(data_source, num_samples=num_samples)
        self.replacement = False
        print(data_source)
        self.max_idx = len(data_source)

    def __iter__(self):
        idx = torch.randint(high=self.max_idx, size=(1,))
        print(idx)
        task = self.data_source[idx[0]]
        task_shape = task.shape
        print(task_shape)
        repeated_task = task.repeat(self.num_samples).reshape(self.num_samples, *task_shape)
        print(repeated_task.shape)
        return None