import torch.nn as nn


def lin_relu(in_feat, out_feat):
    return nn.Sequential(
        nn.Linear(in_feat, out_feat),
        nn.ReLU()
    )


def get_model(im_size):
    return nn.Sequential(
        nn.Flatten(),
        lin_relu(im_size * im_size, 256),
        lin_relu(256, 128),
        lin_relu(128, 64),
        lin_relu(64, 32),
        nn.Linear(32, 1),
    )
