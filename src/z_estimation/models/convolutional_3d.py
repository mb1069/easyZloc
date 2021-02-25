import torch.nn as nn


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


# VGG19
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']


def get_model(out_channels=1):
    return nn.Sequential(
        *make_layers(cfg, batch_norm=True),
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, out_channels),
        nn.Sigmoid()
    )


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.model = get_model()

    def forward(self, x):
        return self.model(x)
