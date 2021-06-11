import torch.nn as nn
import torch


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


# VGG19
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']


def get_model():
    model = nn.Sequential(
        *make_layers(cfg, batch_norm=True),
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1),
        # nn.Hardswish()
    )
    return model


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.model = get_model()

    def forward(self, x):
        return self.model(x).squeeze(dim=1)


class SimpleConv(nn.Module):
    def __init__(self, num_zernike):
        super().__init__()
        # self.model = convolutional_3d.get_model(n_zernike)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=6400, out_features=200),
            nn.Dropout(),
            nn.Linear(in_features=200, out_features=num_zernike),
        )

    def forward(self, x):
        return self.model(x)

    def pred(self, X):
        X = torch.from_numpy(X).float()
        if True:
            X = X.to('cuda:0')
        with torch.no_grad():
            return self(X).numpy()

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)