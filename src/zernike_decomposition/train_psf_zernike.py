from collections import OrderedDict
import os

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchsummary import torchsummary

from src.zernike_decomposition.gen_psf import NamedAbberationConfigGenerator, gen_dataset_named_params
from src.data.data_manager import split_datasets
from src.z_estimation.models import convolutional_3d
import wandb
from tqdm import tqdm, trange
import numpy as np
import argparse
from pyotf.otf import name2noll

model_path = os.path.join(os.path.dirname(__file__), 'zernike.pth')

# Pseudo hashing function
def string_to_int(s):
    ord3 = lambda x: '%.3d' % ord(x)
    return int(''.join(map(ord3, s)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rname', default='debug')
    return parser.parse_args()


USE_GPU = True

if not torch.cuda.is_available():
    USE_GPU = False


# psf_param_config = OrderedDict({
#     'oblique astigmatism': np.linspace(0, 1, 20),
#     'defocus': np.linspace(0, 1, 20),
#     'vertical astigmatism': np.linspace(0, 1, 20),
#     'tilt': np.linspace(0, 1, 20),
# })

psf_param_config = OrderedDict(
    {k: np.linspace(0, 1, 10) for k in list(name2noll.keys())[0:6]}
)


n_zernike = len(psf_param_config.keys())


def get_config_dataset_file(cfg):
    cfg_str = '_'.join([f'{k}-{v.min()}-{len(v)}-{v.max()}' for k, v in cfg.items()])
    # tmp = str(cfg_str)
    # h = str(string_to_int(tmp) % 25253 ** 2)
    config_dir = os.path.join(os.path.dirname(__file__), 'psfs')
    # os.makedirs(config_dir, exist_ok=True)
    dpath = os.path.join(config_dir, f'{cfg_str}.npz')
    return dpath


def check_existing_dataset(cfg):
    dpath = get_config_dataset_file(cfg)
    if os.path.exists(dpath):
        print('Loading dataset')
        dataset = np.load(dpath)
        print(os.path.basename(dpath))
        return dataset['psfs'], dataset['configs']
    return None


def save_dataset(cfg, psfs, configs):
    print('Saving dataset')
    dpath = get_config_dataset_file(cfg)
    dirname = os.path.dirname(dpath)
    os.makedirs(dirname, exist_ok=True)
    np.savez(dpath, psfs=psfs, configs=configs)
    print('Saved to')
    print(dpath)


def gen_psf_dataset():
    existing_dataset = check_existing_dataset(psf_param_config)
    if existing_dataset:
        return existing_dataset

    cg = NamedAbberationConfigGenerator(psf_param_config)
    psfs, _ = gen_dataset_named_params(cg)
    psfs = np.stack(psfs, axis=0)
    cg.reset()

    psfs = psfs[:, np.newaxis, :, :, :]
    configs = list(cg.config_iter)
    configs = np.stack(configs, axis=0)

    save_dataset(psf_param_config, psfs, configs)

    return psfs, configs


def prepare_dataloaders(batch_size, test_size):
    X, y = gen_psf_dataset()
    train_dataset, val_dataset = split_datasets(X, y, test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

    return train_loader, val_loader


class ZernikeModel(nn.Module):
    def __init__(self, num_zernike):
        super().__init__()
        # self.model = convolutional_3d.get_model(n_zernike)
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(True),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=16),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=81920, out_features=200),
            nn.Dropout(),
            nn.Linear(in_features=200, out_features=num_zernike),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def pred(self, X):
        X = torch.from_numpy(X).float()
        if USE_GPU:
            X = X.to('cuda:0')
        with torch.no_grad():
            return self(X).numpy()

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    total_cases = len(val_loader)
    with torch.no_grad():
        for data in val_loader:
            X, y_true = data
            if USE_GPU:
                X = X.to('cuda:0')
                y_true = y_true.to('cuda:0')
            y_pred = model(X)
            loss = criterion(y_pred, y_true).detach()

            total_loss += float(loss)
    return total_loss / total_cases


def main():
    args = parse_args()
    wandb.init(project='smlm_zernike', name=args.rname)

    model = ZernikeModel(n_zernike)
    # torchsummary.summary(model, input_size=(1, 41, 32, 32), device='cpu')

    epochs = 20

    batch_size = 2048
    test_size = 0.1

    train_loader, val_loader = prepare_dataloaders(batch_size, test_size)

    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if USE_GPU:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    num_train = len(train_loader)
    for epoch in trange(epochs, desc='epoch'):
        model.train()
        epoch_loss = 0
        for i, data in enumerate(tqdm(train_loader, desc='batch', total=len(train_loader)), 0):
            X, y_true = data
            if USE_GPU:
                X = X.to('cuda:0')
                y_true = y_true.to('cuda:0')
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.cpu().detach())

        log = {
            'epoch': epoch,
            'train_loss': epoch_loss / num_train
        }

        log['val_loss'] = validate(model, val_loader, criterion)
        wandb.log(log)
        tqdm.write(f" Loss: {round(log['train_loss'], 4)}\t Val Loss: {round(log['val_loss'], 4)}")

        scheduler.step(log['val_loss'])
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
