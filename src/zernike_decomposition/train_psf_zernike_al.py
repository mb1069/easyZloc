from collections import OrderedDict
import os

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.pyOTF.pyotf.otf import apply_aberration, HanserPSF, name2noll
from src.config.optics import model_kwargs
from src.zernike_decomposition.gen_psf import NamedAbberationConfigGenerator, gen_dataset_named_params, \
    gen_psf_modelled_param
from src.data.data_manager import split_datasets, CustomDataset
import wandb
from tqdm import tqdm, trange
import numpy as np
import argparse
import torchsummary
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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
    {k: np.linspace(0, 1, 100) for k in list(name2noll.keys())[0:1]}
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


def prepare_val_loader(batch_size):
    X, y = gen_psf_dataset()
    dataset = CustomDataset(X, y)
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)
    return val_loader


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
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=115520, out_features=200),
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


def validate(model, val_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_cases = len(val_loader)
    coefs = []
    losses = []
    with torch.no_grad():
        for data in val_loader:
            X, y_true = data
            coefs.append(y_true)
            if USE_GPU:
                X = X.to('cuda:0')
                y_true = y_true.to('cuda:0')
            y_pred = model(X)
            case_loss = criterion(y_pred, y_true).detach().cpu().sum(axis=1)
            losses.append(case_loss)
            total_loss += float(case_loss.mean())

    coefs = np.concatenate(coefs).squeeze()
    losses = np.concatenate(losses).squeeze()
    # plt.scatter(coefs, losses)
    # plt.title(f'Val data {epoch}')
    # plt.show()
    return total_loss / total_cases


class ActiveDataset:
    ndim = 2
    min_p_val = 0
    max_p_val = 1
    n_datasets = 256
    learning_rate = 0.1

    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.n_batches = None
        self.pcoefs = None
        self.psfs = None
        self.init_dataset()

        self.set_num_batches()

    def init_dataset(self):
        print('Initialising dataset...')
        self.init_pcoefs()
        self.gen_psfs()
        print('Finished initialising dataset.')

    def init_pcoefs(self):
        self.pcoefs = torch.rand(size=(self.n_datasets, self.ndim))

    def gen_psfs(self):
        self.psfs = np.apply_along_axis(self.gen_psf, axis=1, arr=self.pcoefs)
        self.psfs = self.psfs[:, np.newaxis, :, :, :]
        self.psfs = torch.from_numpy(self.psfs).float()

    @staticmethod
    def gen_psf(pcoefs):
        psf = HanserPSF(**model_kwargs)
        mcoefs = np.zeros(shape=pcoefs.shape)
        psf = apply_aberration(psf, mcoefs, pcoefs)
        return psf.PSFi

    def __iter__(self):
        self.all_losses = []
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.n_batches:
            start = self.n * self.batch_size
            end = start + self.batch_size
            psfs = self.psfs[start:end]
            pcoefs = self.pcoefs[start:end]
            self.n += 1
            return psfs, pcoefs
        else:
            raise StopIteration

    def set_num_batches(self):
        self.n_batches = math.ceil(self.psfs.shape[0] / self.batch_size) - 1

    def __len__(self):
        return self.n_batches

    def report_loss(self, losses):
        self.all_losses.append(losses)

    def regen_data(self):
        self.all_losses = np.concatenate(self.all_losses)
        assert self.all_losses.shape[0] == self.psfs.shape[0]
        grad = np.gradient(self.all_losses.squeeze(), self.pcoefs.squeeze(), axis=(0))
        # plt.scatter(self.pcoefs, self.all_losses)
        # plt.scatter(self.pcoefs, grad)
        # plt.xlabel('Coefficient value')
        # plt.ylabel('Loss')
        # plt.show()
        diff = self.learning_rate * np.dot(self.all_losses, grad)

        new_pcoefs = self.pcoefs.squeeze() + diff


        # plt.scatter(self.pcoefs, new_pcoefs)
        # plt.xlabel('Original value')
        # plt.ylabel('New val')
        # plt.show()
        self.pcoefs = new_pcoefs[:, np.newaxis]

        self.all_losses = []
        self.n = 0
        self.gen_psfs()

    def summarise_pcoefs(self):
        return self.pcoefs.numpy().mean()


def main():
    args = parse_args()
    # wandb.init(project='smlm_zernike', name=args.rname)

    ds = ActiveDataset()
    model = ZernikeModel(ds.ndim)
    model = model.float()
    # torchsummary.summary(model, input_size=(1, 41, 32, 32), device='cpu')

    epochs = 100

    # train_loader, val_loader = prepare_dataloaders(batch_size, test_size)

    val_loader = prepare_val_loader(ds.batch_size)
    criterion = nn.MSELoss(reduction='none')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_GPU:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = criterion.to(device)
    print(f'Loaded model and criterion.')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    for epoch in trange(epochs, desc='epoch', position=0, leave=True):
        model.train()
        epoch_loss = 0
        for i, batch_data in enumerate(ds):
            psfs, coefs_true = batch_data
            if USE_GPU:
                psfs = psfs.to('cuda:0')
                coefs_true = coefs_true.to('cuda:0')
            optimizer.zero_grad()
            coefs_pred = model(psfs)
            case_loss = criterion(coefs_pred, coefs_true)

            loss = case_loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.cpu().detach())

            case_loss = case_loss.cpu().detach()
            ds.report_loss(case_loss)
        ds.regen_data()

        log = {
            'epoch': epoch,
            'train_loss': epoch_loss / ds.n_datasets
        }

        log['val_loss'] = validate(model, val_loader, criterion, epoch)
        # wandb.log(log)
        tqdm.write(
            f" Loss: {round(log['train_loss'], 4)}\t Val Loss: {round(log['val_loss'], 4)}, Pcoef: {round(ds.summarise_pcoefs(), 4)}")

        writer.add_scalar('training_loss', log['train_loss'], global_step=epoch)
        writer.add_scalar('val_loss', log['val_loss'], global_step=epoch)
        writer.add_histogram('pcoefs', ds.pcoefs.squeeze(), global_step=epoch)

        # scheduler.step(log['val_loss'])
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
