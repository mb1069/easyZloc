import os
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.data_manager import load_custom_psfs
from src.z_estimation.models import convolutional, default
import wandb
from tqdm import tqdm, trange
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rname', default='debug')
    return parser.parse_args()


model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
USE_GPU = torch.cuda.is_available()


def prepare_dataloaders(batch_size, test_size):
    train_dataset, val_dataset = load_custom_psfs(test_size)
    # train_dataset, val_dataset = load_matlab_datasets(test_size, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def mse_to_unormalised(loss):
    return math.sqrt(loss) * 4100


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = convolutional.SimpleConv(1)
        self.model = convolutional.get_model()

    def forward(self, x):
        return self.model(x)

    def pred(self, X):
        X = torch.from_numpy(X).float()
        if USE_GPU:
            X = X.to('cuda:0')
        with torch.no_grad():
            return self(X).numpy()


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
    wandb.init(project='smlm_z', name=args.rname)
    wandb.save(__file__)
    model = Model()

    epochs = 100

    batch_size = 2500
    test_size = 0.1

    train_loader, val_loader = prepare_dataloaders(batch_size, test_size)

    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if USE_GPU:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    num_train = len(train_loader)
    for epoch in trange(epochs):
        model.train()
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            X, y_true = data
            if USE_GPU:
                X = X.to(device)
                y_true = y_true.to(device)
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.cpu().detach())

        log = {
            'epoch': epoch,
            'train_loss': mse_to_unormalised(epoch_loss / num_train)
        }

        log['val_loss'] = mse_to_unormalised(validate(model, val_loader, criterion))
        wandb.log(log)
        tqdm.write(f" Loss: {round(log['train_loss'], 4)}\t Val Loss: {round(log['val_loss'], 4)}")

        scheduler.step(log['val_loss'])

    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
