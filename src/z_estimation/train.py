import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
import numpy as np
import argparse
from pytorch_lightning.loggers.wandb import wandb

from src.data import data_manager
from src.z_estimation.models import convolutional, resnet

model_save_path = os.path.join(os.path.dirname(__file__), 'model.pth')

DEBUG = False

test_size = 0.1
epochs = 100
n_psfs = 10000

MODEL = 'resnet'

if MODEL == 'conv':
    BATCH_SIZE = 2048
    MODEL = convolutional.SimpleConv(1)
    LR = 0.01
else:
    BATCH_SIZE = 256
    MODEL = resnet.get_model()
    LR = 0.01

if DEBUG:
    epochs = 5
    BATCH_SIZE = 5
    n_psfs = 100


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MODEL
        self.loss = RMSELoss()
        self.training_losses = []
        self.validation_losses = []

        self.train_mse = pl.metrics.MeanSquaredError()
        self.val_mse = pl.metrics.MeanSquaredError()
        self.train_loss_label = 'train_loss'
        self.val_loss_label = 'val_loss'
        self.test_loss_label = 'test_loss'
        self.test_loss_label_mean = 'test_loss_mean'
        self.val_outputs = []

    def pred(self, X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return self(X).numpy()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(self.train_loss_label, loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(self.val_loss_label, loss, on_epoch=True, on_step=False)
        res = {
            'loss': torch.sqrt(torch.pow(y-y_hat, 2)),
            'pred': y_hat,
            'truth': y
        }
        res = {k: v.cpu().numpy().ravel() for k, v in res.items()}
        df = pd.DataFrame.from_dict(res)
        self.val_outputs.append(df)
        return loss

    def on_validation_epoch_end(self) -> None:
        df = pd.concat(self.val_outputs)
        bin_count = 17
        bins = np.linspace(-4000, 4000, bin_count).squeeze()
        bin_size = int(bins[1] - bins[0])
        groups = df.groupby(pd.cut(df['truth'], bins))
        log_data = {f'Range {g[0].left - (bin_size / 2)}-{g[0].right + (bin_size / 2)}': float(g[1]['loss'].mean()) for g in groups}
        self.log_dict(log_data)

        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=LR, weight_decay=5e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, verbose=True, patience=5),
            'monitor': self.val_loss_label
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rname')
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        BATCH_SIZE = 1
        epochs = 10
        n_psfs = 5

    print('Batch size', BATCH_SIZE)
    dm = data_manager.DataModule(BATCH_SIZE, test_size=test_size, debug=args.debug)
    wandb_logger = WandbLogger(project='smlm_z', name=args.rname)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = LitModel()
    count_gpus = torch.cuda.device_count() if torch.cuda.is_available() else None
    backend = 'dp' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else None


    def get_early_stop(monitor='val_loss'):
        return EarlyStopping(
            monitor=monitor,
            min_delta=0.001,
            patience=8,
            verbose=True,
            mode='min',
        )


    dm.prepare_data(n_psfs)
    dm.print_stats()

    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=count_gpus,
                         distributed_backend=backend,
                         logger=wandb_logger,
                         callbacks=[get_early_stop(monitor=model.val_loss_label), lr_monitor],
                         gradient_clip_val=0.5,
                         check_val_every_n_epoch=1)

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(model_save_path)
    print(f'Saved to {model_save_path}')
    wandb_logger.experiment.save(__file__)
    wandb_logger.experiment.save(data_manager.__file__)
    wandb_logger.experiment.save(model_save_path)
