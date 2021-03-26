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

from src.data.data_manager import DataModule

from src.z_estimation.models import convolutional

model_save_path = os.path.join(os.path.dirname(__file__), 'model.pth')

DEBUG = False

test_size = 0.1
epochs = 100
batch_size = 2048
n_psfs = 10000

if DEBUG:
    epochs = 5
    batch_size = 5
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
        self.model = convolutional.get_model()
        # self.model = convolutional.SimpleConv(1)
        self.loss = RMSELoss()
        self.training_losses = []
        self.validation_losses = []

        self.train_mse = pl.metrics.MeanSquaredError()
        self.val_mse = pl.metrics.MeanSquaredError()
        self.train_loss_label = 'train_loss'
        self.val_loss_label = 'val_loss'
        self.test_loss_label = 'test_loss'
        self.test_loss_label_mean = 'test_loss_mean'

    def pred(self, X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return self(X).numpy()

    def forward(self, x):
        return self.model(x)

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
        return {
            'loss': loss,
            'pred': y_hat,
            'truth': y
        }

    def validation_epoch_end(self, outputs):
        cols = {k: [float(v.squeeze().cpu().numpy())] for k, v in outputs[0].items()}
        for bp in outputs[1:]:
            for k, v in bp.items():
                cols[k].append(float(v.cpu().numpy()))
        df = pd.DataFrame.from_dict(cols)
        bins = np.linspace(0, df['truth'].max(), 20)
        groups = df.groupby(['truth', pd.cut(df.loss, bins)])
        data_x = []
        data_y = []
        for middle, gdf in groups:
            data_x.append(middle[0])
            data_y.append(gdf.loss.mean())

        data = [[x, y] for (x, y) in zip(data_x, data_y)]
        table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log({"my_custom_plot_id": wandb.plot.line(table, "x", "y",
                                                        title="Variation in loss over axial positions")})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=5e-4)
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
        batch_size = 1
        epochs = 10
        n_psfs = 5

    print('Batch size', batch_size)
    dm = DataModule(batch_size, test_size=test_size, debug=args.debug)
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

    # trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger, callbacks=[get_early_stop(monitor=model.val_loss_label)])
    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=count_gpus,
                         distributed_backend=backend,
                         logger=wandb_logger,
                         callbacks=[get_early_stop(monitor=model.val_loss_label), lr_monitor],
                         gradient_clip_val=0.5)

    trainer.fit(model, datamodule=dm)

    # dm2 = ReptileDataModule(1, test_size=test_size, debug=args.debug, jonny_datasets=[0])
    # model.train_loss_label = 'train2_loss'
    # model.val_loss_label = 'val2_loss'
    # trainer2 = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger,
    #                      callbacks=[get_early_stop(monitor=model.val_loss_label)])
    #
    # trainer2.fit(model, dm2)
    #
    #
    # print('Training batch 3')
    # model2 = LitModel(im_size=im_size)
    # model2.train_loss_label = 'train3_loss'
    # model2.val_loss_label = 'val3_loss'
    # trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger,
    #                      callbacks=[get_early_stop(monitor=model2.val_loss_label)])
    # trainer.fit(model2, dm2)

    trainer.save_checkpoint(model_save_path)
    print(f'Saved to {model_save_path}')
    wandb_logger.experiment.save(__file__)
    wandb_logger.experiment.save(model_save_path)
