import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.data_manager import DataModule
import argparse

from src.z_estimation.models import convolutional

test_size = 0.1
epochs = 100
batch_size = 100

im_size = 32


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LitModel(pl.LightningModule):
    def __init__(self, im_size):
        super().__init__()
        self.model = self.make_model(im_size)
        self.loss = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []

        self.train_mse = pl.metrics.MeanSquaredError()
        self.val_mse = pl.metrics.MeanSquaredError()
        self.train_loss_label = 'train_loss'
        self.val_loss_label = 'val_loss'

    @staticmethod
    def make_model(im_size):
        return convolutional.SimpleConv(1)
        # return default.get_model(im_size)

    def forward(self, x):
        return self.model(x).squeeze(dim=1)

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
        return loss


    # def validation_step_end(self, outputs):
    #     # update and log
    #     self.val_mse(outputs['preds'], outputs['target'])
    #     self.log(self.val_loss_label, self.val_mse, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, verbose=True),
            'monitor': self.val_loss_label
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rname')
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        batch_size = 64
        epochs = 10
    dm = DataModule(batch_size, test_size=test_size, debug=args.debug)
    wandb_logger = WandbLogger(project='smlm_z', name=args.rname)

    model = LitModel(im_size=im_size)
    count_gpus = torch.cuda.device_count() if torch.cuda.is_available() else None
    backend = 'dp' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else None


    def get_early_stop(monitor='val_loss'):
        return EarlyStopping(
            monitor=monitor,
            min_delta=0.001,
            patience=50,
            verbose=True,
            mode='min',
        )


    dm.prepare_data()

    # trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger, callbacks=[get_early_stop(monitor=model.val_loss_label)])
    trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger)

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

    model_name = (args.rname or 'default') + '.pth'
    trainer.save_checkpoint(model_name)
    wandb_logger.experiment.save(__file__)
    wandb_logger.experiment.save(model_name)
