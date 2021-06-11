import argparse
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics import MeanSquaredError

from src.data_manager import load_jonny_datasets
from src.models.convolutional import ConvolutionalModel
from src.reptile.data_module import ReptileDataModule, load_storm_datasets
from torchsummary.torchsummary import summary
from tqdm import trange

test_size = 0.1
epochs = 1000

im_size = 32

k = 32


def print_vars():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def get_free_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    print(f)


class ReptileTrainer:
    def __init__(self, model):
        self.model = model
        self.model.float()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = MeanSquaredError()

        if torch.cuda.is_available():
            self.model = self.model.to('cuda:0')
            self.criterion = self.criterion.to('cuda:0')

    def step_model(self, features, target):
        outputs = self.model(features).squeeze()
        loss = self.criterion(outputs, target)
        return loss

    def metastep_model(self, features, target, meta_step_size):
        weights_original = deepcopy(self.model.state_dict())
        new_weights = []
        for _ in range(k):
            loss = self.step_model(features, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            new_weights.append(deepcopy(model.state_dict()))
            model.load_state_dict({name: weights_original[name] for name in weights_original})

        ws = len(new_weights)
        fweights = {name: new_weights[0][name] / float(ws) for name in new_weights[0]}
        for i in range(1, ws):
            for name in new_weights[i]:
                fweights[name] += new_weights[i][name] / float(ws)
        model.load_state_dict(
            {name: weights_original[name] + ((fweights[name] - weights_original[name]) * meta_step_size) for name in
             weights_original})

    def train_model(self, meta_iters, train_dataset, val_dataset, wandb):
        num_tasks = len(train_dataset)
        self.run_eval(val_dataset)
        meta_step_size = 0.1
        for i in trange(meta_iters):
            frac_done = float(i) / meta_iters
            current_step_size = meta_step_size * (1. - frac_done)
            task = np.random.randint(0, num_tasks)

            task_input, task_target = train_dataset[task]
            task_target = np.asarray([task_target], dtype=np.float32)

            task_input = task_input[np.newaxis, :, :, :]

            task_input = torch.from_numpy(task_input)
            task_target = torch.from_numpy(task_target).squeeze()

            if torch.cuda.is_available():
                task_input = task_input.to('cuda:0')
                task_target = task_target.to('cuda:0')

            self.metastep_model(task_input, task_target, current_step_size)

            if i % 2 == 0:
                val_loss = self.run_eval(val_dataset)
                wandb.log({'val_loss': val_loss})

    def run_eval(self, val_dataset, batch_size=16):
        loss = 0
        num_batches = round((len(val_dataset) / batch_size) + 0.5)
        for i in range(num_batches):
            imgs, target = val_dataset[i * batch_size:(i + 1) * batch_size]
            imgs = torch.from_numpy(imgs)
            target = torch.from_numpy(target).squeeze()

            if torch.cuda.is_available():
                imgs = imgs.to('cuda:0')
                target = target.to('cuda:0')
            with torch.no_grad():
                loss += float(self.step_model(imgs, target).item())
        return loss



# class LitModel(pl.LightningModule):
#     def __init__(self, im_size):
#         super().__init__()
#         self.model = self.make_model(im_size)
#         self.loss = nn.MSELoss()
#         self.training_losses = []
#         self.validation_losses = []
#
#         self.train_mse = pl.metrics.MeanSquaredError()
#         self.val_mse = pl.metrics.MeanSquaredError()
#         self.train_loss_label = 'train_loss'
#         self.val_loss_label = 'val_loss'
#
#     @staticmethod
#     def make_model(im_size):
#         return convolutional.get_model()
#         # return default.get_model(im_size)
#
#     def forward(self, x):
#         return self.model(x).squeeze(dim=1)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log(self.train_loss_label, loss, on_epoch=True, on_step=False)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         return {'loss': loss, 'preds': y_hat, 'target': y}
#
#     def validation_step_end(self, outputs):
#         # update and log
#         self.val_mse(outputs['preds'], outputs['target'])
#         self.log(self.val_loss_label, self.val_mse, on_epoch=True, on_step=False)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': ReduceLROnPlateau(optimizer),
#             'monitor': self.val_loss_label
#         }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rname')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    # if args.debug:
    #     batch_size = 64
    #     epochs = 10

    model = ConvolutionalModel()

    train_dataset, val_dataset = load_jonny_datasets(test_size, datasets=[0])

    wandb.init(project='smlm_z', name=args.rname)

    wandb.watch(model)
    wandb.save(__file__)

    trainer = ReptileTrainer(model)

    trainer.train_model(epochs, train_dataset, val_dataset, wandb)

    #
    # dm = ReptileDataModule(batch_size, test_size=test_size, debug=args.debug, jonny_datasets=[0])
    # wandb_logger = WandbLogger(project='smlm_z', name=args.rname)
    #
    # model = LitModel(im_size=im_size)
    # count_gpus = torch.cuda.device_count() if torch.cuda.is_available() else None
    # backend = 'dp' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else None

    # def get_early_stop(monitor='val_loss'):
    #     return EarlyStopping(
    #         monitor=monitor,
    #         min_delta=0.05,
    #         patience=5,
    #         verbose=True,
    #         mode='min',
    #     )
    #
    #
    # # trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger, callbacks=[get_early_stop(monitor=model.val_loss_label)])
    # trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger)
    #
    # trainer.fit(model, dm)
    #
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

    # model_name = (args.rname or 'default') + '.pth'
    # trainer.save_checkpoint(model_name)
    # wandb_logger.experiment.save(__file__)
    # wandb_logger.experiment.save(model_name)
