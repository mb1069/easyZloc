from scipy.sparse import data
from torch.utils.data.dataloader import DataLoader
from data.visualise import scatter_3d
import os

import matplotlib.pyplot as plt
import numpy as np

from data.datasets import TrainingDataSet, ExperimentalDataSet
from config.datafiles import res_file
from config.datasets import dataset_configs
from workflow_v2 import eval_model
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.models import resnet, resnet34
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules import Conv2d
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset


DEBUG = False

BATCH_SIZE = 4096
USE_GPU = True
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src/wavelets/wavelet_data/output')

model_path = os.path.join(os.path.dirname(__file__), 'tmp/model.json')


model_path = os.path.join(os.path.dirname(__file__), 'model.h5')

checkpoint_path =  os.path.join(os.path.dirname(__file__), 'chkp')


class FullDataset(Dataset):
    """Basic class encapulating dataset functions, for use with dataloader."""

    def __init__(self, img_numpy_array: np.ndarray, labels_numpy_array: np.ndarray) -> None:
        """Initialise with numpy array."""
        self.img_data = img_numpy_array
        self.labels = torch.from_numpy(labels_numpy_array)
        print("Array shape loaded into torch dataset:", self.img_data.shape)

    def __len__(self) -> int:
        """Return length of dataset."""
        assert self.img_data.shape[0] == self.labels.shape[0]
        return self.img_data.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        """Return items from dataset as torch tensors."""
        img_data_selection = self.img_data[idx]
        img_tensor = torch.from_numpy(img_data_selection)
        return (img_tensor, self.labels[idx])

class Model(LightningModule):
    def __init__(self, lr=0.001) -> None:
        super().__init__()
        self.model = self.load_model()
        self.save_hyperparameters()

    def load_model(self):
        model = resnet34(pretrained=False, num_classes=1)
        model.conv1 = Conv2d(1, 64, (3,3))
        return model
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).squeeze()
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self(x).squeeze()
        loss = F.mse_loss(pred, y)
        acc = float(torch.abs(pred - y).mean())

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
        # steps_per_epoch = self.train_dataset_len // BATCH_SIZE
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=25, verbose=False)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


if __name__ == '__main__':
    z_range = 1000

    dataset = 'paired_bead_stacks'

    train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=False)

    # exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False)

    for t in ['train', 'val']:
        train_dataset.data[t] = list(train_dataset.data[t])
        train_dataset.data[t][0] = np.moveaxis(train_dataset.data[t][0], 3, 1).astype(np.float32)
        train_dataset.data[t][1] = train_dataset.data[t][1].astype(np.float32)

    train_dataloader = DataLoader(FullDataset(*train_dataset.data['train']), batch_size=BATCH_SIZE, num_workers=8)
    val_dataloader = DataLoader(FullDataset(*train_dataset.data['val']), batch_size=BATCH_SIZE, num_workers=8)

    model = Model()
    model = model.float()

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1, patience=50, verbose=True, mode="min")
    ]
    trainer = Trainer(
        progress_bar_refresh_rate=1,
        log_every_n_steps=2,
        max_epochs=1000,
        gpus=1,
        logger=TensorBoardLogger("lightning_logs/", name="resnet"),
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
