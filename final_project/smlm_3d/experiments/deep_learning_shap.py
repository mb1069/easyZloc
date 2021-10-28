import torch
from torch import nn
import torchvision
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from torch.utils.data import Dataset

SEED = 42
BATCH_SIZE=1000

coords_error = pd.read_csv('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/tmp/coords.csv')
imgs = np.load('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/tmp/imgs.npy')
coords_error['failure'] = coords_error['error'] > 70
good_images_idx = np.where(coords_error['failure'])[0]
bad_images_idx = np.where(~coords_error['failure'])[0]
good_images = imgs[good_images_idx]
bad_images = imgs[bad_images_idx]

def create_model(num_classes):
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

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
        img_tensor = torch.from_numpy(img_data_selection).float()
        return (img_tensor, self.labels[idx])


class LitResnet(LightningModule):
    def __init__(self, lr=0.05, train_dataset_len=640, num_classes=2):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(num_classes)
        
        self.train_dataset_len = train_dataset_len

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).float()

        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        print(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.argmax(logits, dim=1)
        micro_acc = accuracy(preds, y, average="micro")
        macro_acc = accuracy(preds, y, average="macro", num_classes=2)

        return {
            "loss": loss.cpu(), 
            "micro_acc": micro_acc.cpu(),
            "macro_acc": macro_acc.cpu(),
            "preds": preds.cpu(),
            "labels": y.cpu()
        }
    
    def evaluation_epoch_end(self, outputs, stage=None):
        micro_acc = torch.stack([tmp[f"micro_acc"] for tmp in outputs]).mean()
        macro_acc = torch.stack([tmp[f"macro_acc"] for tmp in outputs]).mean()
        loss = torch.stack([tmp[f"loss"] for tmp in outputs]).mean()
        self.log_dict({f"{stage}_micro_acc": micro_acc, f"{stage}_macro_acc": macro_acc, f"{stage}_loss": loss}, logger=True, prog_bar=True)
        
        # preds = torch.cat([tmp[f"preds"] for tmp in outputs])
        # labels = torch.cat([tmp[f"labels"] for tmp in outputs])
        
        # conf_mat = confusion_matrix(preds, labels, num_classes=NUM_CLASSES)
        
        # np.save("/camp/project/proj-data-challenge/2021/Project1-El-Oakley/scripts/models/run-2/preds.npy", preds.detach().numpy())
        # np.save("/camp/project/proj-data-challenge/2021/Project1-El-Oakley/scripts/models/run-2/labels.npy", labels.detach().numpy())
        
        # np.save("/camp/project/proj-data-challenge/2021/Project1-El-Oakley/scripts/models/run-2/conf_mat.npy", conf_mat.numpy())
        
        

    def validation_step(self, batch, batch_idx):
        metrics = self.evaluate(batch, "val")
        return metrics
    
    def validation_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        metrics = self.evaluate(batch, "test")
        return metrics
    
    def test_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
        return [optimizer], [scheduler]

all_images = np.concatenate((good_images, bad_images))
all_images = np.moveaxis(all_images, 3, 1).astype(np.float32)

labels = np.concatenate((np.ones((good_images.shape[0])), np.zeros((bad_images.shape[0]))))

labels = labels.astype(np.float32)


train_x, val_x, train_y, val_y = train_test_split(all_images, labels, test_size=0.3, stratify=labels, shuffle=True, random_state=SEED) 

train_dataset = FullDataset(train_x, train_y)
val_dataset = FullDataset(val_x, val_y)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

early_stop_callback = EarlyStopping(monitor='val_macro_acc', patience=6, mode='max')

trainer = Trainer(
    progress_bar_refresh_rate=1,
    log_every_n_steps=1,
    max_epochs=30,
    gpus=-1,
    accelerator='dp',
    callbacks=[LearningRateMonitor(logging_interval="step"), early_stop_callback],
    checkpoint_callback=True,
)

model = LitResnet()
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
