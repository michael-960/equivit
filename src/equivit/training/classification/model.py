from typing import Callable, Optional, Any
import torch.nn as nn
import torch
# import lightning as L
from lightning import LightningModule
from hydra.utils import instantiate
# import lightning.pytorch.callbacks as LC

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    ...


class ClassificationModel(LightningModule):
    """
    A LightningModule for classification tasks.
    """
    def __init__(self,
        model_cfg,
        loss_fn_cfg,
        optimizer_cfg,
        compile: bool = False,
        binary: bool = False,
        scheduler_cfg=None,
        extra_cfg=None
    ):
        super().__init__()

        self.model = instantiate(model_cfg, _convert_='all')
        if compile:
            self.model = torch.compile(self.model)

        self.binary = binary

        self.loss_fn = instantiate(loss_fn_cfg)
        self.optimizer_factory = instantiate(optimizer_cfg)
        self.scheduler_factory = instantiate(scheduler_cfg) if scheduler_cfg is not None else None

        self.extra_cfg = extra_cfg # not used now 

        self.epoch_batch_count = 0
        self.epoch_total_loss = 0.

        self.save_hyperparameters()

        # self.save_hyperparameters({
        #     'model': model_cfg,
        #     'compile': compile,
        #     'loss_fn': loss_fn_cfg,
        #     'optimizer': optimizer_cfg,
        #     'scheduler': scheduler_cfg,
        #     'extra': extra_cfg
        # })
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        if self.binary:
            loss = self.loss_fn(logits.squeeze(-1), y.to(torch.float32))
        else:
            loss = self.loss_fn(logits, y)

        self.epoch_batch_count += 1
        self.epoch_total_loss += loss.detach().item()
        # self.log('train_loss', loss, prog_bar=True)
        self.log('avg_train_loss', self.epoch_total_loss / self.epoch_batch_count, prog_bar=True)
        self.log('train_loss', loss.detach())
        return loss

    def on_train_epoch_start(self):
        self.epoch_batch_count = 0
        self.epoch_total_loss = 0.

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        if self.binary:
            loss = self.loss_fn(logits.squeeze(-1), y.to(torch.float32))
        else:
            loss = self.loss_fn(logits, y)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return optimizer