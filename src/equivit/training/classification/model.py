from typing import Callable, Optional, Any
import torch.nn as nn
import torch
# import lightning as L
from lightning import LightningModule
from hydra.utils import instantiate
# import lightning.pytorch.callbacks as LC

from dataclasses import dataclass

from .confmat import ConfusionMatrixCalculator

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
            # self.model = torch.compile(self.model, backend='aot_eager')

        self.binary = binary

        self.loss_fn = instantiate(loss_fn_cfg)
        self.optimizer_factory = instantiate(optimizer_cfg)
        self.scheduler_factory = instantiate(scheduler_cfg) if scheduler_cfg is not None else None

        self.extra_cfg = extra_cfg # not used now 

        self.epoch_batch_count = 0
        self.epoch_total_loss = 0.

        self.save_hyperparameters()

        self.train_confmat_calculator = ConfusionMatrixCalculator()
        self.val_confmat_calculator = ConfusionMatrixCalculator()


        # count number of parameters
        num_params = 0
        num_params_trainable = 0
        for p in self.parameters():
            num_params += p.numel()
            if p.requires_grad:
                num_params_trainable += p.numel()

        self.val_best_loss = None


    def on_train_epoch_start(self):
        self.epoch_batch_count = 0
        self.epoch_total_loss = 0.
        self.train_confmat_calculator.reset_confusion_matrix()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x) # (B, num_classes) or (B, 1) for binary
        if self.binary:
            loss = self.loss_fn(logits.squeeze(-1), y.to(torch.float32))
            preds = (logits.detach().squeeze(-1) > 0).to(torch.int64) # (B,)
        else:
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits.detach(), dim=-1) # (B,)

        self.epoch_batch_count += 1
        self.epoch_total_loss += loss.detach().item()

        self.train_confmat_calculator.update(y, preds)

        # self.log('train_loss', loss, prog_bar=True)
        self.log('train/avg_loss', self.epoch_total_loss / self.epoch_batch_count, prog_bar=True)
        self.log('train/loss', loss.detach())

        return loss

    def on_validation_epoch_start(self):
        self.val_confmat_calculator.reset_confusion_matrix()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x) # (B, num_classes) or (B, 1) for binary
        if self.binary:
            loss = self.loss_fn(logits.squeeze(-1), y.to(torch.float32))
            preds = (logits.detach().squeeze(-1) > 0).long() # (B,)
        else:
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits.detach(), dim=-1) # (B,)

        self.val_confmat_calculator.update(y, preds)

        if self.val_best_loss is None or loss.detach().item() < self.val_best_loss:
            self.val_best_loss = loss.detach().item()

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/best_loss', self.val_best_loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return optimizer