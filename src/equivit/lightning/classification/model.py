from typing import Callable, Optional, Any
import torch.nn as nn
import torch
import lightning as L
import lightning.pytorch.callbacks as LC

from dataclasses import dataclass



class TrainingConfig(dataclass):
    ...



class ClassificationModel(L.LightningModule):
    """
    A LightningModule for classification tasks.
    """
    def __init__(self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer_factory: Callable[[Any], torch.optim.Optimizer],
        scheduler_factory: Optional[Callable] = None,
        config: Optional[TrainingConfig] = None,
    ):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.config = config
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return optimizer