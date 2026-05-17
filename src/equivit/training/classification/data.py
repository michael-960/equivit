from typing import Dict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate

class ClassificationDataModule(LightningDataModule):
    def __init__(
        self, 
        train_loader_cfg,
        val_loader_cfg,
    ):
        super().__init__()
        
        # self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])

        self.train_loader: DataLoader = instantiate(train_loader_cfg)
        self.val_loader: DataLoader = instantiate(val_loader_cfg)

        _hparams = {
            'train_loader': train_loader_cfg,
            'val_loader': val_loader_cfg
        }

        _hparams.update(get_augmentation_info(self.train_loader, split="train"))
        _hparams.update(get_augmentation_info(self.val_loader, split="val"))

        self.save_hyperparameters(_hparams, logger=False)

        
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader



def get_augmentation_info(data_loader: DataLoader, split: str) -> Dict[str, object]:
    transform = getattr(data_loader.dataset, "transform", None)

    has_augmentation = getattr(transform, "has_augmentation", "unknown")
    num_augmentation_steps = getattr(transform, "num_augmentation_steps", "unknown")
    num_preprocessing_steps = getattr(transform, "num_preprocessing_steps", "unknown")

    return {
        f"data.{split}.augmentation.enabled": has_augmentation,
        f"data.{split}.augmentation.num_steps": num_augmentation_steps,
        f"data.{split}.preprocessing.num_steps": num_preprocessing_steps,
    }