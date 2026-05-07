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

        self.train_loader = instantiate(train_loader_cfg)
        self.val_loader = instantiate(val_loader_cfg)

        self.save_hyperparameters({
            'train_loader': train_loader_cfg,
            'val_loader': val_loader_cfg
        })
        
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
