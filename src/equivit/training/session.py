import torch
import torch.nn as nn
from typing import Callable, Any
from pathlib import Path
import h5py
import yaml
import os


from ..registry import MODULE, OPTIMIZER, LOSS, DATASET

from ..version import __version__

from .hooks import TrainingHook, _dummy_training_hook

from .train import train



class TrainingSession:
    def __init__(self, config: dict, savedir: str = None):
        if 'savedir' in config.keys():
            assert savedir is None, 'savedir should be provided either in the config or as an argument, but not both'
            self.savedir = config['savedir']
        else:
            assert savedir is not None, 'savedir should be provided either in the config or as an argument'
            self.savedir = savedir

        self.config = config

        self.model = MODULE.build(config['model'])
        self.optimizer = OPTIMIZER.build(config['optimizer'])(self.model.parameters())
        self.loss_fn = LOSS.build(config['loss'])

        self.trainset = DATASET.build(config['trainset'])
        self.trainloader = torch.utils.data.DataLoader(self.trainset, **config['trainloader'])

        self.valset = DATASET.build(config['valset'])
        self.valloader = torch.utils.data.DataLoader(self.valset, **config['valloader'])

        # self.savedir = config['savedir']
        self.device = config['device']

        self.num_epochs = config['num_epochs']
        self.display_update_frequency = config['display_update_frequency']
        self.checkpoint_frequency = config['checkpoint_frequency']

        self.load_epoch = config.get('load_epoch', None)

        self.binary = config.get('binary', False)

        if type(self.loss_fn) in [nn.CrossEntropyLoss] and self.binary:
            raise ValueError('CrossEntropyLoss should not be used for binary classification. Please set binary to False or use a different loss function.')
        
        if type(self.loss_fn) in [nn.BCELoss, nn.BCEWithLogitsLoss] and not self.binary:
            raise ValueError('BCELoss and BCEWithLogitsLoss should be used for binary classification. Please set binary to True or use a different loss function.')

    def run(self, training_hook: TrainingHook = _dummy_training_hook):
        if self.load_epoch is None:
            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            (Path(self.savedir) / 'checkpoints').mkdir(exist_ok=False)


            with open((Path(self.savedir) / 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)


        train(
            savedir=self.savedir,
            device=self.device,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            trainloader=self.trainloader,
            valloader=self.valloader,
            N_epochs=self.num_epochs,
            load_epoch=self.load_epoch,

            # deterministic=self.deterministic,
            # seed=self.seed,
            checkpoint_frequency=self.checkpoint_frequency,
            display_update_frequency=self.display_update_frequency,
            binary=self.binary,
            training_hook=training_hook
        )



