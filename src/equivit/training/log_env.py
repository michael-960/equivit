import torch
from ..version import __version__ as equivit_version
import lightning as L

from lightning.pytorch import loggers as pl_loggers
# from pytorch_lighting import loggers as pl_loggers_

import socket
import hashlib
import platform

from ._core import add_tags




class EnvironmentLoggerCallback(L.Callback):
    def on_train_start(self, trainer: L.Trainer, pl_module):

        # short hash for anonymity
        hostname = socket.gethostname()
        hostname_hash = hashlib.sha256(hostname.encode()).hexdigest()[:8] 

        env_info = {
            "env/pytorch_version": torch.__version__,
            "env/lightning_version": L.__version__,
            "env/cuda_available": torch.cuda.is_available(),
            "env/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "env/equivit_version": equivit_version,
            "env/python_version": platform.python_version(),
            "hostname_hash": hostname_hash,
        }

        for logger in trainer.loggers:

            add_tags(logger, env_info)

            # if isinstance(logger, (pl_loggers.MLFlowLogger)):
            #     for key, value in env_info.items():
            #         logger.experiment.set_tag(logger.run_id, key, value)

            # elif isinstance(logger, pl_loggers.WandbLogger):

            #     new_tags = tuple([f"{key}:{value}" for key, value in env_info.items()])
            #     current_tags = logger.experiment.tags or ()

            #     logger.experiment.tags = current_tags + new_tags

            # elif isinstance(logger, pl_loggers.TensorBoardLogger):
            #     env_str = "\n".join([f"{key}: {value}" for key, value in env_info.items()])
            #     logger.experiment.add_text("Environment Info", env_str, global_step=0)

            # else:
            #     logger.log_hyperparams(env_info)
