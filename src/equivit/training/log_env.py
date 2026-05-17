import torch
from ..version import __version__ as equivit_version
import lightning as L

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.rank_zero import rank_zero_only
# from pytorch_lighting import loggers as pl_loggers_

import socket
import hashlib
import platform

from ._core import add_tags




class EnvironmentLoggerCallback(L.Callback):
    
    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module):

        # short hash for anonymity
        hostname = socket.gethostname()
        hostname_hash = hashlib.sha256(hostname.encode()).hexdigest()[:8] 

        env_info = {
            "env.pytorch_version": torch.__version__,
            "env.lightning_version": L.__version__,
            "env.cuda_available": torch.cuda.is_available(),
            "env.cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "env.equivit_version": equivit_version,
            "env.python_version": platform.python_version(),
            "hostname_hash": hostname_hash,
        }

        for logger in trainer.loggers:

            add_tags(logger, env_info)

