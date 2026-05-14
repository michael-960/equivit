from typing import Dict, Any
from ..version import __version__ as equivit_version
import lightning as L

from lightning.pytorch import loggers as pl_loggers



def add_tags(logger, tags: Dict[str, Any]):

    if isinstance(logger, (pl_loggers.MLFlowLogger)):
        for key, value in tags.items():
            logger.experiment.set_tag(logger.run_id, key, value)

    elif isinstance(logger, pl_loggers.WandbLogger):

        new_tags = tuple([f"{key}:{value}" for key, value in tags.items()])
        current_tags = logger.experiment.tags or ()

        logger.experiment.tags = current_tags + new_tags

    elif isinstance(logger, pl_loggers.TensorBoardLogger):
        env_str = "\n".join([f"{key}: {value}" for key, value in tags.items()])
        logger.experiment.add_text("Environment Info", env_str, global_step=0)

    else:
        logger.log_hyperparams(tags)
