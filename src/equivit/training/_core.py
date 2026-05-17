from typing import Dict, Any
from ..version import __version__ as equivit_version
import lightning as L
import warnings

from lightning.pytorch import loggers as pl_loggers



def add_tags(logger, tags: Dict[str, Any]):
    """
    Add tags to a logger in a way that is compatible with different logger types.
    """

    if isinstance(logger, (pl_loggers.MLFlowLogger)):
        for key, value in tags.items():
            logger.experiment.set_tag(logger.run_id, key, value)

    elif isinstance(logger, pl_loggers.WandbLogger):

        # new_tags = tuple([f"{key}:{value}" for key, value in tags.items()])
        # current_tags = logger.experiment.tags or ()

        # logger.experiment.tags = current_tags + new_tags
        logger.experiment.config.update(tags)

    elif isinstance(logger, pl_loggers.TensorBoardLogger):
        # env_str = "\n".join([f"{key}:{value}" for key, value in tags.items()])
        # logger.experiment.add_text("Environment Info", env_str, global_step=0)
        warnings.warn("Adding tags is not implemented for TensorBoardLogger. Consider using MLFlowLogger or WandbLogger for this feature.")
        logger.log_hyperparams(tags) # temporary solution
    else:
        logger.log_hyperparams(tags)



from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def to_plain_container(x: Any) -> Any:
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True, enum_to_str=True)
    return x


# import json


def flatten_config(
    x: Any,
    *,
    parent_key: str = "",
    sep: str = ".",
    flatten_list_of_dicts: bool = True,
) -> dict[str, Any]:
    x = to_plain_container(x)

    if isinstance(x, Mapping):
        out: dict[str, Any] = {}
        for key, value in x.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            out.update(
                flatten_config(
                    value,
                    parent_key=new_key,
                    sep=sep,
                    flatten_list_of_dicts=flatten_list_of_dicts,
                )
            )
        return out

    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        if flatten_list_of_dicts and all(isinstance(v, Mapping) for v in x):
            out: dict[str, Any] = {}
            for i, value in enumerate(x):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                out.update(
                    flatten_config(
                        value,
                        parent_key=new_key,
                        sep=sep,
                        flatten_list_of_dicts=flatten_list_of_dicts,
                    )
                )
            return out

        return {parent_key: repr(list(x))}

    return {parent_key: x}