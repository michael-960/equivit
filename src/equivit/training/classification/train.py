import sys
from typing import Tuple
from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import lightning as L
import warnings

from lightning.pytorch import loggers as pl_loggers

from uuid import uuid4
from copy import deepcopy

from .model import ClassificationModel
from .data import ClassificationDataModule

from ..log_env import EnvironmentLoggerCallback

from .callbacks import LogModelSize

from .._core import add_tags, flatten_config, INTENTIONAL_FAIL_EXIT_CODE
from .._petnames import random_pet_name
from ..stop_big_model import ParameterBudgetExceededError


def generate_run_name(cfg: DictConfig) -> Tuple[str, str]:
    """
    Generate a (essentially) unique run name and shared run ID for the current training run.
    """
    shared_run_id = uuid4().hex[:12]
    shared_run_id_short = shared_run_id[:4]

    name_prefix = cfg.get("run_name", random_pet_name(words=2, separator="-"))


    run_name = f"{name_prefix}-{shared_run_id_short}"
    return run_name, shared_run_id_short, shared_run_id

def patch_logger_cfg(
        logger_cfg, *, 
        run_name: str, 
        shared_run_id_short: str,
        shared_run_id: str
) -> None:
    """
    Inject run name into a single logger config. 
    """
    target = logger_cfg.get("_target_")

    if target.startswith("pytorch_lightning"):
        raise ValueError("Detected logger with old PyTorch Lightning import path. Please update to the new lightning.pytorch loggers. For example, change 'pytorch_lightning.loggers.MLFlowLogger' to 'lightning.pytorch.loggers.MLFlowLogger'.")

    if target in ["lightning.pytorch.loggers.MLFlowLogger"]:
        OmegaConf.update(logger_cfg, "run_name", run_name, force_add=True)
        OmegaConf.update(logger_cfg, "tags.shared_run_id", shared_run_id, force_add=True)
        OmegaConf.update(logger_cfg, "tags.shared_run_id_short", shared_run_id_short, force_add=True)

    elif target in ["lightning.pytorch.loggers.WandbLogger"]:
        OmegaConf.update(logger_cfg, "name", run_name, force_add=True)
        OmegaConf.update(logger_cfg, "config.shared_run_id", shared_run_id, force_add=True)
        OmegaConf.update(logger_cfg, "config.shared_run_id_short", shared_run_id_short, force_add=True)

    elif target in ["lightning.pytorch.loggers.TensorBoardLogger"]:
        # TensorBoard logs to: save_dir / name / version
        OmegaConf.update(logger_cfg, "name", run_name, force_add=True)
        OmegaConf.update(logger_cfg, "version", shared_run_id, force_add=True) 
    
    else:
        warnings.warn(f"Logger of type {target} may not support run names or tags. Please verify that your logger is properly configured to include the run name and shared_run_id in its logs.")

def instantiate_loggers(cfg: DictConfig) -> list:
    """
    Instantiate loggers from config, after patching them with the generated run name and shared run ID.
    """
    run_name, shared_run_id_short, shared_run_id = generate_run_name(cfg)

    loggers_cfg = deepcopy(cfg.loggers)

    for logger_cfg in loggers_cfg.values():
        patch_logger_cfg(
            logger_cfg,
            run_name=run_name,
            shared_run_id_short=shared_run_id_short,
            shared_run_id=shared_run_id,
        )

    loggers = [
        instantiate(logger_cfg)
        for logger_cfg in loggers_cfg.values()
    ]

    return loggers, run_name, shared_run_id


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):


    module = ClassificationModel(
                model_cfg=cfg.model, 
                loss_fn_cfg=cfg.loss_fn, 
                optimizer_cfg=cfg.optimizer,
                binary=cfg.binary,
                compile=cfg.compile
            )

    data_module = ClassificationDataModule(
                    train_loader_cfg=cfg.data.train_loader,
                    val_loader_cfg=cfg.data.val_loader
                ) 


    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()] + [EnvironmentLoggerCallback(), LogModelSize()]
    # loggers = [instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()]
    loggers, _run_name, _shared_run_id = instantiate_loggers(cfg)


    # log model and data_module hyperparameters to all loggers
    for logger in loggers:
        logger.log_hyperparams(flatten_config(module.hparams))
        logger.log_hyperparams(flatten_config(data_module.hparams))


    # record hydra metadata
    hydra_out = Path(HydraConfig.get().runtime.output_dir)
    hydra_dir = hydra_out / ".hydra"


    if hydra_dir.exists():
        run_string = "/".join(hydra_out.parts[-2:])
        for logger in loggers:
            add_tags(logger, {"hydra_dir": run_string})
            if isinstance(logger, pl_loggers.MLFlowLogger):
                logger.experiment.log_artifact(logger.run_id, str(hydra_dir))
            elif isinstance(logger, pl_loggers.WandbLogger):
                import wandb
                artifact = wandb.Artifact(name=f'{logger.experiment.name}-{logger.experiment.id}-hydra-config',
                                          type='hydra-config')
                artifact.add_dir(str(hydra_dir))
                logger.experiment.log_artifact(artifact, aliases=["latest"])
            else:
                warnings.warn("Artifact logging is only implemented for MLFlowLogger and WandbLogger. Please add an MLFlowLogger or WandbLogger to your Trainer to log Hydra output artifacts.")

    # training

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    try:
        trainer.fit(module, datamodule=data_module)
    except ParameterBudgetExceededError as e:
        print(f"Training stopped: {e}")
        sys.exit(INTENTIONAL_FAIL_EXIT_CODE)


if __name__ == "__main__":
    main()
