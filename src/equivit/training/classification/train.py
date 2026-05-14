from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import lightning as L

from lightning.pytorch import loggers as pl_loggers

from .model import ClassificationModel
from .data import ClassificationDataModule

from ..log_env import EnvironmentLoggerCallback

from .callbacks import LogModelSize

from .._core import add_tags



@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):


    module = ClassificationModel(
                model_cfg=cfg.model, 
                loss_fn_cfg=cfg.loss_fn, 
                optimizer_cfg=cfg.optimizer,
                compile=cfg.compile
            )

    data_module = ClassificationDataModule(
                    train_loader_cfg=cfg.data.train_loader,
                    val_loader_cfg=cfg.data.val_loader
                ) 


    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()] + [EnvironmentLoggerCallback(), LogModelSize()]
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()]


    # record hydra metadata
    hydra_out = Path(HydraConfig.get().runtime.output_dir)
    hydra_dir = hydra_out / ".hydra"


    if hydra_dir.exists():
        run_string = "/".join(hydra_out.parts[-2:])
        for logger in loggers:
            add_tags(logger, {"hydra_dir": run_string})
            if isinstance(logger, pl_loggers.MLFlowLogger):
                logger.experiment.log_artifact(logger.run_id, str(hydra_dir), artifact_path="hydra")
            else:
                raise NotImplementedError("Logging hydra metadata is currently only implemented for MLFlowLogger. Please implement for other loggers if needed.")

    # training

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
