import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L

from .model import ClassificationModel
from .data import ClassificationDataModule

from ..log_env import EnvironmentLoggerCallback



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


    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()] + [EnvironmentLoggerCallback()]
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()]

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
