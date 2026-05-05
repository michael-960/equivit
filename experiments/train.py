import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L

import equivit
from equivit.lightning import ClassificationModel, ClassificationDataModule
import torch


class EnvironmentLoggerCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.logger:
            env_info = {
                "env/pytorch_version": torch.__version__,
                "env/lightning_version": L.__version__,
                "env/cuda_available": torch.cuda.is_available(),
                "env/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "env/equivit_version": equivit.__version__,
            }
            # Log the info using the trainer's logger
            trainer.logger.log_hyperparams(env_info)


@hydra.main(version_base=None, config_path="conf", config_name="config")
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
