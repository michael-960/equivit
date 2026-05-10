import torch
from lightning.pytorch.callbacks import Callback

from typing import List, Dict, Any, Union, TYPE_CHECKING

import seaborn as sns
import matplotlib.pyplot as plt

from lightning.pytorch import loggers as pl_loggers

from .metrics import ClassificationMetric

if TYPE_CHECKING:
    from .model import ClassificationModel
    import lightning as L



class LogMetrics(Callback):
    """
    A Callback that logs the confusion matrix at the end of each training and validation epoch
    as well as any additional metrics specified in the `metrics` dictionary. 

    Warning: currently, only MLFLOW is tested.
    """
    def __init__(
        self, metrics: Dict[str, ClassificationMetric]
    ):
        self.metrics = metrics

        # we assume higher is better for all metrics

        self._best_train_metrics = {name: None for name in metrics.keys()}
        self._best_val_metrics = {name: None for name in metrics.keys()}

        self._best_metrics = {
            'train': {name: None for name in metrics.keys()},
            'val': {name: None for name in metrics.keys()}
        }


    def _on_epoch_end(self, trainer: "L.Trainer", pl_module: "ClassificationModel", stage: str):
        assert stage in ["train", "val"], "Stage must be either 'train' or 'val'"
        if stage == "train":
            confmat = pl_module.train_confmat_calculator.pop_confusion_matrix()
            stage_name = "Training"
        else:
            confmat = pl_module.val_confmat_calculator.pop_confusion_matrix()
            stage_name = "Validation"

        fig, ax = plt.subplots()
        sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{stage_name} Confusion Matrix - Epoch {trainer.current_epoch}")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("GT Labels")

        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                logger.experiment.add_figure(tag=f"{stage_name} Confusion Matrix", figure=fig, global_step=trainer.current_epoch)

            elif isinstance(logger, pl_loggers.MLFlowLogger):
                logger.experiment.log_figure(run_id=logger.run_id, figure=fig, artifact_file=f"{stage}_confmat_epoch_{trainer.current_epoch}.png")

            elif isinstance(logger, pl_loggers.WandbLogger):
                try:
                    import wandb
                    logger.experiment.log({f"{stage_name} Confusion Matrix": wandb.Image(fig), "epoch": trainer.current_epoch})
                except ImportError:
                    raise ImportError("WandbLogger requires the wandb library. Please install it with `pip install wandb`.")


        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(confmat)
            pl_module.log(f"{stage}/{metric_name}", metric_value)

            if self._best_metrics[stage][metric_name] is None or metric_value > self._best_metrics[stage][metric_name]:
                self._best_metrics[stage][metric_name] = metric_value

            pl_module.log(f"{stage}/best_{metric_name}", self._best_metrics[stage][metric_name]) 

        plt.close(fig)



    def on_train_epoch_end(self, trainer, pl_module: "ClassificationModel"):
        self._on_epoch_end(trainer, pl_module, stage="train")


    def on_validation_epoch_end(self, trainer, pl_module: "ClassificationModel"):
        self._on_epoch_end(trainer, pl_module, stage="val")
    


