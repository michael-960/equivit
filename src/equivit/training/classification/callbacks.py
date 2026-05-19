import torch
import numpy as np
from PIL import Image
from pathlib import Path
from lightning.pytorch.callbacks import Callback

from typing import List, Dict, Any, Union, TYPE_CHECKING

import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tempfile import TemporaryDirectory

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
        self, 
        metrics: Dict[str, ClassificationMetric],
        log_every_n_epochs: int = 10
    ):
        self.metrics = metrics
        self.log_every_n_epochs = log_every_n_epochs

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


        if confmat.shape[0] <= 65: # arbitrary threshold to avoid trying to plot huge confusion matrices

            if trainer.current_epoch % self.log_every_n_epochs == 0:

                fig = Figure(figsize=(5, 5))
                FigureCanvasAgg(fig)

                ax = fig.subplots()

                # fig, ax = plt.subplots()
                sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"{stage_name} Confusion Matrix - Epoch {trainer.current_epoch}")
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("GT Labels")
                ax.set_aspect('equal')


                with TemporaryDirectory() as tmpdir:
                    imgname = f"{stage}_confmat_epoch_{trainer.current_epoch:03d}"
                    path = Path(tmpdir) / f"{imgname}.png"
                    fig.savefig(path, bbox_inches='tight')


                    npz_path = Path(tmpdir) / f"{imgname}.npz"
                    np.savez_compressed(npz_path, confmat=confmat)

                    for logger in trainer.loggers:
                        if isinstance(logger, pl_loggers.TensorBoardLogger):
                            # logger.experiment.add_figure(tag=f"{stage_name} Confusion Matrix", figure=fig, global_step=trainer.current_epoch)
                            logger.experiment.add_image(
                                tag=imgname,
                                img_tensor=np.asarray(Image.open(path).convert("RGB")).transpose(2,0,1), # convert to CxHxW format
                                global_step=trainer.current_epoch
                            )

                        elif isinstance(logger, pl_loggers.MLFlowLogger):
                            logger.experiment.log_artifact(logger.run_id, str(path), artifact_path=f"confmats")
                            logger.experiment.log_artifact(logger.run_id, str(npz_path), artifact_path=f"confmats")

                        elif isinstance(logger, pl_loggers.WandbLogger):
                            import wandb
                            artifact = wandb.Artifact(
                                name=f'{logger.experiment.name}-{logger.experiment.id}-{imgname}',
                                type='confusion-matrix'
                            )
                            artifact.add_file(str(npz_path))

                            logger.experiment.log_artifact(artifact)

                        else:
                            pass


        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(confmat)
            pl_module.log(f"{stage}/{metric_name}", metric_value, on_step=False, on_epoch=True)

            if self._best_metrics[stage][metric_name] is None or metric_value > self._best_metrics[stage][metric_name]:
                self._best_metrics[stage][metric_name] = metric_value

            pl_module.log(f"{stage}/best_{metric_name}", self._best_metrics[stage][metric_name], on_step=False, on_epoch=True) 




    def on_train_epoch_end(self, trainer, pl_module: "ClassificationModel"):
        if trainer.sanity_checking:
            return
        self._on_epoch_end(trainer, pl_module, stage="train")


    def on_validation_epoch_end(self, trainer, pl_module: "ClassificationModel"):
        if trainer.sanity_checking:
            return
        self._on_epoch_end(trainer, pl_module, stage="val")
    




class LogModelSize(Callback):

    def on_fit_start(self, trainer, pl_module: "ClassificationModel"):
        num_params = 0
        num_params_trainable = 0
        for p in pl_module.parameters():
            num_params += p.numel()
            if p.requires_grad:
                num_params_trainable += p.numel()

        for logger in trainer.loggers:
            # do we need to guard this with rank_zero_only? 
            # Probably not
            logger.log_hyperparams({
                'model.num_params': num_params,
                'model.num_params_trainable': num_params_trainable,
                'model.num_params_M': num_params / 1e6,
                'model.num_params_trainable_M': num_params_trainable / 1e6
            })





def wandb_confusion_matrix_from_array(cm, class_names=None, title="Confusion Matrix"):
    import wandb
    cm = np.asarray(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square matrix, got {cm.shape}")

    n = cm.shape[0]

    if class_names is None:
        class_names = [f'{i:02d}' for i in range(n)]

    data = [
        [class_names[i], class_names[j], int(cm[i, j])]
        for i in range(n)
        for j in range(n)
    ]

    table = wandb.Table(
        columns=["Actual", "Predicted", "nPredictions"],
        data=data,
    )

    return wandb.plot_table(
        vega_spec_name="wandb/confusion_matrix/v1",
        data_table=table,
        fields={
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        string_fields={"title": title},
    )