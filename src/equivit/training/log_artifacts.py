from pathlib import Path

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import warnings


class LogArtifacts(Callback):
    
    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        path = Path(self.dirpath).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Artifact path is not a directory: {path}")


        num_mlflow_loggers = 0

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_artifacts(
                    run_id=logger.run_id,
                    local_dir=str(path),
                    artifact_path=path.name,
                )
                num_mlflow_loggers += 1

        if num_mlflow_loggers == 0:
            warnings.warn(
                "No MLFlowLogger found. Artifacts were not logged. Please add an MLFlowLogger to your Trainer to log artifacts."
            )

