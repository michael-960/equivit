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

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_artifacts(
                    run_id=logger.run_id,
                    local_dir=str(path),
                    artifact_path=path.name,
                )
            else:
                warnings.warn(f"Logger of type {type(logger)} does not support artifact logging. Skipping artifact logging for this logger.")

