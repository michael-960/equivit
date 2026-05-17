from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch

class LogCudaMemory(Callback):
    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return

        step = trainer.global_step
        if step % self.every_n_steps != 0:
            return

        device = torch.cuda.current_device()

        metrics = {
            "cuda/allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "cuda/reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "cuda/max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "cuda/max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        }

        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)

