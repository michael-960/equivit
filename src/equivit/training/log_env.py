import torch
from ..version import __version__ as equivit_version
import lightning as L

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.rank_zero import rank_zero_only
# from pytorch_lighting import loggers as pl_loggers_

import socket
import hashlib
import platform

import os

from ._core import add_tags




class EnvironmentLoggerCallback(L.Callback):
    
    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module):

        env_info = self._gather_env_info()


        for logger in trainer.loggers:
            add_tags(logger, env_info)

    def _gather_env_info(self):

        # short hash for anonymity
        hostname = socket.gethostname()
        hostname_hash = hashlib.sha256(hostname.encode()).hexdigest()[:8] 

        env_info = {
            "env.pytorch_version": torch.__version__,
            "env.lightning_version": L.__version__,
            "env.cuda_available": torch.cuda.is_available(),
            "env.cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "env.equivit_version": equivit_version,
            "env.python_version": platform.python_version(),
            "hostname_hash": hostname_hash,
        }

        env = os.environ

        # SLURM
        env_info.update({
            "env.slurm.is_slurm_job": "SLURM_JOB_ID" in env,
            "env.slurm.is_array_job": "SLURM_ARRAY_TASK_ID" in env,
            # SLURM job info
            "env.slurm.job_id": env.get("SLURM_JOB_ID"),
            "env.slurm.job_name": env.get("SLURM_JOB_NAME"),
            "env.slurm.array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
            "env.slurm.array_task_id": env.get("SLURM_ARRAY_TASK_ID"),

            # SLURM allocation / hardware
            "env.slurm.cluster_name": env.get("SLURM_CLUSTER_NAME"),
            "env.slurm.partition": env.get("SLURM_JOB_PARTITION"),
            "env.slurm.nodelist": env.get("SLURM_JOB_NODELIST"),
            "env.slurm.num_nodes": env.get("SLURM_JOB_NUM_NODES"),
            "env.slurm.num_tasks": env.get("SLURM_NTASKS"),
            "env.slurm.tasks_per_node": env.get("SLURM_TASKS_PER_NODE"),
            "env.slurm.cpus_per_task": env.get("SLURM_CPUS_PER_TASK"),
            "env.slurm.mem_per_node": env.get("SLURM_MEM_PER_NODE"),
            "env.slurm.mem_per_cpu": env.get("SLURM_MEM_PER_CPU"),
            "env.slurm.gpus": env.get("SLURM_GPUS"),
            "env.slurm.gpus_on_node": env.get("SLURM_GPUS_ON_NODE"),

            # cuda
            "env.cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        })

        env_info = {k: v for k, v in env_info.items() if v is not None}

        return env_info