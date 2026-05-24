"""
This module contains the training loop and related utilities, such as hooks for
custom code execution during training and evaluation.

Currently, only classification training is implemented. 

TODO: segmentation
"""

from .classification import *

from .log_gradient import LogGradientNorm

from .log_weight import LogParamNorm

from .log_artifacts import LogArtifacts


from .log_cuda import LogCudaMemory

from .stop_big_model import LimitParameterBudget