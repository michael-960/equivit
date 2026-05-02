"""
This module contains the training loop and related utilities, such as hooks for
custom code execution during training and evaluation.

Currently, only classification training is implemented. 

TODO: segmentation
"""

from .classification import *

from .log_gradient import EquivariantLinearGradientNorm