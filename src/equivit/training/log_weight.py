import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from hydra.utils import get_class
from typing import List, Dict, Any, Union

from .. import nn as eqnn



class LogParamNorm(Callback):
    def __init__(self,
        layer_types: List[Union[str, type]],
        norm_type: float = 2.0,
        log_freq: int = 20
    ):
        super().__init__()
        self.norm_type = norm_type
        self.log_freq = log_freq

        self.layer_types = []
        for lt in layer_types:
            if isinstance(lt, str):
                cls = get_class(lt)
                self.layer_types.append(cls)
            elif isinstance(lt, type):
                self.layer_types.append(lt)
            else:
                raise ValueError(f"Invalid layer type: {lt}. Must be a string or a class.")

        self._layer_map = {}

    def setup(self, trainer, pl_module, stage=None):
        """
        Runs once at the start. Validates that the model is a transformer
        and maps out the layers to avoid searching every step.
        """

        model = pl_module.model
        
        # If compiled, we need to look at the original module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # Identify equivariant linear layers
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.layer_types)):
                self._layer_map[name] = module

        print(f"{self.__class__.__name__}: Found {len(self._layer_map)} layers to track.")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_freq == 0:
            for name, layer in self._layer_map.items():
                for pname, p in layer.named_parameters():
                    param_norm = torch.norm(p.detach(), self.norm_type)

                    pl_module.log(
                        f"param_norm/{name}.{pname}", 
                        param_norm, 
                        on_step=True, 
                        on_epoch=False
                    )