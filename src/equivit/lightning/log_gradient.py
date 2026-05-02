import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

from .. import nn as eqnn


class EquivariantLinearGradientNorm(Callback):
    def __init__(self, norm_type: float = 2.0):
        super().__init__()
        self.norm_type = norm_type
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
        self._layer_map = {
            name: module for name, module in model.named_modules()
            if isinstance(module, eqnn.EquivariantLinear)
        }

        print(f"GradNormLogger: Found {len(self._layer_map)} equivariant layers to track.")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, layer in self._layer_map.items():
            # Collect gradients for this specific layer's parameters
            grads = [
                p.grad.detach() for p in layer.parameters() 
                if p.grad is not None
            ]

            if grads:
                # Compute L2 norm for this specific layer
                device = grads[0].device
                layer_norm = torch.norm(
                    torch.stack([torch.norm(g, self.norm_type).to(device) for g in grads]), 
                    self.norm_type
                )
                
                # Log using the layer's name (e.g., grad_norm/encoder.layers.0)
                pl_module.log(
                    f"grad_norm/{name}", 
                    layer_norm, 
                    on_step=True, 
                    on_epoch=False
                )