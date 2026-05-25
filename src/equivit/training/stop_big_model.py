import torch
from lightning.pytorch.callbacks import Callback

class ParameterBudgetExceededError(RuntimeError):
    pass



class LimitParameterBudget(Callback):
    def __init__(
        self,
        max_params: int,
        trainable_only: bool = False,
    ):
        self.max_params = max_params
        self.trainable_only = trainable_only

    def on_fit_start(self, trainer, pl_module):
        params = [
            p for p in pl_module.parameters()
            if (p.requires_grad or not self.trainable_only)
        ]
        num_params = 0
        for p in params:
            if torch.is_complex(p):
                num_params += 2*p.numel()
            else:
                num_params += p.numel()

        # num_params = sum(p.numel() for p in params)
    
        if num_params > self.max_params:
            msg = (
                f"Model has {num_params:,} parameters, "
                f"which exceeds the limit of {self.max_params:,}."
            )
            raise ParameterBudgetExceededError(msg)