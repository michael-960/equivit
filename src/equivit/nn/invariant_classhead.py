import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import List

from .invariant import Invariantization 


class InvariantClassificationHead(nn.Module):
    """
    :class:`Invariantization` -> 
    :class:`torch.nn.LayerNorm` ->
    Pluck out class token ->
    (Optional) Dropout ->
    :class:`torch.nn.Linear` to logits.
    """
    def __init__(self, dim: int, num_logits: int, drop_rate: float = 0.0):
        super().__init__()
        self.invariantize = Invariantization()

        self.norm = nn.LayerNorm(dim)
        self.drop_rate = drop_rate
        self.head = nn.Linear(dim, num_logits)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            x: List of tensors, each of shape :math:`(B, L+1, C_i, d_i)`, where :math:`L` is the number of patches.

        Returns:
            Tensor of shape :math:`(B, N_{\text{logit}})`
        """

        # (B, L+1, C)
        y = self.invariantize(x)

        # (B, L+1, C)
        y = self.norm(y)

        # Pluck out class token
        y = y[..., -1, :]

        if self.drop_rate > 0.:
            y = F.dropout(y, p=float(self.drop_rate), training=self.training)
        
        y = self.head(y)

        return y

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
