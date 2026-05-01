import torch
import torch.nn as nn
from typing import Union


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(init_values*torch.ones((dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor of shape :math:`(*, C)`
        """
        return self.alpha * x


class Affine(nn.Module):
    def __init__(self, dim: int, bias=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((dim)))

        self.beta = None

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor of shape :math:`(*, C)`
        
        Returns:
            a tensor of shape :math:`(*, C)`
        """

        if self.beta is not None:
            y = self.alpha * x + self.beta
        else:
            y = self.alpha * x
        return y



class LayerNorm(nn.Module):
    """
    """
    def __init__(self, 
        dim, eps=1e-05, 
        elementwise_affine=True, bias=True
    ):
        super().__init__()

        if elementwise_affine:
            self.scaling = Affine(dim, bias=bias) 
        else: 
           self.scaling = nn.Identity()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: a tensor of shape :math:`(*, C)`
        Returns:
            a tensor of shape :math:`(*, C)`
        """
        std = torch.sqrt(
            x.var(dim=-1, correction=0, keepdim=True)
            + self.eps
        )

        y = (x - x.mean(dim=-1, keepdim=True)) / std
        return self.scaling(y)




