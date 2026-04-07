import torch
import torch.nn as nn
from typing import Union, List



# adapted from octic-vit
class ListLayerScale(nn.Module):
    def __init__(
        self,
        dims: List[int],
        init_values: Union[float, torch.Tensor] = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = nn.ParameterList([init_values*torch.ones((d,1)) for d in dims])
        self.n_tensors = len(dims)

    def forward(self, x):
        """
        Args:
            x: list of tesnors, each of shape (*, C_i, d_i)
        """
        return [self.alpha[i]*x[i] for i in range(self.n_tensors)]


# adapted from octic-vit
class ListAffine(nn.Module):
    def __init__(self, 
        dims: List[int], bias=True
    ):
        """
        Note: bias is only added to the first tensor (the trivial representation).
        """
        super().__init__()
        self.alpha = nn.ParameterList([torch.ones((d,1)) for d in dims])

        if bias:
            self.beta = nn.Parameter(torch.zeros(dims[0], 1))
        else:
            self.register_parameter('beta', None)

        self.n_tensors = len(dims)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tesnors, each of shape (*, C_i, d_i)
        """
        y = [self.alpha[i]*x[i] for i in range(self.n_tensors)]

        if self.beta is not None:
            y[0] = y[0] + self.beta
        
        return y


# adapted from octic-vit
class EquivariantLayerNorm(nn.Module):
    """
    We implement an irrep-wise layer norm, which is more precisely a group normalization. 
    """
    def __init__(self, 
        dims: List[int], 
        eps=1e-05, 
        elementwise_affine=True, 
        bias=True
    ):
        super().__init__()

        if elementwise_affine:
            self.scaling = ListAffine(dims, bias=bias) 
        else: 
           self.scaling = nn.Identity()
        self.eps = eps

        self.dims = dims

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tesnors, each of shape (*, C_i, d_i)
        """
        stds = [
            torch.sqrt(z.var(dim=-2, correction=0, keepdim=True).sum(dim=-1, keepdim=True) + self.eps)
            if self.dims[i] > 0 else 1.
            for i, z in enumerate(x)
        ]        

        y = [(z - z.mean(dim=-2, keepdim=True)) / stds[i] for i, z in enumerate(x)]

        return self.scaling(y)

