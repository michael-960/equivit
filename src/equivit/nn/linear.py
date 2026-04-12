import torch
import torch.nn as nn
import math
from typing import List

from ..geometry import Group, IrrepType

from .utils import assert_all_not_quaternionic


class EquivariantLinear(nn.Module):
    """
    Per-irrep linear layer. Each irrep is treated separately, and the linear
    transformation is applied to the feature dimension of each irrep.

    Bias can be optionally added to the trivial representation (the first irrep).
    """
    def __init__(self,
        group: Group,
        dims_in: List[int], 
        dims_out: List[int], 
        trivial_rep_bias: bool=True,
    ):
        """
        Args:
            group: the symmetry group that we want to respect
            dims_in: list of input channels for each irrep
            dims_out: list of output channels for each irrep
            trivial_rep_bias: whether to include bias for the trivial representation (the first irrep)
        """
        super().__init__()
        assert_all_not_quaternionic(group)

        self.group = group
        irreps = group.real_irreps()

        self.num_irreps = len(irreps)
        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64 for irrep in irreps.values()]

        assert len(dims_in) == self.num_irreps, "Length of dims_in should match number of irreps"
        assert len(dims_out) == self.num_irreps, "Length of dims_out should match number of irreps"

        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(dims_out[i], dims_in[i], dtype=self.dtypes[i])) 
            for i in range(self.num_irreps)
        ])

        if trivial_rep_bias:
            self.bias = nn.Parameter(torch.zeros(dims_out[0], 1))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # imitates source code of torch.nn.Linear
        for i in range(self.num_irreps):
            if self.weights[i].numel() > 0:
                nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))

        if self.bias is not None:
            if self.bias.numel() > 0:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x: list of tensors, each of shape (*, Ci, di) for each irrep

        Returns: list of tensors, each of shape (*, Di, di) for each irrep (where Di is the output dimension for that irrep)

        Note:
            - If the i-th irrep is of real type, then the dtype of x[i] should be real
            - If the i-th irrep is of complex type, then the dtype of x[i] should be complex.
        """
        outs = [None for _ in range(self.num_irreps)]

        for i in range(self.num_irreps):
            z = self.weights[i] @ x[i]
            outs[i] = z.view(-1).view(*z.shape) 
            # this is to ensure that z has the correct strides when there are 1's in z.shape
            # without this, PyTorch will throw an error if we later do torch.view_as_real(z) if z.shape[-1] == 1 and z.stride(-1) != 1
            if i == 0 and self.bias is not None:
                outs[i] += self.bias
        return outs



