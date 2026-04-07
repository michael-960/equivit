import torch
import torch.nn as nn
import math
from typing import List


class EquivariantLinear(nn.Module):
    """
    Per-irrep linear layer. Each irrep is treated separately, and the linear
    transformation is applied to the feature dimension of each irrep.

    This actually has nothing to do with group theory. 
    It's just a collection of independent linear layers.

    Bias can be optionally added to the trivial representation (the first irrep).
    """
    def __init__(self,
        dims_in: List[int], dims_out: List[int], 
        trivial_rep_bias: bool=True,
        streams: List[torch.cuda.Stream]=None
    ):
        """
        Args:
            dims_in: list of input channels for each irrep
            dims_out: list of output channels for each irrep
            trivial_rep_bias: whether to include bias for the trivial representation (the first irrep)
            streams: list of CUDA streams to use for each irrep (optional)
        """
        super().__init__()
        assert len(dims_in) == len(dims_out), "dims_int and dims_out must have the same length"
        self.num_irreps = len(dims_in)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(dims_out[i], dims_in[i])) for i in range(len(dims_in))
        ])

        if trivial_rep_bias:
            self.bias = nn.Parameter(torch.zeros(dims_out[0], 1))
        else:
            self.register_parameter('bias', None)
            
        if streams is None:
            self.streams = [None for _ in range(len(dims_in))]
        else:
            assert len(dims_in) == len(streams), "Length of dims_in and streams must be the same"
            self.streams = streams

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

        returns: list of tensors, each of shape (*, Di, di) for each irrep (where Di is the output dimension for that irrep)
        """
        outs = [None for _ in range(self.num_irreps)]

        for i in range(self.num_irreps):
            with torch.cuda.stream(self.streams[i]):
                # outs[i] = torch.einsum('ij, ...jzk->...izk', self.weights[i], x[i])
                outs[i] = self.weights[i] @ x[i]
                if i == 0 and self.bias is not None:
                    outs[i] += self.bias
        return outs



