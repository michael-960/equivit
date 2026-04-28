import torch
import torch.nn as nn
import numpy as np
from .nonlinear import EquivariantNonlinear
from .linear import EquivariantLinear
from .drop import ListDropout
from typing import Tuple, List
from ..geometry import Group
import math

from .init import kaiming_uniform_, complex_kaiming_uniform_



class EquivariantMLP(nn.Module):
    r"""
    EquivariantLinear -> EquivariantNonlinear -> ListDropout 
    (-> norm layer) 
    -> EquivariantLinear -> ListDropout

    Args:
        group: the symmetry group :math:`G` that we want to respect
        dims_in: list of integers :math:`C_0, \dotsb, C_{M-1}` specifying the number of channels for each irrep in the input feature vector
        homogeneous_space_copies: list of integers :math:`\mu_0, \dotsc, \mu_{N-1}` specifying the number of copies of each homogeneous space (used in the EquivariantNonlinear layer)
        dims_out: list of integers :math:`C_0', \dotsb, C_{M-1}'` specifying the number of channels for each irrep in the output feature vector
        activation: activation function to use in the EquivariantNonlinear layer
        trivial_rep_bias: whether to include bias for the trivial representation in the EquivariantLinear layers
        drop_probs: tuple of two dropout probabilities for the two dropout layers
        norm_layer: normalization layer to use after the first dropout layer.
    """
    def __init__(self,
        group: Group,
        dims_in: List[int],
        homogeneous_space_copies: List[int],
        dims_out: List[int],
        activation: nn.Module = nn.ReLU(),
        trivial_rep_bias: bool = True,
        drop_probs: Tuple[float, float] = (0.,0.),
        norm_layer = None
    ):
        super().__init__()

        _num_irreps = len(group.real_irreps())
        assert len(dims_in) == _num_irreps, f"Length of dims_in ({len(dims_in)}) must be equal to the number of real irreps of the group ({_num_irreps})"
        assert len(dims_out) == _num_irreps, f"Length of dims_in ({len(dims_out)})must be equal to the number of real irreps of the group ({_num_irreps})"
        assert len(homogeneous_space_copies) == len(group.all_homogeneous_space_actions()), f"Length of homogeneous_space_copies ({len(homogeneous_space_copies)}) must be equal to the number of homogeneous space actions of the group ({len(group.all_homogeneous_space_actions())})"

        self.dims_in = dims_in
        
        # shape: (num_homog_spaces, num_irreps)
        self.multipilcity_matrix = np.array([list(action.irrep_multiplicities().values()) for action in group.all_homogeneous_space_actions()])

        self.homogeneous_space_copies = np.array(homogeneous_space_copies)
        self.dims_hidden = (self.homogeneous_space_copies @ self.multipilcity_matrix).tolist()

        self.fc1 = EquivariantLinear(group, dims_in, self.dims_hidden, trivial_rep_bias=trivial_rep_bias)

        self.act = EquivariantNonlinear(group, homogeneous_space_copies, activation=activation)

        self.drop1 = ListDropout(drop_probs[0])

        if norm_layer is not None:
            self.norm = norm_layer(self.dims_hidden)
        else:
            self.norm = nn.Identity()

        self.fc2 = EquivariantLinear(group, self.dims_hidden, dims_out, trivial_rep_bias=trivial_rep_bias)

        self.drop2 = ListDropout(drop_probs[1])

        self.reset_parameters()

    def reset_parameters(self):
        gain1 = math.sqrt(2)

        with torch.no_grad():
            for i, weight in enumerate(self.fc1.weights):
                if weight.numel() > 0:
                    if weight.dtype.is_complex:
                        complex_kaiming_uniform_(weight, fan_in=self.dims_in[i], gain=gain1)
                    else:
                        kaiming_uniform_(weight, fan_in=self.dims_in[i], gain=gain1)


            gain2 = 1.0
            for i, weight in enumerate(self.fc2.weights):
                if weight.numel() > 0:
                    if weight.dtype.is_complex:
                        complex_kaiming_uniform_(weight, fan_in=self.dims_hidden[i], gain=gain2)
                    else:
                        kaiming_uniform_(weight, fan_in=self.dims_hidden[i], gain=gain2)


    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)`, where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep
        Returns: 
            list of tensors, each of shape :math:`(*, L, C_i', d_i)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


