import torch
import torch.nn as nn
import numpy as np
from .nonlinear import EquivariantNonlinear
from .linear import EquivariantLinear
from .drop import ListDropout
from typing import Tuple, List
from ..geometry import Group



class EquivariantMLP(nn.Module):
    """
    EquivariantLinear -> EquivariantNonlinear -> EquivariantLinear
    """
    def __init__(self,
        group: Group,
        dims_in: List[int],
        homogeneous_space_copies: List[int],
        dims_out: List[int],
        activation: nn.Module = nn.ReLU(),
        trivial_rep_bias: bool = True,
        drop_probs: Tuple[float] = (0.,0.),
        norm_layer = None
    ):
        """
        Args:
            dims_in: list of input channels for each irrep
            homogeneous_space_copies: list of number of copies for each homogeneous space (used in the EquivariantNonlinear layer)
            dims_out: list of output channels for each irrep
            activation: activation function to use in the EquivariantNonlinear layer
            trivial_rep_bias: whether to include bias for the trivial representation in the EquivariantLinear layers
            drop_probs: tuple of dropout probabilities for the two dropout layers
            norm_layer: normalization layer to use after the first dropout layer.
        """
        super().__init__()
        
        # shape: (num_homog_spaces, num_irreps)
        self.multipilcity_matrix = np.array([list(action.irrep_multiplicities().values()) for action in group.all_homogeneous_space_actions()])

        self.homogeneous_space_copies = np.array(homogeneous_space_copies)
        self.dims_hidden = (self.homogeneous_space_copies @ self.multipilcity_matrix).tolist()

        self.fc1 = EquivariantLinear(dims_in, self.dims_hidden, trivial_rep_bias=trivial_rep_bias)

        self.act = EquivariantNonlinear(group, homogeneous_space_copies, activation=activation)

        self.drop1 = ListDropout(drop_probs[0])

        if norm_layer is not None:
            self.norm = norm_layer(self.dims_hidden)
        else:
            self.norm = nn.Identity()

        self.fc2 = EquivariantLinear(self.dims_hidden, dims_out, trivial_rep_bias=trivial_rep_bias)

        self.drop2 = ListDropout(drop_probs[1])


    def forward(self, x):
        """
        Args:
            x: list of tensors, each of shape (*, L, Ci, di), where di is the dimension of the i-th irrep and Ci=dims_in[i]
        Returns: 
            list of tensors, each of shape (*, L, Ci_out, di), where Ci_out=dims_out[i]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


