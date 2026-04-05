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
        trivial_rep_bias: bool = True,
        drop_probs: Tuple[float] = (0.,0.),
        norm_layer = None
    ):
        super().__init__()
        
        # shape: (num_homog_spaces, num_irreps)
        self.multipilcity_matrix = np.array([list(action.irrep_multiplicities().values()) for action in group.all_homogeneous_space_actions()])

        self.homogeneous_space_copies = np.array(homogeneous_space_copies)
        self.dims_hidden = (self.homogeneous_space_copies @ self.multipilcity_matrix).tolist()

        self.fc1 = EquivariantLinear(dims_in, self.dims_hidden, trivial_rep_bias=trivial_rep_bias)

        self.act = EquivariantNonlinear(group, homogeneous_space_copies)

        self.drop1 = ListDropout(drop_probs[0])

        self.norm = nn.Identity()

        self.norm = norm_layer(self.dims_hidden) if (norm_layer is not None) else nn.Identity()

        self.fc2 = EquivariantLinear(self.dims_hidden, dims_out, trivial_rep_bias=trivial_rep_bias)

        self.drop2 = ListDropout(drop_probs[1])


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


