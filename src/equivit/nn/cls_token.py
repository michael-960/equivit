import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
from timm.layers import trunc_normal_

class AppendClassToken(nn.Module):
    """
    This layer appends a learnable class token to the input sequence for the
    trivial representation, and pads zeros for the non-trivial representations.

    Args:
        dim: number of channels for the trivial representation (the first irrep).
    """
    def __init__(self, 
        dim: int,
    ):
        super().__init__()
        self.dim = dim

        self.cls_token = nn.Parameter(torch.zeros(1, dim, 1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: In general, I'm not sure what the best initilization scheme is. 
        std = 4*.02
        trunc_normal_(self.cls_token , std=std)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, L, C_i, d_i)` for each irrep
        Returns: 
            list of tensors, each of shape :math:`(*, L+1, C_i, d_i)`
        """
        y = []
        
        for i,z in enumerate(x):
            if i == 0:
                y.append(torch.cat([z, self.cls_token.expand(*z.shape[:-3], 1, self.dim, 1)], dim=-3))
            else:
                y.append(F.pad(z, (0, 0, 0, 0, 0, 1), value=0.)) # pad a zero token for non-trivial irreps
        return y



# class ExtractClassToken(nn.Module):
#     def forward(self, x: torch.Tensor):
#         """
#         Args:
#             x: tensor of shape :math:`(*, L+1, C_i, d_i)` for each irrep
#         """
#         return x[...,-1,:,:]
        