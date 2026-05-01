import torch
import torch.nn as nn
from typing import Tuple

# from timm.layers import trunc_normal_


class AppendClassToken(nn.Module):
    """
    No equivariance.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.reset_parameters() 

    def reset_parameters(self):
        std = 4*.02
        # trunc_normal_(self.cls_token , std=std)
        nn.init.trunc_normal_(self.cls_token, std=std)

        
    def forward(self, x: torch.Tensor)  :
        """
        Args:
            x: tensor of shape (*, L, C)

        Returns:
            tensor of shape (*, L+1, C)       
        """
        batch_shape = x.shape[:-2]

        y = torch.cat([x, self.cls_token.expand(*batch_shape, 1, self.dim)], dim=-2)

        return y
