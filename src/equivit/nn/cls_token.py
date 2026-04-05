import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
from timm.layers import trunc_normal_

class AppendClassToken(nn.Module):
    def __init__(self, 
        dim: int,
        streams: List[torch.cuda.Stream]=None
    ):
        super().__init__()
        self.dim = dim

        self.cls_token = nn.Parameter(torch.zeros(1, dim, 1), requires_grad=True)

        # self.cls_tokens = nn.ParameterList([
        #     nn.Parameter(torch.zeros(dims[i], 1), requires_grad=(i==0)) 
        #     for i in range(len(dims))
        # ])

        # TODO: In general, I'm not sure what the best initilization scheme is. 
        std = 4*.02
        trunc_normal_(self.cls_token , std=std)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x: list of tensors, each of shape (*, L, Ci, di) for each irrep
        note: d0 = 1 because the trivial representation is one-dimensional
        returns: list of tensors, each of shape (*, L+1, Ci, di)
        """
        y = []
        
        for i,z in enumerate(x):
            if i == 0:
                y.append(torch.cat([z, self.cls_token.expand(*z.shape[:-3], 1, self.dim, 1)], dim=-3))
            else:
                y.append(F.pad(z, (0, 0, 0, 0, 0, 1), value=0.)) # pad a zero token for non-trivial irreps
        return y



class ExtractClassToken(nn.Module):
    def forward(self, x: torch.Tensor):
        return x[...,-1,:,:]
        