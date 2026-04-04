import torch
import torch.nn as nn
import math
from typing import Tuple, List
from timm.layers import trunc_normal_

class AppendClassToken(nn.Module):
    def __init__(self, 
        dims: List[int],
        streams: List[torch.cuda.Stream]=None
    ):
        super().__init__()

        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(dims[i], 1), requires_grad=(i==0)) 
            for i in range(len(dims))
        ])


        # TODO: In general, I'm not sure what the best initilization scheme is. 
        std = 4*.02
        trunc_normal_(self.cls_tokens[0] , std=std)

    def forward(self, x: Tuple[torch.Tensor]):
        """
        x: list of tensors, each of shape (*, L, Ci, di) for each irrep
        returns: list of tensors, each of shape (*, L+1, Ci, di)
        """

        x_a1, x_a2, x_e = x
        batch_shape = x_a1.shape[:-2]

        y_a1 = torch.cat([x_a1, self.a1_cls_token.expand(*batch_shape, self.C_A1, 1)], dim=-1)
        y_a2 = torch.cat([x_a2, self.a2_cls_token.expand(*batch_shape, self.C_A2, 1)], dim=-1)
        y_e = torch.cat([x_e, self.e_cls_token.expand(*batch_shape, self.C_E, 2, 1)], dim=-1)

        return y_a1, y_a2, y_e



class ExtractClassToken(nn.Module):
    def forward(self, x: torch.Tensor):
        return x[...,-1]
        