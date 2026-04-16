import torch.nn as nn 
import torch
from typing import List, Callable


class ListDropout(nn.Module):
    """
    A normal dropout layer, but it processes a list of tensors instead of a single tensor.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors
        Returns:
            list of tensors of the same shapes as the input, but with dropout applied
        """
        return [self.dropout(z) for z in x]


# stolen & adapted from octic-vits
def list_drop_path(
    x: List[torch.Tensor],
    drop_prob: float = 0.,
    training: bool = False,
    scale_by_keep: bool = True
):
    """ 
    Modified from timm 
    xs: list of tensors, each of shape (B, *)
    """
    if drop_prob == 0. or not training:
        return x
    x0 = x[0]
    keep_prob = 1 - drop_prob

    # (B, 1, ..., 1)
    shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x0.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    
    return [z * random_tensor for z in x]



class ListDropPath(nn.Module):
    """
    A normal drop path layer, but it processes a list of tensors instead of a single tensor.
    """
    def __init__(self, 
                 drop_prob: float = 0., 
                 scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(B, *)`
        Returns:
            list of tensors, each of shape :math:`(B, *)`
        """
        return list_drop_path(x, drop_prob=self.drop_prob, training=self.training, scale_by_keep=self.scale_by_keep)
