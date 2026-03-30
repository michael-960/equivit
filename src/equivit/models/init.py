from timm.layers import trunc_normal_
import torch.nn as nn


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.LayerNorm) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
