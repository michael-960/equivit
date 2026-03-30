from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

from ..layers.cls_token import AppendClassToken
from ..layers.position_encoding import PositionEncoding
from ..layers.transformer_block import TransformerBlock

from ..registry import MODULE

from .init import init_weights


@MODULE.register('VisionTransformerBackbone')
class VisionTransformerBackbone(nn.Module):
    """
    """
    def __init__(
        self, 
        img_N: int, # size of input image (assumed to be square)
        img_patch_div: int = 8, # number of trinagular patches on each side
        in_chans: int = 3, # input channel

        # C_A1: int = 256, C_A2: int = 128, C_E: int = 128,
        dim: Tuple[int, int, int] = 640,

        # C_mlp_A1: int = 512, C_mlp_A2: int = 256, C_mlp_E: int = 256,
        dim_mlp: Tuple[int,int,int] = 1280,

        attn_bias: bool = False,

        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,

        ls_init_scale: float = 1e-4,

        depth: int = 12,
        num_heads: int = 8,

    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=img_N//img_patch_div, stride=img_N//img_patch_div, padding='valid')
        self.pos_enc = PositionEncoding(spatial_size=img_patch_div, dim=dim)
        self.add_cls_token = AppendClassToken(dim=dim)

        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
             TransformerBlock(
                  dim=dim, 
                  num_heads=num_heads,
                  mlp_dim=dim_mlp,
                  attn_bias=attn_bias,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  ls_init_values=ls_init_scale,
             )
             for i in range(depth)
        ])

        self.apply(init_weights)

    def tokenization_stem(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x.flatten(-2,-1)
        x = self.pos_enc(x)
        x = self.add_cls_token(x)
        return x

    def apply_transformer_blocks(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.tokenization_stem(x) 
        x = self.apply_transformer_blocks(x)
        return x

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)



@MODULE.register('VisionTransformer')
class VisionTransformer(nn.Module):
    """
    Reference: https://arxiv.org/abs/2010.11929
    This is probably different from the standard implementation in some details.
    This implementation is only meant as a reference baseline for our other
    equivariant or nonequivariant models.
    """
    def __init__(self,
        img_N: int, # size of input image (assumed to be square)
        img_patch_div: int = 8, # number of trinagular patches on each side
        in_chans: int = 3, # input channel

        # C_A1: int = 256, C_A2: int = 128, C_E: int = 128,
        dim: Tuple[int, int, int] = 640,

        # C_mlp_A1: int = 512, C_mlp_A2: int = 256, C_mlp_E: int = 256,
        dim_mlp: Tuple[int,int,int] = 1280,

        attn_bias: bool = False,

        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,

        ls_init_scale: float = 1e-4,

        depth: int = 12,
        num_heads: int = 8,

        drop_rate: float = 0.,
        num_classes=1000
    ):
        super().__init__()

        self.backbone = VisionTransformerBackbone(
            img_N=img_N,
            img_patch_div=img_patch_div,
            in_chans=in_chans,
            dim=dim,
            dim_mlp=dim_mlp,
            attn_bias=attn_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ls_init_scale=ls_init_scale,
            depth=depth,
            num_heads=num_heads
        )

        self.norm = nn.LayerNorm(dim)

        self.drop_rate = drop_rate

        self.head = nn.Linear(dim, num_classes)

        self.apply(init_weights)

    def forward_features(self, x: torch.Tensor):
        x = self.backbone(x)

        # (*, C, L+1) --> (*, L+1, C) --> (*, C, L+1)
        x = self.norm(x.movedim(-1,-2)).movedim(-1,-2)

        # Pluck out class token
        x = x[..., -1]

        return x


    def forward(self, x):
        x = self.forward_features(x)

        if self.drop_rate > 0.:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        
        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)