from dataclasses import dataclass
from typing import List, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F


from .. import nn as eqnn

# from .init import init_weights

@dataclass
class ViTBackboneConfig:
    r"""
    Config dataclass for the :class:`OcticViTBackbone` model.
    """
    img_size: int 
    """Size of input image (assumed to be square)."""

    patch_size: int 
    """Number of pixels on each side of one patch (assumed to be square)."""

    dim: int
    """Number of channels for each irrep."""

    transformer_block_config: eqnn.vanilla.TransformerBlockConfig

    in_channels: int = 3 # input channel
    """Number of channels in the input image."""

    depth: int = 12
    """Number of transformer blocks."""

    def __post_init__(self):
        self.transformer_block_config.dim = self.dim



class ViTBackbone(nn.Module):
    """
    Reference implementation of ViT backbone, which is used as a reference baseline for equivariant models.
    """
    def __init__(
        self, 
        config: ViTBackboneConfig
    ):
        super().__init__()

        assert config.img_size % config.patch_size == 0, f'Image size {config.img_size} must be divisible by patch size {config.patch_size}'

        self.patch_embed = nn.Conv2d(config.in_channels, out_channels=config.dim, kernel_size=config.patch_size, stride=config.patch_size, padding='valid')
        self.pos_enc = eqnn.vanilla.PositionEncoding(spatial_size=config.img_size//config.patch_size, dim=config.dim)
        self.add_cls_token = eqnn.vanilla.AppendClassToken(dim=config.dim)

        # dpr = [config.drop_path_rate for i in range(config.depth)]

        self.blocks = nn.ModuleList([
             eqnn.vanilla.TransformerBlock(config.transformer_block_config)
             for i in range(config.depth)
        ])

        # self.apply(init_weights)

    def tokenization_stem(self, x: torch.Tensor):
        r"""
        Args:
            x: Input image tensor of shape :math:`(B, C, H, W)`
        Returns:
            Tensor of shape :math:`(B, L+1, C)`, where :math:`L` is the number of patches and :math:`C` is the embedding dimension.
        """
        x = self.patch_embed(x)
        x = x.flatten(-2,-1).permute(0,2,1)
        x = self.pos_enc(x)
        x = self.add_cls_token(x)
        return x

    def apply_transformer_blocks(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenization_stem(x) 
        x = self.apply_transformer_blocks(x)
        return x

