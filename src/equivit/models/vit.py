from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from ..geometry import Square

from .. import nn as eqnn



@dataclass
class ViTBackboneConfig:
    img_size: int # size of input image (assumed to be square)
    patch_size: int # number of trinagular patches on each side

    dims: List[int]

    transformer_block_config: eqnn.EquivariantTransformerBlockConfig

    subgroup: tuple = ('D', 4, 0)

    in_channels: int = 3 # input channel

    depth: int = 12

    def __post_init__(self):
        self.transformer_block_config.dims = self.dims





class ViTBackbone(nn.Module):
    def __init__(self, config: ViTBackboneConfig):
        super().__init__()
        # eqnn.EquivariantPatchEmbed()
        assert config.img_size % config.patch_size == 0, f'Image size {config.img_size} must be divisible by patch size {config.patch_size}'
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patches = config.img_size // config.patch_size

        self.square1 = Square(self.n_patches-1)
        self.square2 = Square(self.patch_size-1)

        self.patch_embed = eqnn.EquivariantPatchEmbed(self.square2.action.pullback(self.square2.action.group.subgroup(*config.subgroup)), 
                                                      config.in_channels, config.dims)
        self.pos_enc = eqnn.EquivariantPositionalEncoding(self.square1.action.pullback(self.square2.action.group.subgroup(*config.subgroup)), 
                                                          config.dims)
        self.add_cls_token = eqnn.AppendClassToken(config.dims[0])

        group = self.square1.action.group.subgroup(*config.subgroup).source
        config.transformer_block_config.group = group
        self.blocks = nn.ModuleList([
            eqnn.EquivariantTransformerBlock(config.transformer_block_config)
            for _ in range(config.depth)
        ])

    def tokenization_stem(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Applies patch embedding, positional encoding, and appends a class token to the input image tensor.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, L)
        """
        x = einops.rearrange(x, 'b c (n1 p1 n2 p2) -> b (n1 n2) (p1 p2) c',
                             n1=self.n_patches, p1=self.patch_size, 
                             n2=self.n_patches, p2=self.patch_size)

        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x = self.add_cls_token(x)
        return x

    def apply_transformer_blocks(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies a sequence of transformer blocks to the input token sequence.

        Args:
            x: list of tensors, each of shape :math:`(B, L, C_i, d_i)`
        
        Returns:
            list of tensors, each of shape :math:`(B, L, C_i, d_i)`
        """
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenization_stem(x)
        x = self.apply_transformer_blocks(x)
        return x 






# class EquivariantViT(nn.Module):
#     def __init__(self, num_layers: int, some_object):
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.backbone(x)
#         x = self.head(x)
#         return x