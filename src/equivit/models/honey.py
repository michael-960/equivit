from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F

from dataclasses import dataclass

from ..geometry import HexPatches

from .. import nn as eqnn



@dataclass
class HoneyViTBackboneConfig:
    N1: int
    N2: int

    dims: List[int]

    transformer_block_config: eqnn.EquivariantTransformerBlockConfig

    subgroup: tuple = ('D', 6, 0)

    in_channels: int = 3 # input channel

    depth: int = 12

    def __post_init__(self):
        self.transformer_block_config.dims = self.dims



class HoneyViTBackbone(nn.Module):
    def __init__(self, config: HoneyViTBackboneConfig):
        super().__init__()

        self.N1 = config.N1
        self.N2 = config.N2

        self.hexhex = HexPatches(self.N1, self.N2)

        patch_action = self.hexhex.hex2.action 

        self.patch_embed = eqnn.EquivariantPatchEmbed(
                                patch_action.pullback(patch_action.group.subgroup(*config.subgroup)), 
                                config.in_channels, config.dims)

        interpatch_action = self.hexhex.hex1.action
        self.pos_enc = eqnn.EquivariantPositionalEncoding(
                        interpatch_action.pullback(interpatch_action.group.subgroup(*config.subgroup)), 
                        config.dims)
        self.add_cls_token = eqnn.AppendClassToken(config.dims[0])

        group = interpatch_action.group.subgroup(*config.subgroup).source
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
        # (B, C, L) -> (B, L, C) -> (B, L, C, d)
        x = x.transpose(-1,-2)[..., self.hexhex.patch_inds,:]

        x = self.patch_embed(x)
        x = self.pos_enc(x)
        return self.add_cls_token(x)

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xs = self.tokenization_stem(x)
        xs = self.apply_transformer_blocks(xs)
        return xs





