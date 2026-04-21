from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F

from dataclasses import dataclass

from ..geometry import HoneyTriangle, EquivariantPullbackBundle, D6

from .. import nn as eqnn




@dataclass
class HexViTBackboneConfig:
    N_honey: int    
    N_triangle: int


    dims: List[int]

    transformer_block_config: eqnn.EquivariantTransformerBlockConfig

    # subgroup: tuple = ('D', 4, 0)
    # TODO: think about how to break symmetry

    in_channels: int = 3 # input channel

    depth: int = 12

    def __post_init__(self):
        self.transformer_block_config.dims = self.dims



class HexViTBackbone(nn.Module):
    """
    """
    def __init__(self, config: HexViTBackboneConfig):
        super().__init__()
        # eqnn.EquivariantPatchEmbed()
        # assert config.img_size % config.patch_size == 0, f'Image size {config.img_size} must be divisible by patch size {config.patch_size}'
        self.N_honey = config.N_honey
        self.N_triangle = config.N_triangle

        self.honey_triangle = HoneyTriangle(N_honey=self.N_honey, N_triangle=self.N_triangle)

        self.n_patches = self.honey_triangle.honey.L


        self.patch_embed = eqnn.EquivariantPatchEmbed(
                            self.honey_triangle.triangle.action,
                            config.in_channels, config.dims)

        pullback_bundle = EquivariantPullbackBundle(
                self.honey_triangle.honey.action,
                ('D', 3, 0),
                [D6[''], D6['rrr']],
                [0] * len(self.honey_triangle.honey.action.orbits())
        )

        self.pos_enc = eqnn.EquivariantInducedPositionalEncoding(
                        pullback_bundle, 
                        config.dims)

        self.add_cls_token = eqnn.AppendClassToken(config.dims[0])

        group = self.honey_triangle.honey.action.group # should be D6
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
        # (B, C, L) -> (B, L, C) -> (B, Npatch, Lpatch, C)
        x = x.transpose(-1,-2)[..., self.honey_triangle.patch_inds,:]
        xs = self.patch_embed(x)
        xs = self.pos_enc(xs)
        return self.add_cls_token(xs)

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





