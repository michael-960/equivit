from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F

from dataclasses import dataclass

from ..geometry import HexPatches, D6

from .. import nn as eqnn

from .._core import MISSING, resolve_values

@dataclass
class HoneyTokenizeConfig:
    r"""
    Config dataclass for the :class:`HoneyTokenize` module.
    """
    N1: int
    r"""Number of hexagonal patches along one side of the hexagon *minus one* """

    N2: int 
    r"""Number of pixels along one side of the hexagonal patch *minus one* """

    in_channels: int
    """Number of channels in the input image."""

    dims: List[int] = MISSING
    """Number of channels for each irrep."""

    subgroup: tuple = MISSING
    """Subgroup for the equivariant operations. See :meth:`equivit.geometry.DihedralGroup.subgroup` for details."""

    def resolve_defaults(self):
        if self.subgroup is MISSING:
            self.subgroup = ('D', 6, 0)


@dataclass
class HoneyViTBackboneConfig:
    r"""
    Config dataclass for the :class:`HoneyViTBackbone` model.
    """
    dims: List[int]

    tokenizer_config: HoneyTokenizeConfig

    transformer_block_config: eqnn.EquivariantTransformerBlockConfig

    depth: int = MISSING
    """Number of transformer blocks."""

    subgroup: tuple = MISSING
    """Subgroup for the equivariant operations. See :meth:`equivit.geometry.DihedralGroup.subgroup` for details."""

    def __post_init__(self):
        _group = D6.subgroup(*self.subgroup).source
        self.dims = eqnn.resolve_dims(_group, self.dims)

        resolve_values(self, self.tokenizer_config, keys=('dims', 'subgroup'))
        resolve_values(self, self.transformer_block_config, keys=('dims',))

        if self.transformer_block_config.group is MISSING:
            self.transformer_block_config.group = _group
        else:
            assert self.transformer_block_config.group is _group, f"Group in transformer block config ({self.transformer_block_config.group}) does not match subgroup specified in backbone config ({_group})"

    def resolve_defaults(self):
        ...


class HoneyTokenize(nn.Module):
    r"""
    Tokenize an image defined on a collection of hexagonal patches (see :class:`HexPatches`).

    The tokenization process consists of four steps:

    1. Perform an indexing on the input image of shape :math:`(B, C, L)` to
       :math:`(B, N_1, N_2)`, where :math:`N_1` is the number of hexagonal patches
       along one side of the hexagon *minus one*, and :math:`N_2` is the number of
       pixels along one side of the hexagonal patch *minus one*.

    2. Apply :class:`EquivariantPatchEmbed` to the indexed tensor.

    3. Apply :class:`EquivariantPositionalEncoding`.

    4. Append a class token using :class:`AppendClassToken`.
    """
    def __init__(self, config: HoneyTokenizeConfig):
        super().__init__()
        config.resolve_defaults()

        self.N1 = config.N1
        self.N2 = config.N2

        self.hexhex = HexPatches(self.N1, self.N2)

        patch_action = self.hexhex.hex2.action 

        self.patch_embed = eqnn.EquivariantPatchEmbed(
                                patch_action.pullback(patch_action.group.subgroup(*config.subgroup)), 
                                in_channels=config.in_channels, 
                                dims=config.dims)

        interpatch_action = self.hexhex.hex1.action
        self.pos_enc = eqnn.EquivariantPositionalEncoding(
                        interpatch_action.pullback(interpatch_action.group.subgroup(*config.subgroup)), 
                        config.dims)
        self.add_cls_token = eqnn.AppendClassToken(config.dims[0])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
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



class HoneyViTBackbone(nn.Module):
    r"""
    :math:`D_6`-equivariant ViT backbone for images on the honeycomb lattice.
    """
    def __init__(self, config: HoneyViTBackboneConfig):
        super().__init__()
        config.resolve_defaults()

        self.tokenization_stem = HoneyTokenize(config.tokenizer_config)

        self.blocks = nn.ModuleList([
            eqnn.EquivariantTransformerBlock(config.transformer_block_config)
            for _ in range(config.depth)
        ])

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
