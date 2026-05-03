from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from ..geometry import Square, D4

from .. import nn as eqnn


@dataclass
class OcticTokenizeConfig:
    r"""
    Config dataclass for the :class:`OcticTokenize` module.
    """
    img_size: int 
    """Size of input image (assumed to be square)."""

    patch_size: int 
    """Number of pixels on each side of one patch (assumed to be square)."""

    in_channels: int
    """Number of channels in the input image."""

    dims: List[int] = None
    """Number of channels for each irrep."""

    subgroup: tuple = ('D', 4, 0)
    """Subgroup for the equivariant operations. See :meth:`equivit.geometry.DihedralGroup.subgroup` for details."""


@dataclass
class OcticViTBackboneConfig:
    r"""
    Config dataclass for the :class:`OcticViTBackbone` model.
    """
    dims: List[int]

    tokenizer_config: OcticTokenizeConfig

    transformer_block_config: eqnn.EquivariantTransformerBlockConfig

    depth: int = 12
    """Number of transformer blocks."""

    subgroup: tuple = ('D', 4, 0)
    """Subgroup for the equivariant operations. See :meth:`equivit.geometry.DihedralGroup.subgroup` for details."""

    def __post_init__(self):
        self.tokenizer_config.dims = self.dims
        self.tokenizer_config.subgroup = self.subgroup

        self.transformer_block_config.dims = self.dims
        self.transformer_block_config.group = D4.subgroup(*self.subgroup).source


class OcticTokenize(nn.Module):
    r"""
    Tokenize a square image.

    rearrange input image of shape :math:`(B, C, N^2)` to :math:`(B, (N/P)^2, P^2,C)
    -> :class:`EquivariantPatchEmbed` 
    -> :class:`EquivariantPositionalEncoding`
    -> :class:`AppendClassToken`
    """
    def __init__(self, config: OcticTokenizeConfig):
        super().__init__()
        assert config.img_size % config.patch_size == 0, f'Image size {config.img_size} must be divisible by patch size {config.patch_size}'
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patches = config.img_size // config.patch_size

        self.square1 = Square(self.n_patches-1)
        self.square2 = Square(self.patch_size-1)

        patch_action = self.square2.action
        self.patch_embed = eqnn.EquivariantPatchEmbed(
                                patch_action.pullback(patch_action.group.subgroup(*config.subgroup)), 
                                config.in_channels, config.dims)

        interpatch_action = self.square1.action
        self.pos_enc = eqnn.EquivariantPositionalEncoding(
                                interpatch_action.pullback(interpatch_action.group.subgroup(*config.subgroup)), 
                                config.dims)

        self.add_cls_token = eqnn.AppendClassToken(config.dims[0])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Applies patch embedding, positional encoding, and appends a class token to the input image tensor.

        Args:
            x (torch.Tensor): Input image tensor of shape :math:`(B, C, L_0)`, where :math:`L_0`
                is the number of pixels (i.e., ``img_size**2``).
        Returns:
            List[torch.Tensor]: A list of tensors, each of shape :math:`(B, L, C_i, d_i)`, 
            where :math:`L` is the number of tokens (number of patches plus one for the class token), 
            :math:`C_i` is the number of channels for each irrep (specifie by ``dims``), 
            and :math:`d_i` is the complex dimension of each irrep.
        """
        x = einops.rearrange(x, 'b c (n1 p1 n2 p2) -> b (n1 n2) (p1 p2) c',
                             n1=self.n_patches, p1=self.patch_size, 
                             n2=self.n_patches, p2=self.patch_size)

        x = self.patch_embed(x)
        x = self.pos_enc(x)
        return self.add_cls_token(x)



class OcticViTBackbone(nn.Module):
    r"""
        :math:`D_4`-equivariant Vision Transformer backbone for images defined on a square grid 
        (`arXiv:2505.15441 <https://arxiv.org/abs/2505.15441>`_).
        The :math:`D_4` symmetry can be optionally broken to a subgroup (e.g.,
        :math:`C_4` or :math:`C_2`) by specifying the ``subgroup`` parameter in
        the config.

        Args:
            config: An instance of :class:`OcticViTBackboneConfig` containing the configuration parameters for the model.
    """
    def __init__(self, config: OcticViTBackboneConfig):
        super().__init__()

        self.tokenization_stem = OcticTokenize(config.tokenizer_config)

        self.blocks = nn.ModuleList([
            eqnn.EquivariantTransformerBlock(config.transformer_block_config)
            for _ in range(config.depth)
        ])

    def apply_transformer_blocks(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
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
        r"""
        :meth:`tokenization_stem` followed by :meth:`apply_transformer_blocks`.

        Args:
            x (torch.Tensor): Input image tensor of shape :math:`(B, C, L_0)`, where :math:`L_0`
                is the number of pixels (i.e., ``img_size**2``).
        Returns:
            List[torch.Tensor]: A list of tensors, each of shape :math:`(B, L, C_i, d_i)`, 
            where :math:`L` is the number of tokens (number of patches plus one for the class token), 
            :math:`C_i` is the number of channels for each irrep (specified by ``dims``), 
            and :math:`d_i` is the complex dimension of each irrep.

        """
        xs = self.tokenization_stem(x)
        xs = self.apply_transformer_blocks(xs)
        return xs






