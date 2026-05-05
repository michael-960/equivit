from ..geometry import Group, GroupHomomorphism
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Literal, Optional, TypedDict

from .restriction import SymmetryRestriction
from .transformer_block import EquivariantTransformerBlock, EquivariantTransformerBlockConfig

from dataclasses import dataclass

from .._core import MISSING, resolve_values


@dataclass
class SymmetryBreakingTransformerConfig:
    r"""
    Configuration for :class:`SymmetryBreakingTransformer`.
    """

    group: Group
    """The initial group :math:`G`.  The first symmetry restriction will be from this group to the first subgroup in the sequence."""

    subgroups: List[tuple]
    """List of subgroups to break symmetry to. Each subgroup is specified as a tuple of arguments 
        to be passed to the :meth:`Group.subgroup` method.
        The subgroup arguments should be specified with respect to the previous group in the sequence.
    """    

    depths: List[int]
    """Number of equivariant transformer blocks to apply at each stage (after
    each symmetry restriction). Should have the same length as
    :attr:`subgroups`."""

    transformer_configs: List[EquivariantTransformerBlockConfig]
    """
    List of transformer block configs, one for each stage. Should have the same length as :attr:`subgroups`. 
    Each config should specify the parameters for the transformer blocks to be
    applied at that stage, except for the group (which will be set automatically
    based on the subgroup sequence).
    """

    def __post_init__(self):
        ...


    def validate(self):
        _n_subgroups = len(self.subgroups)
        _n_depths = len(self.depths)

        assert _n_subgroups == _n_depths, f"Number of subgroups ({_n_subgroups}) must match number of depths ({_n_depths})."

        assert len(self.transformer_configs) > 0, "At least one transformer config must be provided."

        _n_transformer_blocks_configs = len(self.transformer_configs)

        assert _n_transformer_blocks_configs == _n_subgroups, f"Number of transformer block configs ({_n_transformer_blocks_configs}) must match number of subgroups ({_n_subgroups})."

        for cfg in self.transformer_configs:
            assert cfg.group is MISSING, "Each transformer block config should not specify a group."



class SymmetryBreakingTransformer(nn.Module):
    r"""
    Let 

    .. math::
        H_m \xrightarrow{f_m} H_{m-1} \xrightarrow{f_{m-1}} H_{m-2}\dotsb H_1\xrightarrow{f_{1}} H_0 = G

    be a sequence of group homomorphisms. 

    This module applies a sequence of transformers and symmetry restrictions:

    .. math::
        \mathcal{T}^{H_m} \circ \mathrm{Res}^{H_{m-1}}_{H_m}\circ
        \mathcal{T}^{H_{m-1}} \circ \mathrm{Res}^{H_{m-2}}_{H_{m-1}}\circ 
        \mathcal{T}^{H_{m-2}}\circ 
        \dotsb \circ
        \mathcal{T}^{H_1}\circ \mathrm{Res}^G_{H_1}

    where :math:`\mathcal{T}^{H_i}` is a transformer equivariant to the group :math:`H_i`
    (sequence of :class:`EquivariantTransformerBlock`), and :math:`\mathrm{Res}^{H_{i-1}}_{H_i}` 
    is a :class:`SymmetryRestriction` from :math:`H_{i-1}` to :math:`H_{i}`.

    """
    def __init__(self, 
        config: SymmetryBreakingTransformerConfig
    ):
        super().__init__()
        config.validate()

        self.group = config.group

        self.transformers = nn.ModuleList()
        # list of list of transformer blocks
        # parameters will be called e.g. .transformers.4.5

        self.restrictions = nn.ModuleList() 

        _current_group = self.group

        for i, cfg in enumerate(config.transformer_configs):

            homomorphism = _current_group.subgroup(*config.subgroups[i])
            group = homomorphism.source
            _current_group = group

            cfg.group = group

            transformer_blocks = nn.ModuleList(
                [EquivariantTransformerBlock(cfg) for _ in range(config.depths[i])]
            )

            self.transformers.append(transformer_blocks)

            if homomorphism.is_identity():
                self.restrictions.append(nn.Identity())
            else: 
                self.restrictions.append(SymmetryRestriction(homomorphism=homomorphism))


    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:

        for restriction, transformer in zip(self.restrictions, self.transformers):
            x = restriction(x)
            for blk in transformer:
                x = blk(x)
        return x