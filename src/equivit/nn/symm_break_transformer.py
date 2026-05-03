from ..geometry import Group, GroupHomomorphism
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Literal, Optional, TypedDict

from .restriction import SymmetryRestriction
from .transformer_block import EquivariantTransformerBlock, EquivariantTransformerBlockConfig

from dataclasses import dataclass


def first_not_none(*args, allow_none=False):
    """
    Returns the first argument that is not None. If all arguments are None,
    raises a ValueError (unless allow_none is True, in which case returns None).
    """
    for arg in args:
        if arg is not None:
            return arg
    if allow_none:
        return None
    raise ValueError("All arguments are None")


@dataclass
class SymmetryBreakingTransformerConfig:

    group: Group

    transformer_configs: List['EquivariantTransformerConfig']

    attn_type: Optional[Literal['irrepwise', 'coupled']] = None 
    trivial_rep_attn_bias: Optional[bool] = True
    attn_drop: Optional[float] = 0.
    trivial_rep_proj_bias: Optional[bool] = True
    proj_drop: Optional[float] = 0.

    activation: Optional[str] = 'relu'
    activation_kw: Optional[dict] = None

    trivial_rep_mlp_bias: Optional[bool] = True
    mlp_drop_probs: Optional[Tuple[float, float]] = (0., 0.)

    ls_init_values: Optional[float]=None

    # norm_layer: Callable = None, 
    drop_path: Optional[float] = 0.


    def __post_init__(self):
        assert len(self.transformer_configs) > 0, "At least one transformer config must be provided."
        for i,cfg in enumerate(self.transformer_configs):
            cfg.attn_type = first_not_none(cfg.attn_type, self.attn_type)
            cfg.trivial_rep_attn_bias = first_not_none(cfg.trivial_rep_attn_bias, self.trivial_rep_attn_bias)
            cfg.attn_drop = first_not_none(cfg.attn_drop, self.attn_drop)
            cfg.trivial_rep_proj_bias = first_not_none(cfg.trivial_rep_proj_bias, self.trivial_rep_proj_bias)
            cfg.proj_drop = first_not_none(cfg.proj_drop, self.proj_drop)

            cfg.activation = first_not_none(cfg.activation, self.activation)
            cfg.activation_kw = first_not_none(cfg.activation_kw, self.activation_kw, allow_none=True)

            cfg.trivial_rep_mlp_bias = first_not_none(cfg.trivial_rep_mlp_bias, self.trivial_rep_mlp_bias)
            cfg.mlp_drop_probs = first_not_none(cfg.mlp_drop_probs, self.mlp_drop_probs)

            cfg.ls_init_values = first_not_none(cfg.ls_init_values, self.ls_init_values, allow_none=True)

            cfg.drop_path = first_not_none(cfg.drop_path, self.drop_path)


@dataclass
class EquivariantTransformerConfig:
    subgroup: tuple
    depth: int
    dims: List[int]

    homogeneous_space_copies: List[int]

    num_heads: Union[int, List[int]]

    attn_type: Optional[Literal['irrepwise', 'coupled']] = None 
    trivial_rep_attn_bias: bool = None
    attn_drop: float = None
    trivial_rep_proj_bias: bool = None
    proj_drop: float = None

    activation: str = None
    activation_kw: Optional[dict] = None

    trivial_rep_mlp_bias: bool = None
    mlp_drop_probs: Tuple[float, float] = None

    ls_init_values: Optional[float]=None
    # norm_layer: Callable = None, 
    drop_path: float = None



class SymmetryBreakingTransformer(nn.Module):
    def __init__(self, 
        config: SymmetryBreakingTransformerConfig
    ):
        super().__init__()
        self.group = config.group

        self.transformers = nn.ModuleList()
        # list of list of transformer blocks
        # parameters will be called e.g. .transformers.4.5

        self.restrictions = nn.ModuleList() 

        _current_group = self.group

        for i, cfg in enumerate(config.transformer_configs):

            homomorphism = _current_group.subgroup(*cfg.subgroup)
            group = homomorphism.source
            _current_group = group

            transformer_blocks = nn.ModuleList(
                [EquivariantTransformerBlock(EquivariantTransformerBlockConfig(
                    group=group,
                    dims=cfg.dims,
                    num_heads=cfg.num_heads,
                    homogeneous_space_copies=cfg.homogeneous_space_copies,
                    attn_type=cfg.attn_type,
                    trivial_rep_attn_bias=cfg.trivial_rep_attn_bias,
                    attn_drop=cfg.attn_drop,
                    trivial_rep_proj_bias=cfg.trivial_rep_proj_bias,
                    proj_drop=cfg.proj_drop,
                    activation=cfg.activation,
                    activation_kw=cfg.activation_kw,
                    trivial_rep_mlp_bias=cfg.trivial_rep_mlp_bias,
                    mlp_drop_probs=cfg.mlp_drop_probs,
                    ls_init_values=cfg.ls_init_values,
                    drop_path=cfg.drop_path
                )) for _ in range(cfg.depth)]
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