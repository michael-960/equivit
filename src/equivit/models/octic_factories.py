import torch.nn as nn
from typing import Union, Literal, Tuple, List, Any, Optional

from .. import nn as eqnn
from .octic import OcticViTBackbone, OcticViTBackboneConfig, OcticTokenize, OcticTokenizeConfig

from .factories import build_named_sequential


from ..geometry import D4


def build_octic_single_homogeneous_space(
    img_size: int,
    patch_size: int,
    in_channels: int,

    depth: int,
    base_dim: int,
    mlp_ratio: Union[int,float],
    subgroup: tuple,
    homogeneous_space_index: int,

    attn_type: Literal['irrepwise', 'coupled'],
    num_heads: int,
    activation: str,

    num_logits: int,

    trivial_rep_attn_bias: bool=True,
    attn_drop: float=0,
    trivial_rep_proj_bias: bool=True,
    proj_drop: float=0,
    trivial_rep_mlp_bias: bool=True,
    mlp_drop_probs: Tuple[float,float]=(0,0),
    drop_path: float=0,

    head_drop_rate: float=0
) -> nn.Module:

    subgroup_incl = D4.subgroup(*subgroup)
    subgroup_ = subgroup_incl.source

    homogeneous_space_actions = subgroup_.all_homogeneous_space_actions()
    homogeneous_space_action = homogeneous_space_actions[homogeneous_space_index]

    homog_irrep_mults = homogeneous_space_action.irrep_multiplicities()

    dims = dict()
    for name, irrep in subgroup_.real_irreps().items():
        dims[name] = base_dim*homog_irrep_mults[name]

    dim = sum(dims.values())

    homog_copies = [0] * len(homogeneous_space_actions)
    homog_copies[homogeneous_space_index] = round(base_dim*mlp_ratio)
    

    return build_named_sequential(**{
        'preprocess': nn.Flatten(
            start_dim=-2, end_dim=-1
        ),
        'backbone': OcticViTBackbone(
            config=OcticViTBackboneConfig(
                depth=depth,
                dims=dims,
                subgroup=subgroup,
                tokenizer_config=OcticTokenizeConfig(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                ),
                transformer_block_config=eqnn.EquivariantTransformerBlockConfig(
                    attn_type=attn_type,
                    num_heads=num_heads,
                    homogeneous_space_copies=homog_copies,
                    activation=activation,

                    trivial_rep_attn_bias=trivial_rep_attn_bias,
                    attn_drop=attn_drop,
                    trivial_rep_proj_bias=trivial_rep_proj_bias,
                    proj_drop=proj_drop,
                    trivial_rep_mlp_bias=trivial_rep_mlp_bias,
                    mlp_drop_probs=mlp_drop_probs,
                    drop_path=drop_path
                )
            )
        ),
        'head': eqnn.InvariantClassificationHead(
            dim=dim,
            num_logits=num_logits,
            drop_rate=head_drop_rate
        )
    })




def build_octic_regular(
    img_size: int,
    patch_size: int,
    in_channels: int,
    depth: int,
    base_dim: int,
    mlp_ratio: Union[int,float],
    subgroup: tuple,
    attn_type: Literal['irrepwise', 'coupled'],
    num_heads: int,
    activation: str,
    num_logits: int,


    base_homogs: Optional[list]=None,

    activation_kw: Optional[dict] = None,
    trivial_rep_attn_bias: bool=True,
    attn_drop: float=0,
    trivial_rep_proj_bias: bool=True,
    proj_drop: float=0,
    trivial_rep_mlp_bias: bool=True,
    mlp_drop_probs: Tuple[float,float]=(0,0),
    drop_path: float=0,

    head_drop_rate: float=0
) -> nn.Module:

    dims = dict()
    subgroup_incl = D4.subgroup(*subgroup)
    subgroup_ = subgroup_incl.source
    for name, irrep in subgroup_.real_irreps().items():
        if irrep.dim == 1:
            dims[name] = base_dim
        if irrep.dim == 2:
            # we should condition on whether the irrep is of real or complex type here.
            if subgroup[0] == 'D': # if irrep.rep_type is REAL
                dims[name] = base_dim*2
            else: 
                dims[name] = base_dim

    if base_homogs is None:
        homog_copies = [round(base_dim*mlp_ratio)] + [0] * (len(subgroup_.all_homogeneous_space_actions()) - 1)
    else:
        _num_homogs = len(subgroup_.all_homogeneous_space_actions())
        assert len(base_homogs) == _num_homogs, f"Length of base_homog_copies ({base_homogs}) must match the number of homogeneous space actions ({_num_homogs}) of the subgroup."
        homog_copies = [round(base_dim*mlp_ratio)*ncopies for ncopies in base_homogs]

    dim = sum(dims.values())


    return build_named_sequential(**{
        'preprocess': nn.Flatten(
            start_dim=-2, end_dim=-1
        ),
        'backbone': OcticViTBackbone(
            config=OcticViTBackboneConfig(
                depth=depth,
                dims=dims,
                subgroup=subgroup,
                tokenizer_config=OcticTokenizeConfig(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                ),
                transformer_block_config=eqnn.EquivariantTransformerBlockConfig(
                    attn_type=attn_type,
                    num_heads=num_heads,
                    homogeneous_space_copies=homog_copies,
                    activation=activation,
                    activation_kw=activation_kw,

                    trivial_rep_attn_bias=trivial_rep_attn_bias,
                    attn_drop=attn_drop,
                    trivial_rep_proj_bias=trivial_rep_proj_bias,
                    proj_drop=proj_drop,
                    trivial_rep_mlp_bias=trivial_rep_mlp_bias,
                    mlp_drop_probs=mlp_drop_probs,
                    drop_path=drop_path
                )
            )
        ),
        'head': eqnn.InvariantClassificationHead(
            dim=dim,
            num_logits=num_logits,
            drop_rate=head_drop_rate
        )
    })


# NOTE: in the future, we might want to allow to specify different
# parameters for different subgroups, 

def build_octic_regular_symmetry_breaking(
    img_size: int,
    patch_size: int,
    in_channels: int,
    depths: List[int],
    base_dim: int,
    mlp_ratio: Union[int,float],
    subgroups: List[tuple],
    attn_type: Literal['irrepwise', 'coupled'],
    num_heads: list,
    activation: str,
    num_logits: int,

    activation_kw: dict = None,
    trivial_rep_attn_bias: bool=True,
    attn_drop: float=0,
    trivial_rep_proj_bias: bool=True,
    proj_drop: float=0,
    trivial_rep_mlp_bias: bool=True,
    mlp_drop_probs: Tuple[float,float]=(0,0),
    drop_path: float=0,

    head_drop_rate: float=0
) -> nn.Module:
    """
    Builds a symmetry breaking transformer backbone for D4, where
    the symmetry is broken in stages according to the provided list of
    subgroups. The model consists of a tokenization stem, followed by a sequence
    of transformer blocks, and finally an invariant classification head.
    """
    assert len(subgroups) == len(depths) == len(num_heads), "The lengths of subgroups, depths and num_heads must be the same."

    _group = D4
    transformer_configs = []

    dim = None
    dims_first_stage = None
    _base_dim = base_dim

    for i, subgroup in enumerate(subgroups):
        subgroup_incl = _group.subgroup(*subgroup)
        subgroup_ = subgroup_incl.source
        mults = subgroup_.regular_action().irrep_multiplicities()

        if i > 0:
            subgroup_index = len(_group.left_cosets(subgroup_incl))
            _base_dim *= subgroup_index

        dims = {irrep_name: mult*_base_dim for irrep_name, mult in mults.items()}

        if i == 0:
            dims_first_stage = dims
        dim = sum(dims.values())


        homog_copies = [0] * len(subgroup_.all_homogeneous_space_actions())
        homog_copies[0] = round(_base_dim*mlp_ratio)

        transformer_configs.append(
            eqnn.EquivariantTransformerBlockConfig(
                dims=list(dims.values()),
                num_heads=num_heads[i],
                homogeneous_space_copies=homog_copies,
                attn_type=attn_type,
                activation=activation,
                activation_kw=activation_kw,
                trivial_rep_attn_bias=trivial_rep_attn_bias,
                attn_drop=attn_drop,
                trivial_rep_proj_bias=trivial_rep_proj_bias,
                proj_drop=proj_drop,
                trivial_rep_mlp_bias=trivial_rep_mlp_bias,
                mlp_drop_probs=mlp_drop_probs,
                drop_path=drop_path
            )
        )

        _group = subgroup_

    
    return build_named_sequential(**{
        'preprocess': nn.Flatten(
            start_dim=-2, end_dim=-1
        ),
        'backbone': build_named_sequential(**{
            'tokenization_stem': OcticTokenize(
                config=OcticTokenizeConfig(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    subgroup=subgroups[0],
                    dims=list(dims_first_stage.values())
                )
            ),
            'transformer': eqnn.SymmetryBreakingTransformer(
                config=eqnn.SymmetryBreakingTransformerConfig(
                    group=D4,
                    subgroups=subgroups,
                    depths=depths,
                    transformer_configs=transformer_configs
                )
            )
        }),
        'head': eqnn.InvariantClassificationHead(
            dim=dim,
            num_logits=num_logits,
            drop_rate=head_drop_rate
        )
    })

# _target_: equivit.models.build_named_sequential

# preprocess:
#   _target_: torch.nn.Flatten
#   start_dim: -2
#   end_dim: -1

# backbone: 
#   _target_: equivit.models.build_named_sequential
#   tokenization_stem:
#     _target_: equivit.models.OcticTokenize
#     config:
#       _target_: equivit.models.OcticTokenizeConfig
#       img_size: 256
#       patch_size: 16
#       in_channels: 3
#       dims: [16,16,16,16,32]

#   transformer:
#     _target_: equivit.nn.SymmetryBreakingTransformer
#     config:
#       _target_: equivit.nn.SymmetryBreakingTransformerConfig
#       group:
#         _target_: equivit.DihedralGroup
#         n: 4
#       subgroups:
#         - ['D', 4, 0] # D4
#         - ['C', 4] # C4
#         - [2] # C2
#         - [1] # trivial

#       depths: [3,3,3,3]

#       transformer_configs:
#         # D4
#         - _target_: equivit.nn.EquivariantTransformerBlockConfig
#           dims: [16,16,16,16,32] # 128
#           num_heads: [2,2,2,2,4]
#           homogeneous_space_copies: [32, 0, 0, 0,  0, 0, 0, 0] # 256
#           attn_type: 'irrepwise'
#           activation: 'gelu'

#         # C4
#         - _target_: equivit.nn.EquivariantTransformerBlockConfig
#           dims: [32,32,32]
#           num_heads: [4,4,4]
#           homogeneous_space_copies: [64, 0, 0]
#           attn_type: 'irrepwise'
#           activation: 'gelu'

#         # C2
#         - _target_: equivit.nn.EquivariantTransformerBlockConfig
#           dims: [64,64]
#           num_heads: [8,8]
#           homogeneous_space_copies: [128, 0]
#           attn_type: 'irrepwise'
#           activation: 'gelu'

#         # trivial
#         - _target_: equivit.nn.EquivariantTransformerBlockConfig
#           dims: [128]
#           num_heads: [16]
#           homogeneous_space_copies: [256]
#           attn_type: 'irrepwise'
#           activation: 'gelu'


# head: 
#   _target_: equivit.nn.InvariantClassificationHead
#   dim: 128
#   num_logits: "${oc.select:data.num_logits,10}"

