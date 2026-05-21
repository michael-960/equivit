import torch.nn as nn
from typing import Union, Literal, Tuple

from .. import nn as eqnn
from .octic import OcticViTBackbone, OcticViTBackboneConfig, OcticTokenize, OcticTokenizeConfig

from .factories import build_named_sequential


from ..geometry import D4


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
            if subgroup[0] == 'D':
                dims[name] = base_dim*2
            else:
                dims[name] = base_dim

    homog_copies = [round(base_dim*mlp_ratio)] + [0] * (len(subgroup_.all_homogeneous_space_actions()) - 1)

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



