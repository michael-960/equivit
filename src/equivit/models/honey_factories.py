from typing import Optional, Tuple
import torch.nn as nn
from typing import Literal, Union

from .. import nn as eqnn
from .honey import HoneyTokenizeConfig, HoneyViTBackbone, HoneyViTBackboneConfig

from .factories import build_named_sequential


from ..geometry import D6, HexPatches


def build_honey_regular(
    img_size: int,
    N1: int,
    N2: int,
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

    activation_kw: dict=None,
    trivial_rep_attn_bias: bool=True,
    attn_drop: float=0,
    trivial_rep_proj_bias: bool=True,
    proj_drop: float=0,
    trivial_rep_mlp_bias: bool=True,
    mlp_drop_probs: Tuple[float,float]=(0,0),
    drop_path: float=0,

    head_drop_rate: float=0

) -> nn.Module:
    mlp_ratio = int(mlp_ratio)

    dims = dict()
    subgroup_incl = D6.subgroup(*subgroup)
    subgroup_ = subgroup_incl.source
    for name, irrep in subgroup_.real_irreps().items():
        if irrep.dim == 1:
            dims[name] = base_dim
        if irrep.dim == 2:
            if subgroup[0] == 'D':
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
        'preprocess': 
            eqnn.CropAndInterpolate(
                img_size=(img_size,img_size),
                offset='center',
                scale=1.0746,
                lattice=HexPatches(N1=N1, N2=N2)
        ),
        'backbone': HoneyViTBackbone(
            config=HoneyViTBackboneConfig(
                depth=depth,
                dims=dims,
                subgroup=subgroup,
                tokenizer_config=HoneyTokenizeConfig(
                    N1=N1,
                    N2=N2,
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

