import equivit
from omegaconf import OmegaConf
import yaml

yaml_str = r"""
_target_: equivit.models.build_named_sequential

preprocess:
  _target_: equivit.nn.CropAndInterpolate
  img_size: [256, 256]
  offset: [127.5, 127.5]
  scale: 1.0746 # same pixel density
  lattice:
    _target_: equivit.geometry.HexPatches
    N1: 9
    N2: 9

backbone: 
  _target_: equivit.models.HoneyViTBackbone
  config: 
    _target_: equivit.models.HoneyViTBackboneConfig
    depth: 12
    dims: {A: 0}
    subgroup: ['D', 6, 0]

    tokenizer_config:
      _target_: equivit.models.HoneyTokenizeConfig
      N1: 9
      N2: 9
      in_channels: 3

    transformer_block_config: 
      _target_: equivit.nn.EquivariantTransformerBlockConfig
      attn_type: coupled
      num_heads: 3
      homogeneous_space_copies: [64, 0, 0, 0, 0,  0, 0, 0, 0, 0]
      activation: 'gelu'

head: 
  _target_: equivit.nn.InvariantClassificationHead
  dim: 128
  num_logits: "${oc.select:data.num_logits,10}"
"""

subgroup_args = [
    ('D', 6, 0),
    ('D', 3, 0),
    ('D', 2, 0),
    ('D', 1, 0),
    ('C', 6),
    ('C', 3),
    ('C', 2),
    ('C', 1),
]

class ListFlowDumper(yaml.Dumper):
    pass

# 2. Define a representer rule specifically for Python lists
def list_representer(dumper, data):
    # flow_style=True forces the [1, 2, 3] format
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# 3. Attach the rule to your custom dumper
ListFlowDumper.add_representer(list, list_representer)


def get_config(base_dim, subgroup_arg, num_heads):
    config = OmegaConf.create(yaml_str)

    subgroup = equivit.D6.subgroup(*subgroup_arg).source

    dims = dict()

    assert base_dim % len(subgroup) == 0

    n = base_dim // len(subgroup)


    for name, irrep in subgroup.real_irreps().items():
        if irrep.dim == 1:
            dims[name] = n
        if irrep.dim == 2:
            if subgroup_arg[0] == 'D':
                dims[name] = n*2
            else:
                dims[name] = n

    homog_copies = [n*4] + [0] * (len(subgroup.all_homogeneous_space_actions()) - 1)

    config.backbone.config.dims = dims
    config.backbone.config.subgroup = list(subgroup_arg)
    config.backbone.config.transformer_block_config.homogeneous_space_copies = homog_copies
    config.backbone.config.transformer_block_config.num_heads = num_heads
    config.head.dim = sum(dims.values())


    pyconfig = OmegaConf.to_container(config)

    return yaml.dump(pyconfig, Dumper=ListFlowDumper, default_flow_style=False, sort_keys=False)



def generate(config_dir: str):
    VARIANTS = ['extratiny', 'tiny', 'small', 'base']

    base_dim = {
        'extratiny': 72,
        'tiny': 144,
        'small': 288,
        'base': 576
    }

    num_heads = {
        'extratiny': 3,
        'tiny': 3,
        'small': 6,
        'base': 12
    }

    from pathlib import Path

    for variant in VARIANTS:
        for subgroup_arg in subgroup_args:
            cfg = get_config(base_dim[variant], subgroup_arg, num_heads=num_heads[variant])
            subgroup_str = f'{subgroup_arg[0]}{subgroup_arg[1]}'
            filename = f'{config_dir}/model/honey/{subgroup_str}/a/{variant}.yaml'
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                f.write(cfg)