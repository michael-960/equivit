import equivit
from omegaconf import OmegaConf
import yaml

yaml_str = r"""
_target_: equivit.models.build_named_sequential

preprocess:
  _target_: torch.nn.Flatten
  start_dim: -2
  end_dim: -1

backbone: 
  _target_: equivit.models.OcticViTBackbone
  config: 
    _target_: equivit.models.OcticViTBackboneConfig

    depth: 12
    dims: # 384 in total
      A1: 48
      A2: 48
      B1: 48
      B2: 48
      E1: 96

    subgroup: ['D', 4, 0]

    tokenizer_config: 
      _target_: equivit.models.OcticTokenizeConfig
      img_size: "${oc.select:data.img_size,256}"
      patch_size: 16
      in_channels: 3

    transformer_block_config: 
      _target_: equivit.nn.EquivariantTransformerBlockConfig
      attn_type: coupled
      num_heads: 6
      homogeneous_space_copies: [192, 0, 0, 0,  0, 0, 0, 0]
      activation: 'gelu'

head: 
  _target_: equivit.nn.InvariantClassificationHead
  dim: 288
  num_logits: "${oc.select:data.num_logits,10}"

"""

subgroup_args = [
    ('D', 4, 0),
    ('D', 2, 0),
    ('D', 1, 0),
    ('C', 4),
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

    subgroup = equivit.D4.subgroup(*subgroup_arg).source

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
            filename = f'{config_dir}/model/octic/{subgroup_str}/a/{variant}.yaml'
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                f.write(cfg)