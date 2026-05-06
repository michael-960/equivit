import torch.nn as nn
from collections import OrderedDict
from typing import Dict

def build_named_sequential(**layers) -> nn.Sequential:
    """
    Converts a standard dictionary into a PyTorch named Sequential block.
    This is highly useful for configuration frameworks like Hydra.
    """
    return nn.Sequential(OrderedDict(layers))



# preprocess:
#   _target_: torch.nn.Flatten
#   start_dim: -2
#   end_dim: -1

# backbone: 
#   _target_: equivit.models.OcticViTBackbone
#   config: 
#     _target_: equivit.models.OcticViTBackboneConfig

#     depth: 12
#     dims: [96,96,96] # 384
#     subgroup: ['C', 4]

#     tokenizer_config: 
#       _target_: equivit.models.OcticTokenizeConfig
#       img_size: 256
#       patch_size: 16
#       in_channels: 3

#     transformer_block_config: 
#       _target_: equivit.nn.EquivariantTransformerBlockConfig
#       attn_type: coupled
#       num_heads: 6
#       homogeneous_space_copies: [384, 0, 0] # 384 * 4 = 1536
#       activation: 'gelu'

# head: 
#   _target_: equivit.nn.InvariantClassificationHead
#   dim: 288
#   num_logits: "${oc.select:data.num_logits,10}"


