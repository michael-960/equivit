# Reference implementations of vanilla ViT layers.
# Idea: the equivariant layers for the trivial group should behave identically to these vanilla layers.

from .attn import Attention
from .drop import DropPath
from .mlp import MLP
from .norm import LayerNorm, LayerScale
from .transformer_block import TransformerBlock, TransformerBlockConfig
from .pos_enc import PositionEncoding
from .linear import Linear
from .cls_token import AppendClassToken

from .class_head import ClassificationHead
