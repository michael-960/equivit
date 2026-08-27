from . import geometry
from . import nn

from . import models

from . import data

from . import training

from .geometry import D1, D2, D3, D4, D5, D6, D7, D8, D12
from .geometry import C2, C3, C4, C5, C6, C7, C8, C12

from .geometry import DihedralGroup, CyclicGroup, TRIVIAL_GROUP

from .geometry import Lattice, Hexagon, Triangle, Square, Honeycomb

from .nn import act_on_tensors, random_irrep_tensors, induced_action_on_tensors

# from . import lightning


from .version import __version__

