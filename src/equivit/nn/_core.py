from collections.abc import Sequence, Mapping
from typing import Dict, List, Optional, Callable, Union

from ..geometry import Group


def resolve_dims(group: Group, dims: Union[List[int], Dict[str, int]]) -> List[int]:
    irreps = group.real_irreps()
    if isinstance(dims, Sequence):
        assert len(dims) == len(irreps), f"Length of dims {len(dims)} must match the number of real irreps {len(irreps)}."
        return list(dims)

    if isinstance(dims, Mapping):
        assert set(dims.keys()) == set(irreps.keys()), f"Keys of dims {set(dims.keys())} must match the names of real irreps {set(irreps.keys())}."
        return [dims[name] for name in irreps.keys()]

    raise TypeError(f"Expected dims to be either a list or a dict, but got {type(dims)}")