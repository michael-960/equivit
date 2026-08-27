from collections.abc import Sequence, Mapping
from typing import Dict, List, Optional, Callable, Union

from ..geometry import Group


def resolve_dims(group: Group, dims: Union[List[int], Dict[str, int]]) -> List[int]:
    irreps = group.real_irreps()

    if isinstance(dims, Sequence):
        assert len(dims) == len(irreps), \
            f"Length of dims {len(dims)} must match the number of real irreps {len(irreps)}."
        return list(dims)

    if isinstance(dims, Mapping):
        # a missing name means zero, so {'A1': 1} is a valid spelling of {'A1': 1, 'A2': 0, ...};
        # an unknown name is a typo and must still raise
        unknown = set(dims.keys()) - set(irreps.keys())
        assert not unknown, \
            f"Unknown irrep names in dims: {sorted(unknown)}. The real irreps of {group} are {list(irreps.keys())}."
        return [dims.get(name, 0) for name in irreps.keys()]

    raise TypeError(f"Expected dims to be either a list or a dict, but got {type(dims)}")
