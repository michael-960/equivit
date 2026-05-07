from typing import Any, TYPE_CHECKING, Tuple


if TYPE_CHECKING:
    MISSING: Any = ...

else:
    class _MissingSentinel:
        def __repr__(self) -> str:
            return "MISSING"
    MISSING = _MissingSentinel() # Sentinel value for missing data in dataclasses.


def first_not_missing(*args, allow_missing=False):
    """
    Returns the first argument that is not MISSING. If all arguments are MISSING,
    raises a ValueError. If allow_missing is True, returns MISSING instead of raising an error.
    """
    for arg in args:
        if arg is not MISSING:
            return arg

    if allow_missing:
        return MISSING
    raise ValueError("All arguments are MISSING.")


def last_not_missing(*args, allow_missing=False):
    return first_not_missing(*args[::-1], allow_missing=allow_missing)


def resolve_values(*configs, keys: Tuple[str,...]=(), allow_missing=True):
    # MISSING, x -> MISSING, x
    # x, MISSING -> something, x
    # MISSING, x, MISSING, y -> MISSING, x, x, y
    # MISSING, x, y, MISSING, MISSING, z, MISSING -> MISSING, x, y, y, y, z, z
    
    for key in keys:
        values = [getattr(cfg, key) for cfg in configs]

        for i,cfg in enumerate(configs):
            setattr(cfg, key, last_not_missing(*values[:i+1], allow_missing=False))




