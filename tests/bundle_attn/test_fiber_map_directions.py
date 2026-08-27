r"""
Checks for the ``directions`` argument of :class:`FiberMap`.

The existing `tests/bundle_attn/test_fiber_map.py` covers the ``'both'`` behaviour and is
unaffected; these are the new cases only.
"""
import numpy as np
import pytest
import torch

from equivit.geometry import IrrepType
from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn import functional as EF
from equivit.nn.utils import act_on_tensors, random_irrep_tensors

from equivit.nn.bundle_attn import FiberMap


ATOL = 1e-4

CASES = [
    (DihedralGroup(4), ('D', 2, 0), [2, 1, 1, 1, 2]),
    (DihedralGroup(4), ('C', 4),    [1, 2, 1, 1, 2]),
    (DihedralGroup(3), ('D', 1, 0), [1, 2, 2]),
    (CyclicGroup(6),   (3,),        [2, 1, 1, 2]),
]
IDS = [f"{c[0]}-{c[1]}" for c in CASES]


@pytest.fixture(params=CASES, ids=IDS)
def setup(request):
    torch.manual_seed(0)
    group, args, dims_V = request.param
    inclusion = group.subgroup(*args)
    dims_W = {name: 1 for name in inclusion.source.real_irreps().keys()}
    return group, inclusion, dims_V, dims_W


def _flags(group):
    return [irrep.rep_type is IrrepType.COMPLEX for irrep in group.real_irreps().values()]


def _err(a, b, flags):
    worst = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za, zb = EF.to_real(za).flatten(-2, -1), EF.to_real(zb).flatten(-2, -1)
        worst = max(worst, float((za - zb).detach().abs().max()))
    return worst


def test_default_is_both(setup):
    group, inclusion, dims_V, dims_W = setup
    fm = FiberMap(inclusion, dims_V, dims_W)
    assert fm.directions == 'both' and fm.has_down and fm.has_up


def test_restricted_dims_agree_across_directions(setup):
    """The multiplicities are read off the restriction when there is no corestriction."""
    group, inclusion, dims_V, dims_W = setup
    dims = [FiberMap(inclusion, dims_V, dims_W, d).restricted_dims for d in FiberMap.DIRECTIONS]
    assert dims[0] == dims[1] == dims[2]


def test_down_only_matches_both(setup):
    group, inclusion, dims_V, dims_W = setup
    torch.manual_seed(1)
    a = FiberMap(inclusion, dims_V, dims_W, 'down')
    torch.manual_seed(1)
    b = FiberMap(inclusion, dims_V, dims_W, 'both')
    x = random_irrep_tensors(group, (2, 3), dims_V)
    assert _err(a.down(x), b.down(x), _flags(inclusion.source)) < ATOL


def test_missing_half_raises(setup):
    group, inclusion, dims_V, dims_W = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    down_only = FiberMap(inclusion, dims_V, dims_W, 'down')
    up_only = FiberMap(inclusion, dims_V, dims_W, 'up')
    w = random_irrep_tensors(inclusion.source, (2,), down_only.dims_W)
    with pytest.raises(AssertionError):
        down_only.up(w)
    with pytest.raises(AssertionError):
        up_only.down(x)
    with pytest.raises(AssertionError):
        down_only(x)


def test_unused_half_is_absent_from_the_state_dict(setup):
    group, inclusion, dims_V, dims_W = setup
    keys = set(FiberMap(inclusion, dims_V, dims_W, 'down').state_dict())
    assert not any(k.startswith('up_linear') or k.startswith('corestriction') for k in keys)
    assert any(k.startswith('down_linear') for k in keys)
    params = list(FiberMap(inclusion, dims_V, dims_W, 'up').parameters())
    assert params and all(p.requires_grad for p in params)


def test_two_down_halves_give_a_general_M(setup):
    r""":math:`M = \phi_q^\top \phi_k` is :math:`K`-equivariant and not symmetric, so the pair of
    ``'down'`` halves parameterises the same forms as one of each direction."""
    group, inclusion, dims_V, dims_W = setup
    K = inclusion.source
    q = FiberMap(inclusion, dims_V, dims_W, 'down')
    k = FiberMap(inclusion, dims_V, dims_W, 'down')

    def bilinear(x, y):
        qx, ky = q.down(x), k.down(y)
        total = 0.0
        for zq, zk, cplx in zip(qx, ky, _flags(K)):
            if cplx:
                zq, zk = EF.to_real(zq).flatten(-2, -1), EF.to_real(zk).flatten(-2, -1)
            total = total + (zq * zk).sum()
        return total.detach()

    x = random_irrep_tensors(group, (), dims_V)
    y = random_irrep_tensors(group, (), dims_V)

    for g in K:
        moved = bilinear(act_on_tensors(inclusion(g), x), act_on_tensors(inclusion(g), y))
        assert abs(float(moved) - float(bilinear(x, y))) < 1e-3

    assert abs(float(bilinear(x, y)) - float(bilinear(y, x))) > 1e-3
