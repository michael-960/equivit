r"""Checks for :class:`OrbitTransport`, and for the orbit of forms it assembles."""
import numpy as np
import pytest
import torch

from equivit.geometry import IrrepType
from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn import functional as EF
from equivit.nn.bundle_attn import FiberMap, build_head_set, head_transporters
from equivit.nn.utils import act_on_tensors, random_irrep_tensors

from equivit.nn.bundle_attn import OrbitTransport


ATOL = 1e-4


# group, subgroup list, dims_V, dims_W of the orbit under test
CASES = [
    (DihedralGroup(4), [('D', 2, 0), ('C', 2)], [2, 1, 1, 1, 2], {'A1': 1, 'B1': 1}),
    (DihedralGroup(4), [('C', 2)],              [1, 1, 2, 1, 2], {'A': 2}),
    (DihedralGroup(4), [('C', 1)],              [1, 1, 1, 1, 2], {'A': 3}),   # regular orbit
    (DihedralGroup(4), [('D', 4, 0)],           [1, 1, 1, 1, 2], {'A1': 1, 'E1': 1}),  # one head
    (DihedralGroup(3), [('D', 1, 0), ('C', 3)], [1, 2, 2],       {'A1': 1, 'A2': 1}),
    (CyclicGroup(6),   [(3,), (2,)],            [2, 1, 1, 2],    {'A': 1, 'E1': 1}),   # K = C_3
    (CyclicGroup(4),   [(2,)],                  [1, 2, 2],       {'A': 1, 'B': 1}),    # K = C_2
]
IDS = [f"{c[0]}-{c[1][0]}" for c in CASES]


@pytest.fixture(params=CASES, ids=IDS)
def setup(request):
    torch.manual_seed(0)
    group, subgroups, dims_V, dims_W = request.param
    heads = build_head_set(group, subgroups)
    transporters = head_transporters(heads)[0]          # orbit 0
    transport = OrbitTransport(group, transporters)
    fiber = FiberMap(heads.inclusions[0], dims_V, dims_W)
    return group, heads, transporters, transport, fiber, dims_V


def _flags(group):
    return [irrep.rep_type is IrrepType.COMPLEX for irrep in group.real_irreps().values()]


def _err(a, b, flags):
    worst = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za, zb = EF.to_real(za).flatten(-2, -1), EF.to_real(zb).flatten(-2, -1)
        worst = max(worst, float((za - zb).detach().abs().max()))
    return worst


def test_head_axis_shapes(setup):
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2, 3), dims_V)
    stacked = transport.pull_back(x)
    for z, z0 in zip(stacked, x):
        assert z.shape == (2, 3, len(transporters), *z0.shape[-2:])
        assert z.dtype == z0.dtype


def test_pull_back_matches_act_on_tensors(setup):
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    stacked = transport.pull_back(x)
    for p, g in enumerate(transporters):
        want = act_on_tensors(g.inv(), x)
        got = [z[..., p, :, :] for z in stacked]
        assert _err(got, want, _flags(group)) < ATOL


def test_push_forward_matches_act_on_tensors(setup):
    group, heads, transporters, transport, fiber, dims_V = setup
    y = [torch.stack([z] * len(transporters), dim=-3)
         for z in random_irrep_tensors(group, (2,), dims_V)]
    pushed = transport.push_forward(y)
    for p, g in enumerate(transporters):
        want = act_on_tensors(g, [z[..., p, :, :] for z in y])
        got = [z[..., p, :, :] for z in pushed]
        assert _err(got, want, _flags(group)) < ATOL


def test_round_trip(setup):
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    back = transport.push_forward(transport.pull_back(x))
    for p in range(len(transporters)):
        assert _err([z[..., p, :, :] for z in back], x, _flags(group)) < ATOL


def test_head_zero_is_the_base_form(setup):
    r""":math:`g_0 \in K`, so head 0 must reproduce :math:`M_K` itself."""
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    forms = transport.conjugate(fiber, x)
    assert _err([z[..., 0, :, :] for z in forms], fiber(x), _flags(group)) < ATOL


def test_conjugation_agrees_with_building_each_form_separately(setup):
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    forms = transport.conjugate(fiber, x)
    for p, g in enumerate(transporters):
        want = act_on_tensors(g, fiber(act_on_tensors(g.inv(), x)))
        assert _err([z[..., p, :, :] for z in forms], want, _flags(group)) < ATOL


def test_the_orbit_of_forms_is_equivariant(setup):
    r""":math:`M_{g\cdot p}(\rho(g)x) = \rho(g) M_p(x)` for every :math:`g \in G` and every head --
    Theorem 1's condition, now with a learned :math:`M_K` and the real transporters."""
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    forms = transport.conjugate(fiber, x)
    for g in group:
        moved = transport.conjugate(fiber, act_on_tensors(g, x))
        for p in range(len(transporters)):
            q = heads.structured(heads.action(g)[heads.flat(0, p)])[1]
            want = act_on_tensors(g, [z[..., p, :, :] for z in forms])
            got = [z[..., q, :, :] for z in moved]
            assert _err(got, want, _flags(group)) < ATOL


def test_gradients_reach_the_fiber_map(setup):
    """The head axis is built with einsum against buffers, so autograd must pass through it."""
    group, heads, transporters, transport, fiber, dims_V = setup
    x = random_irrep_tensors(group, (2,), dims_V)
    loss = sum(z.abs().square().sum() for z in transport.conjugate(fiber, x))
    loss.backward()
    grads = [p.grad for p in fiber.parameters() if p.numel() > 0]
    assert grads and all(g is not None for g in grads)
    assert any(float(g.abs().max()) > 0 for g in grads)


def test_forms_are_distinct_unless_the_orbit_is_a_point(setup):
    """A sanity check that conjugation is doing something."""
    group, heads, transporters, transport, fiber, dims_V = setup
    if len(transporters) == 1:
        pytest.skip("single-head orbit")
    x = random_irrep_tensors(group, (2,), dims_V)
    forms = transport.conjugate(fiber, x)
    spread = max(float((z[..., 0, :, :] - z[..., 1, :, :]).detach().abs().max()) for z in forms)
    assert spread > 1e-3
