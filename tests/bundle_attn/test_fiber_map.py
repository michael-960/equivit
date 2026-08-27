r"""Checks for :class:`FiberMap`."""
import numpy as np
import pytest
import torch

from equivit.geometry import IrrepType
from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn import functional as EF
from equivit.nn.utils import act_on_tensors, random_irrep_tensors

from equivit.nn.bundle_attn import FiberMap


ATOL = 1e-4   # the restriction/corestriction buffers are float32


# group, subgroup args, dims_V (one per irrep of G)
CASES = [
    (DihedralGroup(4), ('D', 2, 0), [2, 1, 1, 1, 2]),
    (DihedralGroup(4), ('C', 2),    [1, 1, 2, 1, 2]),
    (DihedralGroup(4), ('C', 4),    [1, 2, 1, 1, 2]),   # complex-type irreps of K
    (DihedralGroup(4), ('C', 1),    [2, 1, 1, 1, 1]),   # trivial K: W is a plain vector space
    (DihedralGroup(4), ('D', 4, 0), [1, 1, 1, 1, 2]),   # K = G
    (DihedralGroup(3), ('D', 1, 0), [1, 2, 2]),
    (CyclicGroup(6),   (3,),        [2, 1, 1, 2]),
    (CyclicGroup(4),   (2,),        [1, 2, 2]),
]
IDS = [f"{c[0]}-{c[1]}" for c in CASES]


def _flags(group):
    return [irrep.rep_type is IrrepType.COMPLEX for irrep in group.real_irreps().values()]


def _random_features(group, dims, batch=(2, 3)):
    return random_irrep_tensors(group, batch, dims)


def _max_err(a, b, flags):
    worst = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za = EF.to_real(za).flatten(-2, -1)
            zb = EF.to_real(zb).flatten(-2, -1)
        worst = max(worst, float(np.abs(za.detach().numpy() - zb.detach().numpy()).max()))
    return worst


def _real_dim(group, dims):
    # irrep.dim is the REAL dimension; a complex-type irrep is stored with half as many
    # complex entries, so no extra factor of two belongs here
    return sum(C * irrep.dim for irrep, C in zip(group.real_irreps().values(), dims))


def _as_matrix(group, dims_V, f):
    r"""Materialise a map :math:`V\to V` as a real matrix, by feeding it a basis of :math:`V`."""
    n = _real_dim(group, dims_V)
    columns = []
    for e in torch.eye(n):
        offset, x = 0, []
        for irrep, C in zip(group.real_irreps().values(), dims_V):
            d = irrep.dim
            block = e[offset:offset + C * d].reshape(C, d)
            offset += C * d
            x.append(EF.to_complex(block.unflatten(-1, (-1, 2)))
                     if irrep.rep_type is IrrepType.COMPLEX else block)
        y = f(x)
        flat = [EF.to_real(z).flatten(-2, -1) if c else z for z, c in zip(y, _flags(group))]
        columns.append(torch.cat([z.reshape(-1) for z in flat]))
    return torch.stack(columns, dim=1).detach().numpy()


@pytest.fixture(params=CASES, ids=IDS)
def setup(request):
    torch.manual_seed(0)
    group, args, dims_V = request.param
    inclusion = group.subgroup(*args)
    K = inclusion.source
    # a W that is a genuine subrepresentation: one copy of every irrep of K
    dims_W = {name: 1 for name in K.real_irreps().keys()}
    return group, inclusion, dims_V, dims_W, FiberMap(inclusion, dims_V, dims_W)


def test_restricted_dims_match_the_restriction(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    x = _random_features(group, dims_V)
    got = [z.shape[-2] for z in fm.restriction(x)]
    assert got == fm.restricted_dims


def test_shapes(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    x = _random_features(group, dims_V)
    w = fm.down(x)
    assert [z.shape[-2] for z in w] == fm.dims_W
    back = fm.up(w)
    assert [z.shape[-2] for z in back] == fm.dims_V
    assert [z.shape[-1] for z in back] == [z.shape[-1] for z in x]


def test_down_is_K_equivariant(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    x = _random_features(group, dims_V)
    for k in K:
        lhs = fm.down(act_on_tensors(inclusion(k), x))
        rhs = act_on_tensors(k, fm.down(x))
        assert _max_err(lhs, rhs, _flags(K)) < ATOL


def test_up_is_K_equivariant(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    w = _random_features(K, fm.dims_W)
    for k in K:
        lhs = fm.up(act_on_tensors(k, w))
        rhs = act_on_tensors(inclusion(k), fm.up(w))
        assert _max_err(lhs, rhs, _flags(group)) < ATOL


def test_the_endomorphism_commutes_with_K(setup):
    r""":math:`M_K \rho(k) = \rho(k) M_K` -- what makes conjugation by a transporter well defined."""
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    x = _random_features(group, dims_V)
    for k in K:
        lhs = fm(act_on_tensors(inclusion(k), x))
        rhs = act_on_tensors(inclusion(k), fm(x))
        assert _max_err(lhs, rhs, _flags(group)) < ATOL


def test_rank_is_bounded_by_dim_W(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    m = _as_matrix(group, dims_V, fm)
    rank = np.linalg.matrix_rank(m, tol=1e-4)
    assert rank <= _real_dim(K, fm.dims_W)
    assert rank > 0, "the endomorphism is identically zero"


def test_empty_fiber_gives_zero(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    empty = FiberMap(inclusion, dims_V, {name: 0 for name in K.real_irreps().keys()})
    x = _random_features(group, dims_V)
    for z in empty(x):
        assert float(np.abs(z.detach().numpy()).max()) == 0.0


def test_list_and_dict_specs_agree(setup):
    group, inclusion, dims_V, dims_W, fm = setup
    K = inclusion.source
    as_list = [dims_W[name] for name in K.real_irreps().keys()]
    other = FiberMap(inclusion, dims_V, as_list)
    assert other.dims_W == fm.dims_W
    assert other.restricted_dims == fm.restricted_dims


def test_up_is_not_the_transpose_of_down(setup):
    r"""The two halves must carry independent parameters, or :math:`M_K` would be symmetric."""
    group, inclusion, dims_V, dims_W, fm = setup
    m = _as_matrix(group, dims_V, fm)
    if np.linalg.matrix_rank(m, tol=1e-4) == 0:
        pytest.skip("degenerate case")
    assert not np.allclose(m, m.T, atol=1e-3)
