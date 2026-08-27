r"""Checks for :class:`BundleAttention`."""
import numpy as np
import pytest
import torch

from equivit.geometry import IrrepType
from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn import functional as EF
from equivit.nn.utils import act_on_tensors, random_irrep_tensors

from equivit.nn.bundle_attn import BundleAttention


ATOL = 1e-4

# group, dims_V, subgroups, attention fibers, value fibers
CASES = [
    ("D4-two-orbits", DihedralGroup(4), [2, 1, 1, 1, 2],
     [('D', 2, 0), ('C', 2)], [{'A1': 1, 'B1': 1}, {'A': 2}], [{'A1': 1}, {'A': 1, 'B': 1}]),
    ("D4-regular", DihedralGroup(4), [1, 1, 1, 1, 2],
     [('C', 1)], [{'A': 2}], [{'A': 3}]),
    ("D4-single-head", DihedralGroup(4), [1, 1, 1, 1, 2],
     [('D', 4, 0)], [{'A1': 1, 'E1': 1}], [{'A1': 1, 'E1': 1}]),
    ("D4-three-orbits", DihedralGroup(4), [1, 1, 1, 1, 1],
     [('C', 4), ('D', 1, 0), ('D', 4, 0)],
     [{'A': 1, 'E1': 1}, {'A1': 1}, {'A1': 1}], [{'A': 1}, {'A1': 1, 'A2': 1}, {'E1': 1}]),
    ("D3", DihedralGroup(3), [1, 2, 2],
     [('D', 1, 0), ('C', 3)], [{'A1': 1, 'A2': 1}, {'A': 1}], [{'A1': 1}, {'A': 1, 'E1': 1}]),
    ("C6", CyclicGroup(6), [2, 1, 1, 2],
     [(3,), (2,)], [{'A': 1, 'E1': 1}, {'A': 1}], [{'A': 1}, {'A': 1, 'B': 1}]),
    ("C4-complex", CyclicGroup(4), [1, 2, 2],
     [(2,), (1,)], [{'A': 1, 'B': 1}, {'A': 2}], [{'A': 1}, {'A': 1}]),
]
IDS = [c[0] for c in CASES]


@pytest.fixture(params=CASES, ids=IDS)
def layer(request):
    torch.manual_seed(0)
    _, group, dims_V, subgroups, attn, value = request.param
    return group, dims_V, BundleAttention(group, dims_V, subgroups, attn, value)


def _flags(group):
    return [irrep.rep_type is IrrepType.COMPLEX for irrep in group.real_irreps().values()]


def _err(a, b, flags):
    worst = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za, zb = EF.to_real(za).flatten(-2, -1), EF.to_real(zb).flatten(-2, -1)
        worst = max(worst, float((za - zb).detach().abs().max()))
    return worst


def test_shapes_and_dtypes_round_trip(layer):
    group, dims_V, attn = layer
    x = random_irrep_tensors(group, (2, 7), dims_V)
    y = attn(x)
    for zy, zx in zip(y, x):
        assert zy.shape == zx.shape and zy.dtype == zx.dtype


def test_head_count_is_the_sum_of_indices(layer):
    group, dims_V, attn = layer
    expected = sum(group.order() // incl.source.order() for incl in attn.head_set.inclusions)
    assert attn.num_heads == expected


def test_the_layer_is_G_equivariant(layer):
    r"""The whole point: :math:`f(\rho(g)x) = \rho(g) f(x)` for every :math:`g \in G`."""
    group, dims_V, attn = layer
    x = random_irrep_tensors(group, (2, 5), dims_V)
    y = attn(x)
    for g in group:
        moved = attn(act_on_tensors(g, x))
        assert _err(moved, act_on_tensors(g, y), _flags(group)) < ATOL, f"failed at g = {g}"


def test_permutation_equivariance_is_not_assumed(layer):
    """The layer must not be invariant under permuting tokens -- attention has to do something."""
    group, dims_V, attn = layer
    x = random_irrep_tensors(group, (5,), dims_V)
    y = attn(x)
    perm = [1, 0, 2, 3, 4]
    permuted = attn([z[perm] for z in x])
    assert _err(permuted, [z[perm] for z in y], _flags(group)) < ATOL   # equivariant to permutation
    assert max(float((z[0] - z[1]).detach().abs().max()) for z in y) > 1e-4  # but not constant


def test_gradients_reach_every_parameter(layer):
    group, dims_V, attn = layer
    x = random_irrep_tensors(group, (2, 4), dims_V)
    sum(z.abs().square().sum() for z in attn(x)).backward()
    live = [(n, p) for n, p in attn.named_parameters() if p.numel() > 0]
    assert live
    missing = [n for n, p in live if p.grad is None or float(p.grad.abs().max()) == 0.0]
    assert not missing, f"no gradient reached: {missing}"


def test_output_is_a_sum_over_orbits(layer):
    group, dims_V, attn = layer
    x = random_irrep_tensors(group, (3,), dims_V)
    full = attn(x)

    total = None
    for i in range(attn.head_set.num_orbits):
        keep = BundleAttention(
            group, dims_V,
            [attn.head_set.inclusions[i]],
            [{name: m for name, m in
              zip(attn.head_set.inclusions[i].source.real_irreps().keys(), attn.queries[i].dims_W)}],
            [{name: m for name, m in
              zip(attn.head_set.inclusions[i].source.real_irreps().keys(), attn.values[i].dims_W)}],
        )
        keep.queries[0].load_state_dict(attn.queries[i].state_dict())
        keep.keys[0].load_state_dict(attn.keys[i].state_dict())
        keep.values[0].load_state_dict(attn.values[i].state_dict())
        part = keep(x)
        total = part if total is None else [a + b for a, b in zip(total, part)]

    assert _err(total, full, _flags(group)) < ATOL


def test_empty_attention_fiber_gives_uniform_weights(layer):
    r""":math:`M_{K} = 0` means every token attends equally, so the output is built from the token
    mean of the values."""
    group, dims_V, _ = layer
    torch.manual_seed(3)
    incl = ('D', group.n, 0) if isinstance(group, DihedralGroup) else (group.order(),)
    empty = BundleAttention(group, dims_V, [incl], [{}], [{'A1': 1} if isinstance(group, DihedralGroup) else {'A': 1}])
    x = random_irrep_tensors(group, (6,), dims_V)
    y = empty(x)
    for z in y:
        spread = float((z - z.mean(dim=0, keepdim=True)).detach().abs().max())
        assert spread < ATOL, "an empty attention fiber should give a token-independent output"


def test_empty_value_fiber_contributes_nothing(layer):
    group, dims_V, _ = layer
    torch.manual_seed(4)
    incl = ('D', group.n, 0) if isinstance(group, DihedralGroup) else (group.order(),)
    dead = BundleAttention(group, dims_V, [incl], [{}], [{}])
    x = random_irrep_tensors(group, (4,), dims_V)
    for z in dead(x):
        assert float(z.detach().abs().max()) == 0.0


# ------------------------------------------------------------------ reference implementation

def _real_layout(group, dims):
    return [(C, irrep.dim) for irrep, C in zip(group.real_irreps().values(), dims)]


def _vec_dim(group, dims):
    return sum(C * d for C, d in _real_layout(group, dims))


def _features_from_vec(group, dims, v):
    """A flat real vector back into the per-irrep list format."""
    out, offset = [], 0
    for (C, d), cplx in zip(_real_layout(group, dims), _flags(group)):
        chunk = v[..., offset:offset + C * d].reshape(*v.shape[:-1], C, d)
        offset += C * d
        out.append(EF.to_complex(chunk.unflatten(-1, (-1, 2))) if cplx else chunk)
    return out


def _vec_from_features(group, x):
    parts = []
    for z, cplx in zip(x, _flags(group)):
        r = EF.to_real(z).flatten(-2, -1) if cplx else z
        parts.append(r.flatten(-2, -1))
    return torch.cat(parts, dim=-1)


def _matrix_of(f, group, dims, out_group, out_dims):
    """Row-convention matrix: ``f(x) == x @ mat``, by feeding the standard basis in one batch."""
    n = _vec_dim(group, dims)
    y = f(_features_from_vec(group, dims, torch.eye(n)))
    return _vec_from_features(out_group, y).detach()


def test_matches_a_from_scratch_reference(layer):
    r"""Rebuild the layer as plain matrix algebra -- :math:`M_p`, :math:`R_p`, softmax -- and
    compare. This checks the assembled module against Theorem 1's formula rather than against
    its own pieces."""
    group, dims_V, attn = layer
    torch.manual_seed(5)
    L = 6
    x = random_irrep_tensors(group, (L,), dims_V)
    xv = _vec_from_features(group, x)                       # (L, n)

    got = _vec_from_features(group, attn(x)).detach()
    want = torch.zeros_like(got)

    for i in range(attn.head_set.num_orbits):
        K = attn.head_set.inclusions[i].source
        w_attn, w_value = attn.queries[i].dims_W, attn.values[i].dims_W
        attn_dim = _vec_dim(K, w_attn)
        if _vec_dim(K, w_value) == 0:
            continue

        Q = _matrix_of(attn.queries[i].down, group, dims_V, K, w_attn)     # (n, dim W)
        Kk = _matrix_of(attn.keys[i].down, group, dims_V, K, w_attn)
        D = _matrix_of(attn.values[i].down, group, dims_V, K, w_value)     # (n, dim W')
        U = _matrix_of(attn.values[i].up, K, w_value, group, dims_V)       # (dim W', n)

        for p, g in enumerate(attn.transports[i].transporters):
            A_inv = _matrix_of(lambda z, g=g: act_on_tensors(g.inv(), z), group, dims_V, group, dims_V)
            A = _matrix_of(lambda z, g=g: act_on_tensors(g, z), group, dims_V, group, dims_V)

            pulled = xv @ A_inv                                            # (L, n)
            scale = 1.0 / np.sqrt(attn_dim) if attn_dim > 0 else 1.0
            logits = (pulled @ Q) @ (pulled @ Kk).T * scale                # (L, L)
            weights = torch.softmax(logits, dim=-1)
            want = want + ((weights @ pulled) @ D) @ U @ A

    assert float((got - want).abs().max()) < 1e-3
