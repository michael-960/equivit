"""Sanity checks for SymmetryCorestriction against SymmetryRestriction.

Run with pytest, or directly:  python3 test_corestriction.py
"""
import numpy as np
import torch

from equivit.geometry import IrrepType
from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn import functional as EF
from equivit.nn.restriction import SymmetryRestriction, SymmetryCorestriction


ATOL = 1e-4   # the buffers of both modules are float32


# ---------------------------------------------------------------- helpers

def _flags(group):
    return [irrep.rep_type is IrrepType.COMPLEX for irrep in group.real_irreps().values()]


def _random_features(group, dims, batch=(2, 3)):
    """A random element of $\\bigoplus_i \\mathbb{R}^{C_i}\\otimes V_i$, in the list format."""
    out = []
    for irrep, C in zip(group.real_irreps().values(), dims):
        r = torch.randn(*batch, C, irrep.dim)
        if irrep.rep_type is IrrepType.COMPLEX:
            r = EF.to_complex(r.unflatten(-1, (-1, 2)))
        out.append(r)
    return out


def _act(group, feats, g):
    """Apply the group element g to a feature list."""
    out = []
    for irrep, z in zip(group.real_irreps().values(), feats):
        M = torch.from_numpy(np.asarray(irrep(g), dtype=np.float32))
        cplx = irrep.rep_type is IrrepType.COMPLEX
        r = EF.to_real(z).flatten(-2, -1) if cplx else z
        r = r @ M.permute(1, 0)
        if cplx:
            r = EF.to_complex(r.unflatten(-1, (-1, 2)))
        out.append(r)
    return out


def _inner(a, b, flags):
    """The real inner product, with complex irreps read as real 2-vectors."""
    total = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za = EF.to_real(za).flatten(-2, -1)
            zb = EF.to_real(zb).flatten(-2, -1)
        total += float((za * zb).sum())
    return total


def _max_err(a, b, flags):
    worst = 0.0
    for za, zb, cplx in zip(a, b, flags):
        if cplx:
            za = EF.to_real(za).flatten(-2, -1)
            zb = EF.to_real(zb).flatten(-2, -1)
        worst = max(worst, float(np.abs(np.asarray(za) - np.asarray(zb)).max()))
    return worst


# group, subgroup args, channel widths (one per irrep of the group)
CASES = [
    (DihedralGroup(4), ('C', 1),    [2, 1, 3, 1, 2]),   # trivial subgroup
    (DihedralGroup(4), ('C', 2),    [2, 1, 3, 1, 2]),   # <r^2>
    (DihedralGroup(4), ('C', 4),    [1, 2, 1, 1, 2]),   # rotations
    (DihedralGroup(4), ('D', 1, 0), [2, 1, 3, 1, 2]),   # a reflection
    (DihedralGroup(4), ('D', 1, 1), [1, 1, 1, 1, 1]),   # the other reflection class
    (DihedralGroup(4), ('D', 2, 0), [2, 2, 1, 1, 2]),
    (DihedralGroup(4), ('D', 2, 1), [1, 1, 2, 2, 1]),
    (DihedralGroup(4), ('D', 4, 0), [2, 1, 1, 1, 3]),   # the whole group
    (DihedralGroup(3), ('C', 3),    [2, 1, 2]),
    (DihedralGroup(3), ('D', 1, 0), [1, 2, 2]),
    (CyclicGroup(4),   (2,),        [1, 2, 2]),         # complex-type irreps
    (CyclicGroup(6),   (3,),        [2, 1, 1, 2]),
    (CyclicGroup(6),   (2,),        [1, 1, 2, 1]),
]


def _build(group, args, dims):
    incl = group.subgroup(*args)
    return incl, SymmetryRestriction(incl), SymmetryCorestriction(incl, dims)


# ---------------------------------------------------------------- tests

def test_shapes():
    """Restriction and corestriction round-trip the declared shapes."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        x = _random_features(group, dims)
        y = res(x)
        assert len(y) == len(incl.source.real_irreps()), (group, args)
        back = cores(y)
        assert len(back) == len(dims), (group, args)
        for xi, bi in zip(x, back):
            assert xi.shape == bi.shape, (group, args, xi.shape, bi.shape)


def test_corestriction_inverts_restriction():
    """cores(res(x)) == x, i.e. the corestriction is a left inverse."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        flags = _flags(group)
        x = _random_features(group, dims)
        err = _max_err(cores(res(x)), x, flags)
        assert err < ATOL, f"{group} / {args}: left-inverse error {err:.2e}"


def test_restriction_inverts_corestriction():
    """res(cores(y)) == y, i.e. it is a right inverse too (the map is a bijection)."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        K = incl.source
        kflags = [irrep.rep_type is IrrepType.COMPLEX for irrep in K.real_irreps().values()]
        x = _random_features(group, dims)
        y = res(x)                       # a valid element of the K-side space
        y2 = torch_like_randomize(y)
        err = _max_err(res(cores(y2)), y2, kflags)
        assert err < ATOL, f"{group} / {args}: right-inverse error {err:.2e}"


def torch_like_randomize(y):
    """Fresh random tensors with the same shapes/dtypes as y."""
    out = []
    for z in y:
        r = torch.randn(*z.shape[:-1], z.shape[-1] * (2 if torch.is_complex(z) else 1))
        if torch.is_complex(z):
            r = EF.to_complex(r.unflatten(-1, (-1, 2)))
        out.append(r)
    return out


def test_adjointness():
    """<cores(y), x> == <y, res(x)> -- the corestriction is the adjoint, not merely an inverse."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        K = incl.source
        gflags = _flags(group)
        kflags = [irrep.rep_type is IrrepType.COMPLEX for irrep in K.real_irreps().values()]
        x = _random_features(group, dims)
        y = torch_like_randomize(res(x))
        lhs = _inner(cores(y), x, gflags)
        rhs = _inner(y, res(x), kflags)
        assert abs(lhs - rhs) < ATOL * max(1.0, abs(rhs)), \
            f"{group} / {args}: <cores(y),x>={lhs:.6f} vs <y,res(x)>={rhs:.6f}"


def test_restriction_is_equivariant():
    """res(phi(k) . x) == k . res(x) for every k in K."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        K = incl.source
        kflags = [irrep.rep_type is IrrepType.COMPLEX for irrep in K.real_irreps().values()]
        x = _random_features(group, dims)
        for k in K:
            lhs = res(_act(group, x, incl(k)))
            rhs = _act(K, res(x), k)
            err = _max_err(lhs, rhs, kflags)
            assert err < ATOL, f"{group} / {args}, k={k}: restriction equivariance error {err:.2e}"


def test_corestriction_is_equivariant():
    """cores(k . y) == phi(k) . cores(y) for every k in K."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        K = incl.source
        gflags = _flags(group)
        x = _random_features(group, dims)
        y = torch_like_randomize(res(x))
        for k in K:
            lhs = cores(_act(K, y, k))
            rhs = _act(group, cores(y), incl(k))
            err = _max_err(lhs, rhs, gflags)
            assert err < ATOL, f"{group} / {args}, k={k}: corestriction equivariance error {err:.2e}"


def test_buffers_are_transposes():
    """The corestriction's buffers are exactly the transposes of the restriction's."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        n_G = len(group.real_irreps())
        n_K = len(incl.source.real_irreps())
        for i in range(n_G):
            for j in range(n_K):
                a = np.asarray(res.get_buffer(f'res_{i}_{j}'))
                b = np.asarray(cores.get_buffer(f'cores_{i}_{j}'))
                assert a.shape == b.T.shape, (group, args, i, j, a.shape, b.shape)
                assert np.allclose(a, b.T, atol=1e-6), (group, args, i, j)


def test_isometry_on_the_whole_space():
    """The map is orthogonal, so it preserves the norm."""
    for group, args, dims in CASES:
        incl, res, cores = _build(group, args, dims)
        K = incl.source
        gflags = _flags(group)
        kflags = [irrep.rep_type is IrrepType.COMPLEX for irrep in K.real_irreps().values()]
        x = _random_features(group, dims)
        nx = _inner(x, x, gflags)
        ny = _inner(res(x), res(x), kflags)
        assert abs(nx - ny) < ATOL * max(1.0, nx), f"{group} / {args}: |x|^2={nx:.6f} vs |res(x)|^2={ny:.6f}"


TESTS = [test_shapes,
         test_corestriction_inverts_restriction,
         test_restriction_inverts_corestriction,
         test_adjointness,
         test_restriction_is_equivariant,
         test_corestriction_is_equivariant,
         test_buffers_are_transposes,
         test_isometry_on_the_whole_space]


if __name__ == '__main__':
    torch.manual_seed(0)
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'  PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {t.__name__}: {e}')
    print()
    print('ALL CHECKS PASSED' if not failed else f'{failed} FAILED')
    raise SystemExit(1 if failed else 0)
