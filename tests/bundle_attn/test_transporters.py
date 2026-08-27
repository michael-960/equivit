r"""
Checks for :func:`head_transporters`.

The last two are the ones that matter: they verify that the family
:math:`M_{(i,p)} = \rho(g_{i,p}) M_{K_i} \rho(g_{i,p})^{-1}` built from a :math:`K_i`-invariant
:math:`M_{K_i}` is well defined and satisfies :math:`M_{gp} = \rho(g) M_p \rho(g)^{-1}`, which is
the equivariance condition Theorem 1 imposes on the attention forms.
"""
import numpy as np
import pytest

from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup
from equivit.nn.bundle_attn import build_head_set, head_transporters


CASES = [
    ("D4-mixed", DihedralGroup(4), [('D', 2, 0), ('C', 2), ('D', 4, 0)]),
    ("D4-all", DihedralGroup(4), DihedralGroup(4).subgroups_up_to_conjugacy()),
    ("D4-regular", DihedralGroup(4), [('C', 1)]),
    ("D4-repeat", DihedralGroup(4), [('C', 2), ('C', 2)]),
    ("D3", DihedralGroup(3), [('D', 1, 0), ('C', 3)]),
    ("C6", CyclicGroup(6), [(2,), (3,), (1,)]),
    ("C4-point", CyclicGroup(4), [(4,)]),
]

IDS = [c[0] for c in CASES]
PARAMS = [(c[1], c[2]) for c in CASES]


@pytest.fixture(params=PARAMS, ids=IDS)
def head_set(request):
    group, subgroups = request.param
    return build_head_set(group, subgroups)


@pytest.fixture
def transporters(head_set):
    return head_transporters(head_set)


def _subgroup_image(inclusion):
    return set(inclusion(k) for k in inclusion.source)


def _regular_representation(group):
    return group.regular_action().to_linear_representation()


def _invariant_form(rho, inclusion, seed):
    r"""A random :math:`K`-invariant :math:`M`: :math:`\rho(k) M \rho(k)^{-1} = M` for all k in K."""
    rng = np.random.default_rng(seed)
    d = rho(inclusion.target.identity()).shape[0]
    m0 = rng.standard_normal((d, d))
    image = _subgroup_image(inclusion)
    return sum(rho(k) @ m0 @ rho(k).T for k in image) / len(image)


def test_transporter_moves_base_point_to_the_head(head_set, transporters):
    for i in range(head_set.num_orbits):
        base = head_set.base_point(i)
        for p, g in enumerate(transporters[i]):
            assert head_set.action(g)[base] == head_set.flat(i, p)


def test_base_point_transporter_lies_in_K(head_set, transporters):
    for i, inclusion in enumerate(head_set.inclusions):
        assert transporters[i][0] in _subgroup_image(inclusion)


def test_transporters_are_a_transversal(head_set, transporters):
    for i, inclusion in enumerate(head_set.inclusions):
        image = _subgroup_image(inclusion)
        cosets = [frozenset(g * k for k in image) for g in transporters[i]]
        assert len(set(cosets)) == len(cosets), "two transporters lie in the same coset"
        assert set().union(*cosets) == set(head_set.group), "the cosets do not cover G"


def test_cocycle_lands_in_K(head_set, transporters):
    r""":math:`g_q^{-1} g\, g_p \in K_i` whenever :math:`q = g \cdot p`. This is what makes the
    conjugated form independent of the transporter choice."""
    for i, inclusion in enumerate(head_set.inclusions):
        image = _subgroup_image(inclusion)
        for g in head_set.group:
            for p, g_p in enumerate(transporters[i]):
                _, q = head_set.structured(head_set.action(g)[head_set.flat(i, p)])
                assert transporters[i][q].inv() * g * g_p in image


def test_conjugation_is_independent_of_the_representative(head_set, transporters):
    group = head_set.group
    rho = _regular_representation(group)
    for i, inclusion in enumerate(head_set.inclusions):
        m = _invariant_form(rho, inclusion, seed=100 + i)
        for p, g_p in enumerate(transporters[i]):
            reference = rho(g_p) @ m @ rho(g_p).T
            for k in _subgroup_image(inclusion):
                other = rho(g_p * k) @ m @ rho(g_p * k).T
                assert np.allclose(other, reference, atol=1e-12)


def test_induced_family_is_equivariant(head_set, transporters):
    r""":math:`M_{g\cdot p} = \rho(g) M_p \rho(g)^{-1}` -- Theorem 1's condition on the attention
    forms, checked over the whole group and every head."""
    group = head_set.group
    rho = _regular_representation(group)
    for i, inclusion in enumerate(head_set.inclusions):
        m = _invariant_form(rho, inclusion, seed=200 + i)
        forms = [rho(g_p) @ m @ rho(g_p).T for g_p in transporters[i]]
        for g in group:
            for p in range(len(forms)):
                _, q = head_set.structured(head_set.action(g)[head_set.flat(i, p)])
                assert np.allclose(forms[q], rho(g) @ forms[p] @ rho(g).T, atol=1e-12)


def test_invariant_form_is_actually_invariant(head_set):
    """Guards the two tests above: a trivial M would make them vacuous."""
    group = head_set.group
    rho = _regular_representation(group)
    for i, inclusion in enumerate(head_set.inclusions):
        m = _invariant_form(rho, inclusion, seed=300 + i)
        assert not np.allclose(m, 0.0)
        for k in _subgroup_image(inclusion):
            assert np.allclose(rho(k) @ m @ rho(k).T, m, atol=1e-12)
