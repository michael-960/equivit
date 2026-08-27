r"""
Checks for :func:`build_head_set` -- the head set H = \bigsqcup_i G/K_i.

Run with:

    PYTHONPATH=<equivit>/src python3 test_head_set.py

or under pytest.
"""
import pytest
import itertools

import numpy as np

from equivit.geometry.groups.cyclic import CyclicGroup
from equivit.geometry.groups.dihedral import DihedralGroup

from equivit.nn.bundle_attn import build_head_set


# (name, group, list of subgroup argument tuples)
CASES = [
    ("D4 / [D2_0, C2, D4]",     DihedralGroup(4), [('D', 2, 0), ('C', 2), ('D', 4, 0)]),
    ("D4 / all 8 classes",      DihedralGroup(4), DihedralGroup(4).subgroups_up_to_conjugacy()),
    ("D4 / [C1] (regular)",     DihedralGroup(4), [('C', 1)]),
    ("D4 / [C2, C2] (repeat)",  DihedralGroup(4), [('C', 2), ('C', 2)]),
    ("D3 / [D1_0, C3]",         DihedralGroup(3), [('D', 1, 0), ('C', 3)]),
    ("C6 / [C2, C3, C1]",       CyclicGroup(6),   [(2,), (3,), (1,)]),
    ("C6 / all classes",        CyclicGroup(6),   CyclicGroup(6).subgroups_up_to_conjugacy()),
    ("C4 / [C4] (one point)",   CyclicGroup(4),   [(4,)]),
]


def _stabilizer(action, x):
    return [g for g in action.group if action(g)[x] == x]

@pytest.mark.parametrize('name,G,specs', CASES)
def test_size_is_sum_of_indices(name, G, specs):
    H = build_head_set(G, specs)
    expected = sum(G.order() // incl.source.order() for incl in H.inclusions)
    assert len(H) == expected, f"{name}: |H| = {len(H)}, expected {expected}"
    assert len(H) == H.action.num_elements


@pytest.mark.parametrize('name,G,specs', CASES)
def test_action_is_valid(name, G, specs):
    H = build_head_set(G, specs)
    H.action.validate()


@pytest.mark.parametrize('name,G,specs', CASES)
def test_blocks_partition_the_set(name, G, specs):
    H = build_head_set(G, specs)
    flat = [h for block in H.blocks for h in block]
    assert flat == list(range(len(H))), f"{name}: blocks are not a contiguous partition"


@pytest.mark.parametrize('name,G,specs', CASES)
def test_blocks_are_the_orbits(name, G, specs):
    H = build_head_set(G, specs)
    got = sorted(sorted(o) for o in H.action.orbits())
    want = sorted(sorted(b) for b in H.blocks)
    assert got == want, f"{name}: orbits {got} != blocks {want}"


@pytest.mark.parametrize('name,G,specs', CASES)
def test_stabilizer_of_base_point_is_K(name, G, specs):
    H = build_head_set(G, specs)
    for i, incl in enumerate(H.inclusions):
        stab = set(_stabilizer(H.action, H.base_point(i)))
        image = set(incl(k) for k in incl.source)
        assert stab == image, (
            f"{name}: orbit {i} stabilizer of base point has order {len(stab)}, "
            f"K_i image has order {len(image)}"
        )


@pytest.mark.parametrize('name,G,specs', CASES)
def test_every_stabilizer_is_conjugate_to_K(name, G, specs):
    H = build_head_set(G, specs)
    for i, incl in enumerate(H.inclusions):
        image = set(incl(k) for k in incl.source)
        for p, h in enumerate(H.blocks[i]):
            stab = set(_stabilizer(H.action, h))
            assert any(set(g * k * g.inv() for k in image) == stab for g in G), (
                f"{name}: stabilizer of head ({i},{p}) is not conjugate to K_{i}"
            )


@pytest.mark.parametrize('name,G,specs', CASES)
def test_index_round_trip(name, G, specs):
    H = build_head_set(G, specs)
    for h in range(len(H)):
        i, p = H.structured(h)
        assert H.flat(i, p) == h, f"{name}: round trip failed at h = {h}"
    for i, block in enumerate(H.blocks):
        for p in range(len(block)):
            assert H.structured(H.flat(i, p)) == (i, p)


@pytest.mark.parametrize('name,G,specs', CASES)
def test_orbit_action_matches_homogeneous_space(name, G, specs):
    H = build_head_set(G, specs)
    for i, incl in enumerate(H.inclusions):
        restricted = H.orbit_action(i)
        direct = G.homogeneous_space_action(incl)
        for g in G:
            assert list(restricted(g)) == list(direct(g)), (
                f"{name}: orbit {i} restricted action differs from G/K_{i} at g = {g}"
            )


@pytest.mark.parametrize('name,G,specs', CASES)
def test_structured_and_flat_agree_on_the_action(name, G, specs):
    """Acting flatly, then splitting, is acting inside the orbit: g.(i,p) = (i, g.p)."""
    H = build_head_set(G, specs)
    for g in G:
        row = H.action(g)
        for i in range(H.num_orbits):
            orbit_row = H.orbit_action(i)(g)
            for p, h in enumerate(H.blocks[i]):
                assert H.structured(row[h]) == (i, orbit_row[p]), (
                    f"{name}: g = {g} moves head ({i},{p}) out of its orbit block"
                )


@pytest.mark.parametrize('name,G,specs', CASES)
def test_order_of_subgroups_only_shifts_offsets(name, G, specs):
    """Permuting the subgroup list permutes the blocks and nothing else."""
    H = build_head_set(G, specs)
    if H.num_orbits < 2:
        return
    perm = list(range(H.num_orbits))[::-1]
    H2 = build_head_set(G, [specs[i] for i in perm])
    assert len(H2) == len(H)
    for j, i in enumerate(perm):
        a1, a2 = H.orbit_action(i), H2.orbit_action(j)
        for g in G:
            assert list(a1(g)) == list(a2(g)), f"{name}: orbit {i} changed under reordering"


@pytest.mark.parametrize('name,G,specs', CASES)
def test_accepts_inclusions_as_well_as_tuples(name, G, specs):
    H = build_head_set(G, specs)
    H2 = build_head_set(G, [G.subgroup(*s) if isinstance(s, tuple) else s for s in specs])
    for g in G:
        assert list(H2.action(g)) == list(H.action(g)), f"{name}: inclusion input differs"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    failures = 0
    for name, G, specs in CASES:
        H = build_head_set(G, specs)
        print(f"\n{name}:  |H| = {len(H)}, blocks = {[len(b) for b in H.blocks]}")
        for t in TESTS:
            try:
                t(name, G, specs, H)
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {t.__name__}: {e}")
            else:
                print(f"  PASS  {t.__name__}")
    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
