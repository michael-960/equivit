import torch
import equivit
import pytest


@pytest.mark.parametrize("lattice", 
                         [equivit.geometry.Square(N) for N in range(2, 20)]
                         + [equivit.geometry.Hexagon(N) for N in range(2, 20)]
                         + [equivit.geometry.Honeycomb(N) for N in range(2, 20)]
                         + [equivit.geometry.Triangle(N) for N in range(2, 10)]
                         )
def test_posenc(lattice):
    """
    Test whether EquivariantPositionalEncoding is equivariant.
    To be more precise, we test whether the positional encoding vectors themselves are invariant under the group action.
    """

    rng = torch.Generator()
    rng.manual_seed(42)

    action = lattice.action
    group = action.group

    subgroup_args = group.subgroups_up_to_conjugacy()

    for subgroup_arg in subgroup_args:
        subgroup_incl = group.subgroup(*subgroup_arg)
        subgroup = subgroup_incl.source
        subgroup_action = action.pullback(subgroup_incl)

        irreps = subgroup.real_irreps()

        dims = [torch.randint(2, 16, (1,), generator=rng).item() for _ in range(len(irreps))]

        posenc = equivit.nn.EquivariantPositionalEncoding(subgroup_action, dims=dims)

        x = posenc.get_positional_encodings()

        for g in subgroup:
            z = equivit.act_on_tensors(g, [y[subgroup_action(g.inv())] for y in x]) # should be the same as x

            for i, irrep_name in enumerate(irreps.keys()):
                error = (z[i] - x[i]).abs().max()
                assert error < 1e-6, f"Error for subgroup {subgroup}, irrep {irrep_name} and group element {g}: {error}"

D6 = equivit.geometry.DihedralGroup(6)
cosets = D6.left_cosets(D6.subgroup('D', 3, 0))

@pytest.mark.parametrize("lattice", 
                         [equivit.geometry.Honeycomb(N) for N in range(2, 35)]
                         )
def test_induced_posenc(lattice):
    """
    Test whether EquivariantPositionalEncoding is equivariant.
    To be more precise, we test whether the positional encoding vectors themselves are invariant under the group action.
    """

    rng = torch.Generator()
    rng.manual_seed(42)

    action = lattice.action
    group = action.group

    r1 = cosets[0][torch.randint(0, len(cosets[0]), (1,), generator=rng).item()]
    r2 = cosets[1][torch.randint(0, len(cosets[1]), (1,), generator=rng).item()]

    bundle = equivit.geometry.EquivariantPullbackBundle(
                action,
                subgroup_args=('D', 3, 0),
                representatives=[r1, r2],
                basepoints=[0] * len(action.orbits())
            )

    irreps = bundle.subgroup.real_irreps()

    dims = [torch.randint(2, 16, (1,), generator=rng).item() for _ in range(len(irreps))]

    posenc = equivit.nn.EquivariantInducedPositionalEncoding(bundle, dims=dims)

    x = posenc.get_positional_encodings()

    for g in group:
        z = equivit.induced_action_on_tensors(bundle, g, x) # should be the same as x

        for i, irrep_name in enumerate(irreps.keys()):
            error = (z[i] - x[i]).abs().max()
            assert error < 1e-6, f"Error for irrep {irrep_name} and group element {g}: {error}"

