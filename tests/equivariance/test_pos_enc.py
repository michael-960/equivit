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
