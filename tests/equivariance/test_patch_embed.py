import torch
import torch.nn as nn
import equivit
import numpy as np
import pytest


@pytest.mark.parametrize("lattice", 
                         [equivit.geometry.Square(N) for N in range(2, 20)]
                         + [equivit.geometry.Hexagon(N) for N in range(2, 20)]
                         + [equivit.geometry.Honeycomb(N) for N in range(2, 20)]
                         + [equivit.geometry.Triangle(N) for N in range(2, 10)]
                         )
def test_patch_embed(lattice: equivit.geometry.Lattice):
    """
    Test whether PatchEmbed is equivariant.
    """

    rng = torch.Generator()
    rng.manual_seed(42)

    # lattice = equivit.geometry.Square(4)
    action = lattice.action
    group = action.group

    subgroup_args = group.subgroups_up_to_conjugacy()

    for subgroup_arg in subgroup_args:
        subgroup_incl = group.subgroup(*subgroup_arg)
        subgroup = subgroup_incl.source
        subgroup_action = action.pullback(subgroup_incl)

        irreps = subgroup.real_irreps()

        in_channels = torch.randint(2, 8, (1,), generator=rng).item()
        out_channels = [torch.randint(2, 16, (1,), generator=rng).item() for _ in range(len(irreps))]

        patch_embed = equivit.nn.EquivariantPatchEmbed(subgroup_action, in_channels=in_channels, out_channels=out_channels)

        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = torch.rand((N, patch_embed.L, in_channels), generator=rng)

        for g in subgroup:
            x_t = x[:,subgroup_action(g.inv()),:]

            y = patch_embed(x)
            y_t = patch_embed(x_t)

            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < 1e-5, f"Error for subgroup {subgroup}, irrep {irrep_name} and group element {g}: {error}"
 