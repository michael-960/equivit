import torch
import torch.nn as nn
import equivit
import numpy as np
import pytest


@pytest.mark.parametrize("group",
                         [equivit.geometry.CyclicGroup(N) for N in range(1, 40)]
                         + [equivit.geometry.DihedralGroup(N) for N in range(1, 40)]
                         )
def test_restrict(group: equivit.geometry.Group):
    rng = torch.Generator()
    rng.manual_seed(42)

    for subgroup_args in group.subgroups_up_to_conjugacy():
        incl = group.subgroup(*subgroup_args)
        subgroup = incl.source

        irreps = subgroup.real_irreps()

        res = equivit.nn.SymmetryRestriction(incl)

        common_shape = tuple([torch.randint(2, 8, (1,), generator=rng).item() for _ in range(3)])
        dims = [torch.randint(4, 64, (1,), generator=rng).item() for _ in range(len(group.real_irreps()))]

        x = equivit.random_irrep_tensors(group, common_shape, dims=dims, rng=rng)
        y = res(x)

        for g in subgroup:
            x_t = equivit.act_on_tensors(incl(g), x)

            y_t = res(x_t)
            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < 1e-6, f"Error for irrep {irrep_name} and group element {g}: {error}"
