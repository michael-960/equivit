import torch
import torch.nn as nn
import equivit
import numpy as np
import pytest

groups = [
            equivit.geometry.CyclicGroup(n) for n in [1,2,3,4,5,6,8,10,12]
            ] + [
                equivit.geometry.DihedralGroup(n) for n in [1,2,3,4,5,6,8,10,12]
            ]

activations = [nn.ReLU(), nn.GELU(), nn.Sigmoid(), nn.ELU()]


@pytest.mark.parametrize("group", groups)
@pytest.mark.parametrize("activation", activations)
def test_nonlinear(group, activation):
    rng = torch.Generator()
    rng.manual_seed(42)

    irreps = group.real_irreps()

    num_irreps = len(irreps)

    homog_actions = group.all_homogeneous_space_actions()

    num_homog_spaces = len(homog_actions)

    num_homog_copies = [torch.randint(1, 7, (1,), generator=rng).item() for _ in range(num_homog_spaces)]

    nonlin = equivit.nn.EquivariantNonlinear(group, num_homog_copies, activation=activation)

    dims = [sum(_) for _ in nonlin.split_sizes]

    x = [torch.rand((64, dim, irrep.dim), generator=rng) for (dim, irrep) in zip(dims, irreps.values())]

    for g in group:
        # transformed input
        x_t = [torch.einsum('ij,...j -> ...i', torch.tensor(irrep(g)).to(torch.float32), z) for irrep, z in zip(irreps.values(), x)]

        y = nonlin(x)
        y_t = nonlin(x_t)

        for i, (name, irrep) in enumerate(irreps.items()):
            error = (torch.einsum('ij,...j->...i', torch.tensor(irrep(g)).to(torch.float32),  y[i]) - y_t[i]).abs().max()
            assert error < 1e-6, f"Error for irrep {name} and group element {g}: {error}"
