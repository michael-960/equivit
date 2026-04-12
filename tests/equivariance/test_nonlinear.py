import torch
import torch.nn as nn
import equivit
import numpy as np
import pytest

groups = [
            equivit.geometry.CyclicGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ] + [
                equivit.geometry.DihedralGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ]

activations = [nn.ReLU(), nn.GELU(), nn.Sigmoid(), nn.ELU()]


@pytest.mark.parametrize("group", groups)
@pytest.mark.parametrize("activation", activations)
def test_nonlinear(group, activation):

    # Let's be lenient with large groups
    ERROR_TOL = 1e-6 if group.order() < 20 else 1e-5

    rng = torch.Generator()
    rng.manual_seed(42)

    irreps = group.real_irreps()

    homog_actions = group.all_homogeneous_space_actions()
    num_homog_spaces = len(homog_actions)
    num_homog_copies = [torch.randint(1, 7, (1,), generator=rng).item() for _ in range(num_homog_spaces)]

    nonlin = equivit.nn.EquivariantNonlinear(group, num_homog_copies, activation=activation)

    dims = [sum(_) for _ in nonlin.split_sizes]

    for _ in range(4): # test 4 random inputs
        N = torch.randint(32, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, shape=(N,), dims=dims, rng=rng)
        y = nonlin(x)

        for g in group:
            # transformed input
            x_t = equivit.act_on_tensors(g, x)

            y_t = nonlin(x_t)

            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < ERROR_TOL, f"Error for irrep {irrep_name} and group element {g}: {error}"

