import torch
import equivit
import pytest


groups = [
            equivit.geometry.CyclicGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ] + [
                equivit.geometry.DihedralGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ]


@pytest.mark.parametrize("group", groups)
def test_linear(group: equivit.geometry.Group):
    rng = torch.Generator()
    rng.manual_seed(43)

    irreps = group.real_irreps()

    dims_in = [torch.randint(4, 32, (1,)).item() for _ in irreps]
    dims_out = [torch.randint(4, 32, (1,)).item() for _ in irreps]

    linear = equivit.nn.EquivariantLinear(group, dims_in, dims_out)

    for _ in range(4): # test 4 random inputs
        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, (N,), dims_in, rng)
        y = linear(x)

        for g in group:
            x_t = equivit.act_on_tensors(g, x)

            y_t = linear(x_t)

            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < 1e-6, f"Error for irrep {irrep_name} and group element {g}: {error}"