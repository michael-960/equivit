import torch
import equivit
import pytest




groups = [
            equivit.geometry.CyclicGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ] + [
                equivit.geometry.DihedralGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ]


@pytest.mark.parametrize("group", groups)
def test_irrepwise_attn(group: equivit.geometry.Group):
    rng = torch.Generator()
    rng.manual_seed(45)

    irreps = group.real_irreps()

    num_heads = [torch.randint(1, 8, (1,), generator=rng).item() for _ in irreps]

    dims = [h * torch.randint(1, 16, (1,), generator=rng).item() for h in num_heads]

    attn = equivit.nn.EquivariantIrrepwiseAttention(
        group, dims=dims, num_heads=num_heads,
        attn_drop=0., proj_drop=0.,
        trivial_rep_attn_bias=True, trivial_rep_proj_bias=True
    )


    for _ in range(4): # test 4 random inputs
        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, (N,), dims, rng)
        y = attn(x)

        for g in group:
            x_t = equivit.act_on_tensors(g, x) 
            y_t = attn(x_t)
            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < 1e-6, f"Error for irrep {irrep_name} and group element {g}: {error}"


@pytest.mark.parametrize("group", groups)
def test_coupled_attn(group: equivit.geometry.Group):
    rng = torch.Generator()
    rng.manual_seed(45)

    irreps = group.real_irreps()

    num_heads = torch.randint(1, 8, (1,), generator=rng).item()

    dims = [num_heads * torch.randint(1, 16, (1,), generator=rng).item() for _ in range(len(irreps))]

    attn = equivit.nn.EquivariantCoupledAttention(
        group, dims=dims, num_heads=num_heads,
        attn_drop=0., proj_drop=0.,
        trivial_rep_attn_bias=True, trivial_rep_proj_bias=True
    )


    for _ in range(4): # test 4 random inputs
        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, (N,), dims, rng)
        y = attn(x)

        for g in group:
            x_t = equivit.act_on_tensors(g, x) 
            y_t = attn(x_t)
            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < 1e-6, f"Error for irrep {irrep_name} and group element {g}: {error}"
