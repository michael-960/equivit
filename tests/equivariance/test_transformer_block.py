import torch
import equivit
import pytest



groups = [
            equivit.geometry.CyclicGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ] + [
                equivit.geometry.DihedralGroup(n) for n in [1,2,3,4,5,6,8,10,12,13,17,31,32]
            ]


@pytest.mark.parametrize("group", groups)
def test_irrepwise_block(group: equivit.geometry.Group):
    ERROR_TOL = 1e-6 if group.order() < 20 else 1e-5
    rng = torch.Generator()
    rng.manual_seed(45)

    irreps = group.real_irreps()

    num_heads = [torch.randint(1, 8, (1,), generator=rng).item() for _ in irreps]

    dims = [h * torch.randint(1, 16, (1,), generator=rng).item() for h in num_heads]

    homog_copies = [torch.randint(3, 16, ()).item() for _ in group.all_homogeneous_space_actions()]

    block = equivit.nn.EquivariantTranformerBlock(
        group=group,
        dims=dims, 
        homogeneous_space_copies=homog_copies,
        num_heads=num_heads,
        attn_type='irrepwise',
    )

    for _ in range(4): # test 4 random inputs
        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, (N,), dims, rng)
        y = block(x)

        for g in group:
            x_t = equivit.act_on_tensors(g, x) 
            y_t = block(x_t)
            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < ERROR_TOL, f"Error for irrep {irrep_name} and group element {g}: {error}"


@pytest.mark.parametrize("group", groups)
def test_coupled_block(group: equivit.geometry.Group):
    ERROR_TOL = 1e-6 if group.order() < 20 else 1e-5
    rng = torch.Generator()
    rng.manual_seed(45)

    irreps = group.real_irreps()

    num_heads = torch.randint(1, 8, (1,), generator=rng).item()

    dims = [num_heads * torch.randint(1, 16, (1,), generator=rng).item() for _ in irreps]

    homog_copies = [torch.randint(3, 16, ()).item() for _ in group.all_homogeneous_space_actions()]

    block = equivit.nn.EquivariantTranformerBlock(
        group=group,
        dims=dims, 
        homogeneous_space_copies=homog_copies,
        num_heads=num_heads,
        attn_type='coupled',
    )

    for _ in range(4): # test 4 random inputs
        N = torch.randint(16, 96, (1,), generator=rng).item()
        x = equivit.random_irrep_tensors(group, (N,), dims, rng)
        y = block(x)

        for g in group:
            x_t = equivit.act_on_tensors(g, x) 
            y_t = block(x_t)
            y_t2 = equivit.act_on_tensors(g, y)

            for i, irrep_name in enumerate(irreps.keys()):
                error = (y_t2[i] - y_t[i]).abs().max()
                assert error < ERROR_TOL, f"Error for irrep {irrep_name} and group element {g}: {error}"


