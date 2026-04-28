import equivit
import pytest
import torch


# TODO: change this

@pytest.mark.parametrize('N2', [2,4,6,7,8,15])
@pytest.mark.parametrize('N1', [2,3,4,6,7,8])
@pytest.mark.parametrize('subgroup_args', equivit.D6.subgroups_up_to_conjugacy())
@pytest.mark.parametrize('attn_type', ['irrepwise', 'coupled'])
def test_octic_backbone(
    N1: int, N2: int, 
    subgroup_args: tuple, 
    attn_type: str
) -> None:
    """
    Test whether HoneyViTBackbone is equivariant.

    Args:
        subgroup_args: arguments specifying the subgroup of D6 that the backbone is equivariant to
        attn_type: type of attention to use in the transformer blocks ('irrepwise' or 'coupled')
    """

    rng = torch.Generator()
    rng.manual_seed(43)


    group_incl = equivit.geometry.D6.subgroup(*subgroup_args)
    group = group_incl.source


    homog_copies = [torch.randint(1, 7, (1,), generator=rng).item() for _ in group.all_homogeneous_space_actions()]

    if attn_type == 'irrepwise':
        num_heads = [torch.randint(1, 8, (1,), generator=rng).item() for _ in group.real_irreps()]
        dims = [h * torch.randint(1, 16, (1,), generator=rng).item() for h in num_heads]

    else:
        num_heads = torch.randint(1, 8, (1,), generator=rng).item()
        dims = [torch.randint(1, 16, (1,), generator=rng).item() * num_heads for _ in group.real_irreps()]


    config = equivit.models.HoneyViTBackboneConfig(
        N1=N1,
        N2=N2,
        dims=dims,
        subgroup=subgroup_args,
        in_channels=torch.randint(1, 4, (1,), generator=rng).item(),
        depth=torch.randint(2, 14, (1,), generator=rng).item(),
        transformer_block_config=equivit.nn.EquivariantTransformerBlockConfig(
                attn_type=attn_type,
                homogeneous_space_copies=homog_copies,
                num_heads=num_heads

        )
    )

    tol = config.depth * 8e-6

    hex_img = equivit.geometry.HexPatches(N1, N2)

    backbone = equivit.models.HoneyViTBackbone(config)

    batch_size = torch.randint(1, 4, (1,), generator=rng).item()

    img = torch.rand((batch_size, config.in_channels, hex_img.L), generator=rng)

    y = backbone(img)

    for g in group:
        
        img_t = img[..., hex_img.action(group_incl(g).inv())]
        y_t = backbone(img_t)

        y_t2 = equivit.act_on_tensors(g, y)

        action_permutation = hex_img.hex1.action(group_incl(g).inv()) + [hex_img.hex1.L]  # add index for class token, which is fixed by the group action
        y_t2 = [z[...,action_permutation ,:,:] for z in y_t2]

        for i, irrep_name in enumerate(group.real_irreps().keys()):
            error = (y_t2[i] - y_t[i]).abs().max()
            assert error < tol, f"Error for subgroup {group}, irrep {irrep_name} and group element {g}: {error} (depth={config.depth}, attn_type={attn_type}, num_heads={num_heads}, dims={dims}, N1={N1}, N2={N2})"
