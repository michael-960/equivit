import torch
from typing import List, Optional, Callable
from ..geometry import Group, IrrepType, GroupElement



# We have this only because we don't know how to deal with quaternionic irreps yet.
# TODO: support quaternionic irreps in the future
def assert_all_not_quaternionic(group: Group):
    """
    Assert that all irreps of the group are not quaternionic.

    Args:
        group: a group
    """
    for irrep in group.real_irreps().values():
        if irrep.rep_type is IrrepType.QUATERNIONIC:
            raise NotImplementedError("Quaternion-type irreps are not supported yet.")

def act_on_tensors(g: GroupElement, x: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply the group element `g` to the list of tensors `x`.

    This function is for testing purposes only, to verify the equivariance of our layers. It is not optimized for speed.

    Args:
        g: a group element
        x: a list of tensors, each of shape (*, Ci, di) for each irrep, where Ci
        is the number of channels for that irrep and di is the (complex)
        dimension of that irrep.

    Returns:
        a list of tensors, each of shape (*, Ci, di) for each irrep 
    """
    assert len(g.group.real_irreps()) == len(x), "Length of input list should match number of irreps"

    assert_all_not_quaternionic(g.group)

    irreps = g.group.real_irreps()

    x_t = []
    for i, irrep in enumerate(irreps.values()):
        dtype = x[i].dtype
        if irrep.rep_type is IrrepType.REAL:
            assert dtype in [torch.float32, torch.float64], f"Expected real dtype for irrep {irrep.name}, but got {dtype}"
            x_t.append(torch.einsum('ij,...j -> ...i', torch.tensor(irrep(g)).to(dtype), x[i]))
        else:
            assert dtype in [torch.complex64, torch.complex128], f"Expected complex dtype for irrep {irrep.name}, but got {dtype}"
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            x_t.append(torch.einsum('ij,...j -> ...i', 
                                    torch.tensor(irrep(g)).to(real_dtype), 
                                    x[i].view(real_dtype)
                                    ).contiguous().view(dtype))
    return x_t



def random_irrep_tensors(group: Group, shape, dims: List[int], rng: Optional[torch.Generator]=None) -> List[torch.Tensor]:
    """
    Generate a list of random tensors, each of shape (*shape, Ci, di) for each irrep,
    where Ci is the number of channels (specified by dims) for that irrep and di is the (complex) dimension of that irrep.

    Args:
        group: a group
        shape: the leading shape of the output tensors (before the channel and irrep dimensions)
        dims: list of number of channels for each irrep
        rng: an optional torch.Generator for reproducibility
    """
    x = []
    irreps = group.real_irreps()
    for i, irrep in enumerate(irreps.values()):
        if irrep.rep_type is IrrepType.REAL:
            x.append(torch.rand((*shape, dims[i], irrep.dim), generator=rng))
        else:
            x.append(torch.rand((*shape, dims[i], irrep.dim//2), generator=rng, dtype=torch.complex64))

    return x


def equivariance_error(
    func: Callable, x: List[torch.Tensor], g: GroupElement,
     error_norm_p=torch.inf
) -> float:
    """
    Computes func(g.x) - g.func(x) and returns the norm of the error for each irrep.

    Args:
        func: a function that takes in a list of tensors (one for each irrep) and returns a list of tensors (one for each irrep)
        x: a list of tensors, each of shape (*, Ci, di) for each irrep, where Ci is the number of channels for that irrep and di is the (complex) dimension of that irrep.
        g: a group element
        error_norm_p: the p value for the norm of the error (default is infinity norm)

    Returns:
        a list of errors for each irrep
    """
    y_t = func(act_on_tensors(g, x))
    y_t2 = act_on_tensors(g, func(x))

    errors = []
    for i in range(len(y_t)):
        errors.append((y_t[i] - y_t2[i]).norm(p=error_norm_p).item())
    return errors


def equivariance_error_over_group(
    func: Callable, x: List[torch.Tensor], 
    group: Group, 
    error_norm_p=torch.inf
) -> List[float]:
    """
    Computes the equivariance error of func over all elements in the group and returns the average error for each irrep.

    Args:
        func: a function that takes in a list of tensors (one for each irrep) and returns a list of tensors (one for each irrep)
        x: a list of tensors, each of shape (*, Ci, di) for each irrep, where Ci is the number of channels for that irrep and di is the (complex) dimension of that irrep.
        group: a group
        error_norm_p: the p value for the norm of the error (default is infinity norm)
    Returns:
        a list of average errors for each irrep
    """

    y = func(x)

    errors_over_group = []
    for g in group:
        errors = []
        y_t = func(act_on_tensors(g, x))
        y_t2 = act_on_tensors(g, y)
        for i in range(len(y_t)):
            errors.append((y_t[i] - y_t2[i]).norm(p=error_norm_p).item())
        errors_over_group.append(errors)
    return errors_over_group

