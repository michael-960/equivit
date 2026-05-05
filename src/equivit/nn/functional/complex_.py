import torch


def to_real(z: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex tensor to a real tensor.

    Args:
        z: A complex tensor of shape :math:`(*,)`.

    Returns:
        A real tensor of shape :math:`(*, 2)`, where the last dimension contains the real and imaginary parts of the complex tensor.

    """
    return torch.stack([z.real, z.imag], dim=-1)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a real tensor to a complex tensor.

    Args:
        x: A real tensor of shape :math:`(*, 2)`, where the last dimension contains the real and imaginary parts of the complex tensor.

    Returns:
        A complex tensor of shape :math:`(*,)`.

    """
    return torch.complex(x[..., 0], x[..., 1])