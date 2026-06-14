import torch
import torch.nn as nn
import math



def complex_uniform_disk_(x: torch.Tensor, radius: float=1.) -> torch.Tensor:
    """
    Fills the input tensor with complex numbers uniformly distributed in the disk of given radius.

    Args:
        x: the input tensor to fill, should have a complex dtype
        radius: the radius of the disk

    Returns:
        The input tensor filled with complex numbers uniformly distributed in the disk.
    """
    assert x.dtype.is_complex, "Input tensor must have a complex dtype"

    if x.dtype == torch.complex64:
        real_dtype = torch.float32
    elif x.dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        raise ValueError("Unsupported complex dtype: {}".format(x.dtype))
    
    # Sample uniformly from the unit disk
    theta = torch.rand(x.shape, device=x.device, dtype=real_dtype) * 2 * math.pi
    r = torch.sqrt(torch.rand(x.shape, device=x.device, dtype=real_dtype)) * radius

    x[:] = r * torch.exp(1j * theta)
    
    return x



def kaiming_uniform_(x: torch.Tensor, fan_in: float, gain=math.sqrt(2)):
    r"""
    Applies Kaiming uniform initialization to the input tensor with given fan_in and gain.
    The entries of ``x`` are sampled uniformly from :math:`[-\text{bound}, \text{bound}]` where

    .. math:: 
    
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_in}}}


    Args:
        x: the input tensor to initialize, should have a real dtype
        fan_in: the number of input units in the weight tensor
        gain: the gain factor for the initialization (default is sqrt(2) for ReLU)

    Returns:
        The input tensor initialized with Kaiming uniform distribution.

    Note:
        ``fan_in`` and ``gain`` should be calculated independently.
    """
    bound = gain * math.sqrt(3 / fan_in) if fan_in > 0 else 0
    return nn.init.uniform_(x, a=-bound, b=bound)



def complex_kaiming_uniform_(x: torch.Tensor, fan_in: float, gain=math.sqrt(2)):
    r"""
    Applies Kaiming uniform initialization to the input tensor with given fan_in and gain, for complex tensors.
    The entries of ``x`` are sampled uniformly from the disk of radius :math:`\text{radius}` where

    .. math:: 
    
        \text{radius} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in}}}

    Args:
        x: the input tensor to initialize, should have a complex dtype
        fan_in: the number of input units in the weight tensor
        gain: the gain factor for the initialization (default is sqrt(2) for ReLU)
    
    Returns:
        The input tensor initialized with complex Kaiming uniform distribution.

    Note:
        ``fan_in`` and ``gain`` should be calculated independently.
    """
 
    radius = gain * math.sqrt(2 / fan_in) if fan_in > 0 else 0
    return complex_uniform_disk_(x, radius=radius)

def complex_trunc_normal_(x: torch.Tensor, std: float = 0.02, bound: float = 0.04):
    assert x.dtype.is_complex
    if x.dtype == torch.complex64:
        real_dtype = torch.float32
    elif x.dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        raise ValueError("Unsupported complex dtype: {}".format(x.dtype))
    with torch.no_grad():
        real_std = std / math.sqrt(2.)
        real_part = torch.empty(x.shape, device=x.device, dtype=real_dtype).normal_(0, real_std)
        imag_part = torch.empty(x.shape, device=x.device, dtype=real_dtype).normal_(0, real_std)

        cand = torch.complex(real_part, imag_part)

        # Rejection sampling: redraw elements where the magnitude exceeds the bound
        while True:
            out_of_bounds = torch.abs(cand) > bound

            if not out_of_bounds.any():
                break

            num_redraws = out_of_bounds.sum().item()

            new_real = torch.empty(num_redraws, device=x.device, dtype=real_dtype).normal_(0, real_std)
            new_imag = torch.empty(num_redraws, device=x.device, dtype=real_dtype).normal_(0, real_std)

            cand[out_of_bounds] = torch.complex(new_real, new_imag)

        x.copy_(cand)
    return x

