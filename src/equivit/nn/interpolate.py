from typing import Tuple, Union, Literal
import torch

import torch.nn as nn
from ..geometry import Lattice, LatticeImageInterpolator





class CropAndInterpolate(nn.Module, LatticeImageInterpolator):
    r"""
    Crop and convert a square image into an image defined on a lattice.
    Currently, only bilinear interpolation is supported.

    Args:
        lattice: the target lattice
        img_size: the size of the input square image :math:`(H, W)`
        scale: scale factor for the lattice points before interpolation. Default is 1.0 (no scaling).
        offset: offset for the lattice points before interpolation. Default is (0., 0.).

    Note:
        The x coordinates of the lattice points correspond to the height (:math:`H`) dimension of the input image. 
    """
    def __init__(self, 
        lattice: Lattice, 
        img_size: Union[int,tuple],
        scale: float = 1.0,
        offset: Union[Tuple[int,int], Literal['center']]='center',
    ):
        nn.Module.__init__(self)
        LatticeImageInterpolator.__init__(self, lattice=lattice, img_size=img_size, scale=scale, offset=offset)


        _I0 = torch.tensor(self.I0, dtype=torch.long)
        _I1 = torch.tensor(self.I1, dtype=torch.long)
        _J0 = torch.tensor(self.J0, dtype=torch.long)
        _J1 = torch.tensor(self.J1, dtype=torch.long)
        _interp_alpha = self.interp_alpha
        _interp_beta = self.interp_beta

        for name in ['I0', 'I1', 'J0', 'J1', 'interp_alpha', 'interp_beta']: delattr(self, name)

        self.register_buffer('I0', _I0)
        self.register_buffer('I1', _I1)
        self.register_buffer('J0', _J0)
        self.register_buffer('J1', _J1)

        self.register_buffer('interp_alpha', _interp_alpha)
        self.register_buffer('interp_beta', _interp_beta)

        # TODO: this feels a bit hacky, maybe we can find a better way to do this?

    def forward(self, img: torch.Tensor):
        return self.crop_and_interpolate(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(lattice={self.lattice}, img_size={self.img_size}, scale={self.scale}, offset={self.offset})"