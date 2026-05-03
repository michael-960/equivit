import torch
from typing import Type, Union, Tuple, Any, overload
from ..groups import Group, decompose_set_action, TRIVIAL_GROUP, GroupAction, GroupElement
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import torch.nn.functional as F



class Lattice:
    """
    Abstract base class for lattice. 

    For us, a 'lattice' is a finite set of points in Euclidean space, 
    together with a group action that permutes these points.

    It might be more accurate to call this a point cloud (especially when the
    symmetry group is trivial), but we will stick with the term lattice.
    """
    symmetry_group: Group = TRIVIAL_GROUP
    """symmetry group of the lattice"""


    L: int
    """number of lattice points"""


    action_dict: dict
    index_dec: dict
    index_enc: dict

    action: GroupAction
    """group action of the symmetry group on the lattice points"""

    def get_points_from_basis(self, basis_vectors: Union[np.ndarray,list]):
        E = np.array(basis_vectors)

        points = np.einsum(
                    'ni,ij->nj', 
                    np.array([self.index_dec[2][q] for q in range(self.L)]), 
                    E)
        return points

    @property
    def points(self):
        r"""
        List of points in the lattice.

        Returns:
            a numpy array of shape :math:`(L, d)`, where :math:`L` is the number of lattice points and :math:`d` is the dimension of the ambient Euclidean space.
        """
        raise NotImplementedError("Subclasses of Lattice must implement the points property")

    @property
    def size(self):
        return self.L

    def get_pixel_polygons(self, radius_eps=0.08):
        """
        Get the vertices of the polygons corresponding to each lattice point for visualization.
        """
        raise NotImplementedError("Subclasses of Lattice must implement the get_pixel_polygons method")

    def colormesh(
        self, 
        ax: plt.Axes, x: torch.Tensor, radius_eps=None,
        swap_axes=False, invert_y=False,
        cmap=None,
    ):
        """
        Show image on the lattice using colored polygons.

        Args:
            ax: a matplotlib Axes object to plot on
            x: a tensor or array of shape :math:`(C,L)` or :math:`(L,)`. If shape is :math:`(L,)`, a cmap must be provided.
            radius_eps: factor by which to increase the pixel size for visualization
            swap_axes: whether to swap the `x` and :math:`y` axes for visualization
            invert_y: whether to invert the `y` axis for visualization (useful for visualizing images in the usual way)
            cmap: a matplotlib colormap to use if ``x`` has shape :math:`(L,)

        Note:
            ``invet_y`` is applied after ``swap_axes``.
        """
        verts = self.get_pixel_polygons(radius_eps=radius_eps)
        if swap_axes:
            verts = verts[...,::-1]

        if len(x.shape) == 2:
            if x.shape[0] == 1: x = x.squeeze(0)

        assert x.shape[-1] == self.L, f"Input image has {x.shape[-1]} pixels, but lattice has {self.L} points"

        if len(x.shape) == 2:
            if x.shape[0] in [3,4]: # interpret as RGB or RGBA
                facecolors = x.transpose(1,0).tolist()
        elif len(x.shape) == 1:
            assert cmap is not None, "If x has shape (L,) or (1,L), a cmap must be provided"
            facecolors = cmap(x.tolist())
        else:
            raise ValueError("Input image must have shape (C,L) or (L,) or (1,L)")

        polycoll = PolyCollection(verts, facecolors=facecolors)
        ax.add_collection(polycoll)
        ax.set_aspect('equal')

        x_min = verts[:,:,0].min()
        x_max = verts[:,:,0].max()
        y_min = verts[:,:,1].min()
        y_max = verts[:,:,1].max()

        ax.set_xlim(x_min-0.1, x_max+0.1)
        ax.set_ylim(y_min-0.1, y_max+0.1)

        if invert_y:
            ax.yaxis.set_inverted(True)
            ax.xaxis.tick_top()

    def imshow(self, ax: plt.Axes, x: torch.Tensor, radius_epsilon=None, cmap=None):
        self.colormesh(ax, x, radius_epsilon, swap_axes=True, invert_y=True, cmap=cmap)


class LatticeImageInterpolator:
    """
    Interpolate a square image defined on a regular grid to an image defined on a lattice.
    """
    def __init__(self, 
        lattice: Lattice, 
        img_size: Union[int,tuple],
        scale: float = 1.0,
        offset: Tuple[int,int]=(0.,0.),
    ):
        if type(img_size) not in [tuple, list]:
            img_size = (img_size, img_size)

        self.lattice = lattice
        self.scale = scale
        self.offset = np.array(offset)

        if (not isinstance(img_size, list)) and (not isinstance(img_size, tuple)): 
            img_size = (img_size, img_size)
        self.img_size = img_size 

        self.setup_interpolation()

    def setup_interpolation(self):
        points = self.lattice.points * self.scale + self.offset
        H, W = self.img_size

        x_coords = np.clip(points[:,0]+1, 0., H+1) # plus one because we zero-pad the input image by one pixel on each side
        y_coords = np.clip(points[:,1]+1, 0., W+1)

        I = np.array(x_coords, dtype=np.int64)
        J = np.array(y_coords, dtype=np.int64)

        self.I0 = np.clip(I, 0, H+1)
        self.I1 = np.clip(I+1, 0, H+1)

        self.J0 = np.clip(J, 0, W+1)
        self.J1 = np.clip(J+1, 0, W+1)


        self.interp_alpha = torch.tensor(x_coords - self.I0)
        self.interp_beta = torch.tensor(y_coords - self.J0)

    def crop_and_interpolate(self, img: torch.Tensor):    
        """
        Crop and convert a square image into a hexagonal image.
        Interpolation is bilinear.

        img: (*, C, N, N)
        """
        img = F.pad(img, (1,1,1,1), mode='constant', value=0)

        shape = img.shape
        new_img = img.new_zeros((*shape[:-2], self.lattice.size))

        new_img[:] = img[...,self.I0,self.J0] * (1-self.interp_alpha)*(1-self.interp_beta) +\
                    img[...,self.I0,self.J1] * (1-self.interp_alpha)*self.interp_beta +\
                    img[...,self.I1,self.J0] * self.interp_alpha*(1-self.interp_beta) +\
                    img[...,self.I1,self.J1] * self.interp_alpha*self.interp_beta

        return new_img

