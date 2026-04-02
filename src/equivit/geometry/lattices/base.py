import torch
from typing import Type, Union, ClassVar, Any
from ..groups import Group, decompose_set_action, TRIVIAL_GROUP, GroupAction, GroupElement
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class Lattice:
    """
    Abstract base class for lattice. 

    For us, a 'lattice' is a finite set of points in Euclidean space, 
    together with a group action that permutes these points.

    It might be more accurate to call this a point cloud (especially when the
    symmetry group is trivial), but we will stick with the term lattice.
    """
    symmetry_group: Group = TRIVIAL_GROUP

    action_dict: dict
    index_dec: dict
    index_enc: dict

    action: GroupAction

    def compute_irrep_projections(self):
        """
        Compute the irreducible representations of the actin of the symmetry group on the lattice.
        """
        # self.irrep_projections = decompose_set_action(
        #     self.action_dict, self.__class__.symmetry_group
        # )
        self.irrep_projections = decompose_set_action(self.action)

    def group_action(self, g: Union[GroupElement,Any], x: torch.Tensor):
        """
        :param g: group element of D6
        :type g: str
        :param x: tensor of shape (..., L)
        :type x: torch.Tensor
        """
        group = self.__class__.symmetry_group
        if isinstance(g, GroupElement):
            assert g in group, f"Group element {g} not in group {group.__name__}"
        else:
            g = group[g]

        ind_dict = self.action(g.inv())
        y = x[...,ind_dict]
        return y

    def get_points_from_basis(self, basis_vectors: Union[np.ndarray,list]):
        E = np.array(basis_vectors)

        points = np.einsum(
                    'ni,ij->nj', 
                    np.array([self.index_dec[2][q] for q in range(self.L)]), 
                    E)
        return points

    @property
    def points(self):
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

        :param ax: ax
        :type ax: plt.Axes
        :param x: input image (C,L)
        :type x: a tensor or array of shape (C,L) or (L,). If shape is (L,), a cmap must be provided.
        :param radius_epsilon: factor by which to increase the pixel size for visualization
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
        offset=[0.,0.],
    ):
        if type(img_size) not in [tuple, list]:
            img_size = (img_size, img_size)


        self.lattice = lattice
        self.offset = np.array(offset)
        self.img_size =img_size 
        self.setup_interpolation()

    def setup_interpolation(self):
        points = self.lattice.points + self.offset

        I = np.array(points[:,0], dtype=np.int64)
        J = np.array(points[:,1], dtype=np.int64)

        H, W = self.img_size

        self.I0 = np.where(I >= H, H-1, I)
        self.I1 = np.where(I+1 >= H, H-1, I+1)

        self.J0 = np.where(J >= W, W-1, J)
        self.J1 = np.where(J+1 >= W, W-1, J+1)

        self.interp_alpha = torch.tensor(points[:,0] - self.I0)
        self.interp_beta = torch.tensor(points[:,1] - self.J0)

    def crop_and_interpolate(self, img: torch.Tensor):    
        """
        Crop and convert a square image into a hexagonal image.
        Interpolation is bilinear.

        img: (*, C, N, N)
        """
        shape = img.shape
        new_img = img.new_zeros((*shape[:-2], self.lattice.size))

        new_img[:] = img[...,self.I0,self.J0] * (1-self.interp_alpha)*(1-self.interp_beta) +\
                    img[...,self.I0,self.J1] * (1-self.interp_alpha)*self.interp_beta +\
                    img[...,self.I1,self.J0] * self.interp_alpha*(1-self.interp_beta) +\
                    img[...,self.I1,self.J1] * self.interp_alpha*self.interp_beta

        return new_img

