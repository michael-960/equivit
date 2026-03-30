import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .groups import D6Irreps

from .groups import D6, get_set_action_rep_matrices, find_irrep_components
from typing import Union, cast, TYPE_CHECKING


# TODO remove HexGrid, rename AbstractHexagonGrid -> HexagonGrid


class Hexagon:
    """
    An object that keeps track of several indexing schemes of a hexaongal lattice
    restricted to a regular hexagon.

    There are three indexing schemes:
        - flattened: lattice sites are indexed by a single integer
        - lattice sites are indexed by a tuple of two integers which correspond to
          the coefficients of the two primal basis vectors (e_1, e_2) of the hexagonal lattice
        - lattice sites are indexed by a tuple of three integers which
          correspond to the coefficients of the three vectors (e_1, e_2-e_1, -e_2). This indexing scheme
          is redundant: both (i,j,k) and (i-a,j-a,k-a) refer to the same site

          
    :var N: side length of the hexagon

    :var L: number of lattice points

    """
    def __init__(self, N: int):
        """
        N: number of lattice sites on a single side of the hexagon
        """

        # side length of hexagon
        self.N = N

        self._setup_indices()
        self._setup_group_action()

    def _setup_indices(self):
        # dictionaries for index conversion
        self.index_enc = {1: dict(), 2: dict(), 3: dict()}
        self.index_dec = {1: dict(), 2: dict(), 3: dict()}

        _q = 0
        for i in range(-self.N, self.N+1):
            for j in range(-self.N, self.N+1):
                if -self.N-1< i + j < self.N+1:
                    self.index_enc[2][i, j] = _q
                    self.index_dec[2][_q] = (i, j)

                    self.index_enc[1][_q] = _q
                    self.index_dec[1][_q] = _q
                    _q += 1

        # number of lattice points
        self.L = _q

        for a in range(-self.N,self.N+1):
            for b in range(-self.N,self.N+1):
                for c in range(-self.N,self.N+1):
                    i, j = a-b, b-c
                    if -self.N <= i + j <= self.N and -self.N <= i <= self.N and -self.N <= j <= self.N:
                        q = self.index_enc[2][i,j]
                        self.index_enc[3][a,b,c] = q
                        self.index_dec[3][q] = (a,b,c)

    def _setup_group_action(self):
        self.action_dict = {}

        for g in D6:
            _dict = []
            for q in range(self.L):
                a,b,c = self.index_dec[3][q]
                for x in g.word[::-1]:
                    if x == 'r':
                        a,b,c = -b,-c,-a
                    elif x == 't':
                        a,b,c = -a,-c,-b
                    else:
                        raise ValueError(f'Invalid D6 generator: {x}')
                q_new = self.index_enc[3][a,b,c]
                _dict.append(q_new)
            self.action_dict[g] = np.array(_dict, dtype=np.int64)

    def d6_action(self, g: Union[D6,str], x: torch.Tensor):
        """
        :param g: group element of D6
        :type g: str
        :param x: tensor of shape (..., L)
        :type x: torch.Tensor
        """
        if type(g) is str:
            if g == 'e': g = D6.e
            else:
                g = D6.from_word(g)

        ind_dict = self.action_dict[g.inv()]
        y = x[...,ind_dict]

        return y

    def points(self):
        e1 = np.array([1.,0])
        e2 = np.array([1.,np.sqrt(3)])/2

        E = np.array([e1, e2])

        points = np.einsum(
                    'ni,ij->nj', 
                    np.array([self.index_dec[2][q] for q in range(self.L)]), 
                    E)
        return points

    def compute_irrep_projections(self):
        """
        The group D6 has 6 irreps:
        - A1, A2, B1, B2 (1-dimensional)
        - E1, E2 (2-dimensional)
        """
        _dots = set(range(self.L))
        orbits = []
        while len(_dots) > 0:
            dot = next(iter(_dots))
            orbit = []
            for g in D6:
                i = self.action_dict[g][dot]
                if i not in orbit: orbit.append(i)
            orbits.append(orbit)
            _dots = _dots.difference(orbit)

        self.irrep_projections = {irrep: [] for irrep in D6Irreps}
        for orbit in orbits:
            _rep_matrices = get_set_action_rep_matrices(self.action_dict, orbit)
            for irrep in D6Irreps:
                projections = torch.tensor(find_irrep_components(_rep_matrices, irrep, D6, clip_small_values=1e-11))

                for i in range(projections.shape[0]):
                    self.irrep_projections[irrep].append(
                        torch.sparse_coo_tensor(
                            indices=torch.tensor(orbit).unsqueeze(0),
                            values=projections[i].T,
                            size=(self.L, irrep.dim)
                        )
                    )






class HexGrid:
    def __init__(self, N: int):
        """
        a hexagonal lattice on a regular hexagon of side length N
        i.e., a single side of the hexagon contains N+1 lattice sites
        the lattice is generated by (1, 0) and (1/2, sqrt(3)/2)
        the lattice is shifted so that it is contained and centered in [0, 2N] x [0, 2N]
        
        :param N: side length of hexagon
        :type N: int

        """
        self.abstract_hex = AbstractHexagonGrid(N)
        self.N = N # side length of hexagon
        self.L = self.abstract_hex.L

        points = []
        
        e1 = np.array([1,0])
        e2 = np.array([1,np.sqrt(3)])/2

        for i in range(-N, N+1):
            for j in range(-N, N+1):
                if  -N-1< i + j < N+1:
                    points.append(i*e1 + j*e2)
                
        self.delta_y = (2*N - N*np.sqrt(3)) / 2
        self.points = (np.array(points) + np.array([N, np.sqrt(3) / 2 * N])) + np.array([0,self.delta_y])

        # number of lattice points
        self.L = self.points.shape[0]

        self.setup_interpolation()


    def setup_interpolation(self):
        I = np.array(self.points[:,0], dtype=np.int64)
        J = np.array(self.points[:,1], dtype=np.int64)

        self.I0 = np.where(I >= 2*self.N, 2*self.N-1, I)
        self.I1 = np.where(I+1 >= 2*self.N, 2*self.N-1, I+1)

        self.J0 = np.where(J >= 2*self.N, 2*self.N-1, J)
        self.J1 = np.where(J+1 >= 2*self.N, 2*self.N-1, J+1)

        self.interp_alpha = torch.tensor(self.points[:,0] - self.I0)
        self.interp_beta = torch.tensor(self.points[:,1] - self.J0)


    def crop_and_interpolate(self, img: torch.Tensor):    
        """
        Crop and convert a square image into a hexagonal image.
        Interpolation is bilinear.
        Ideally, img should have spatial size (2N,2N)

        img: (*, C, 2N, 2N)
        """
        shape = img.shape

        hex_img = img.new_zeros((*shape[:-2], self.points.shape[0]))

        hex_img[:] = img[...,self.J0,self.I0] * (1-self.interp_alpha)*(1-self.interp_beta) +\
                    img[...,self.J1,self.I0]*(1-self.interp_alpha)*self.interp_beta +\
                    img[...,self.J0,self.I1]*self.interp_alpha*(1-self.interp_beta) +\
                    img[...,self.J1,self.I1]*self.interp_alpha*self.interp_beta

        return hex_img

    def get_pixel_polygons(self, hex_radius_epsilon=0.08):
        """
        Vertices of the hexagonal pixels centered at the lattice sites. 
        """
        r = 1 / np.sqrt(3) * (1+hex_radius_epsilon)
        rot = 1
        theta = np.linspace(.5*rot/6*np.pi*2, (6+.5*rot)/6*np.pi*2, 7)
        verts = np.array(
            [self.points[:,None,0] + r*np.cos(theta), 
             # 2*self.N - self.points[:,None,1] + r*np.sin(theta)]
             self.points[:,None,1] + r*np.sin(theta)]
            ).transpose(1,2,0)

        return verts



    def imshow(self, ax: plt.Axes, hex_img: torch.Tensor, hex_radius_epsilon=0.08):
        """
        Show image on a hexagonal lattice
        
        :param ax: ax
        :type ax: plt.Axes
        :param hex_img: input hexagonal image (C,L)
        :type hex_img: torch.Tensor
        :param hex_radius_epsilon: factor by which to increase the hexagon pixel size
        """
        verts = self.get_pixel_polygons(hex_radius_epsilon)
        polycoll = PolyCollection(verts, facecolors=hex_img.permute(1,0).tolist())
        ax.add_collection(polycoll)

        ax.set_xlim(0, self.N*2)
        ax.set_ylim(0, self.N*2)
        ax.set_aspect('equal')

        ax.yaxis.set_inverted(True)
        ax.xaxis.tick_top()


