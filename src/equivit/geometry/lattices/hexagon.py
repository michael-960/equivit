import numpy as np
import torch
from matplotlib.collections import PolyCollection

from ..groups import D6, decompose_set_action
from typing import Union, cast, TYPE_CHECKING

from .base import Lattice, LatticeImageInterpolator


# TODO remove HexGrid

class Hexagon(Lattice):
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

    symmetry_group = D6

    def __init__(self, N: int, orientation: str = 'y'):
        """
        N: number of lattice sites on a single side of the hexagon minus one.
        """

        # side length of hexagon
        self.N = N
        assert orientation in ['x', 'y'], f"Invalid hexagon orientation: {orientation}"
        self.orientation = orientation

        # if orientation == 'y':
        #     raise NotImplementedError("y-oriented hexagons not implemented yet")

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

    @property
    def points(self):
        if self.orientation == 'x':
            e1 = np.array([1.,0])
            e2 = np.array([1.,np.sqrt(3)])/2
        else:
            e1 = np.array([np.sqrt(3),1.])/2
            e2 = np.array([0,1.])
        return self.get_points_from_basis([e1, e2])

    def get_pixel_polygons(self, radius_eps=None):
        """
        Vertices of the hexagonal pixels centered at the lattice sites. 
        """
        if radius_eps is None: radius_eps = -0.01
        r = 1 / np.sqrt(3) * (1+radius_eps)

        rot = 1 if self.orientation == 'x' else 0

        theta = np.linspace(.5*rot/6*np.pi*2, (6+.5*rot)/6*np.pi*2, 7)
        verts = np.array(
            [self.points[:,None,0] + r*np.cos(theta), 
             self.points[:,None,1] + r*np.sin(theta)]
            ).transpose(1,2,0)
        return verts

    def get_interpolator(self):
        delta_y = (2*self.N - self.N*np.sqrt(3)) / 2
        offset = [self.N, np.sqrt(3)/2 * self.N + delta_y]
        return LatticeImageInterpolator(self, img_size=[self.N*2, self.N*2], offset=offset)



