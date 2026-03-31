import numpy as np
import torch

from ..groups import D4, decompose_set_action

from typing import Union, cast, TYPE_CHECKING

from .base import Lattice

from .base import LatticeImageInterpolator


class Square(Lattice):
    """
    A square lattice restricted to a square.

    There are two indexing schemes:
        - flattened: lattice sites are indexed by a single integer
        - lattice sites are indexed by a tuple of two integers which correspond to
          the coefficients of the two primal basis vectors (e_1, e_2) of the square lattice

    :var N: side length of the hexagon
    :var L: number of lattice points

    """
    symmetry_group = D4

    def __init__(self, N: int):
        """
        N: number of lattice sites on a single side of the square minus one.
        """
        self.N = N # side length of the triangle
        self._setup_indices()
        self._setup_group_action()

    def _setup_indices(self):
        N = self.N
        self.L = (N+1)*(N+1) # number of lattice sites

        self.index_enc = {1: dict(), 2: dict()}
        self.index_dec = {1: dict(), 2: dict()}

        verts_3d = []
        _q = 0
         
        for i in range(N+1):
            for j in range(N+1):
                self.index_enc[2][i,j] = _q
                self.index_dec[2][_q] = (i,j)

                self.index_enc[1][_q] = _q
                self.index_dec[1][_q] = _q
                _q += 1

    def _setup_group_action(self):
        self.action_dict = {}
        for g in D4:
            _dict = []
            for q in range(self.L):
                i,j = self.index_dec[2][q]
                for x in g.word[::-1]:
                    if x == 'r':
                        i, j = self.N-j, i
                    elif x == 't':
                        i, j = self.N-i, j
                    else:
                        raise ValueError(f'Invalid D4 generator: {x}')
                q_new = self.index_enc[2][i,j]
                _dict.append(q_new)
            self.action_dict[g] = np.array(_dict, dtype=np.int64)

    @property
    def points(self):
        # e1 = np.array([1,0])
        # e2 = np.array([1,np.sqrt(3)])/2
        e1 = np.array([1., 0.])
        e2 = np.array([0., 1.])
        return self.get_points_from_basis([e1, e2])

    def get_pixel_polygons(self, radius_eps=None):
        if radius_eps is None: radius_eps = 0.0
    
        points = self.points
        r = 1 / np.sqrt(2) * (1+radius_eps)
        rot = 1
        theta = np.linspace(.5*rot/4*np.pi*2, (4+.5*rot)/4*np.pi*2, 5)
        verts = np.array(
            [points[:,None,0] + r*np.cos(theta), 
             points[:,None,1] + r*np.sin(theta)]
            ).transpose(1,2,0)

        return verts

    def get_interpolator(self):
        offset = [0., 0.]
        return LatticeImageInterpolator(self, img_size=[self.N+1, self.N+1], offset=offset)
