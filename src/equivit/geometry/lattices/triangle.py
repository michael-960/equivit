import numpy as np
import torch

from ..groups import D3, dihedral_group_action

from typing import Union, cast, TYPE_CHECKING

from .base import Lattice


class Triangle(Lattice):
    """
    A hexaongal lattice restricted to a regular triangle.

    There are three indexing schemes:
        - flattened: lattice sites are indexed by a single integer
        - lattice sites are indexed by a tuple of two integers which correspond to
          the coefficients of the two primal basis vectors (e_1, e_2) of the hexagonal lattice
        - lattice sites are indexed by a tuple of three integers which
          correspond to the barycentric coordinates

    :var N: side length of the hexagon
    :var L: number of lattice points
    """

    symmetry_group = D3

    def __init__(self, N: int):
        """
        N: number of lattice sites on a single side of the triangle minus one.
        """
        self.N = N # side length of the triangle
        self._setup_indices()
        self._setup_group_action()

    def _setup_indices(self):
        N = self.N
        self.L = (N+1)*(N+2)//2 # number of lattice sites

        self.index_enc = {1: dict(), 2: dict(), 3: dict()}
        self.index_dec = {1: dict(), 2: dict(), 3: dict()}

        verts_3d = []
        _q = 0
         
        for r in range(N+1):
            for s in range(N+1):
                if r + s <= N:
                    barycoord = (N-r-s,r,s)

                    self.index_enc[3][barycoord] = _q
                    self.index_dec[3][_q] = barycoord

                    self.index_enc[2][r,s] = _q
                    self.index_dec[2][_q] = (r,s)

                    self.index_enc[1][_q] = _q
                    self.index_dec[1][_q] = _q
                    _q += 1
                    verts_3d.append(barycoord)

        self.verts_3d = np.array(verts_3d)

    def _setup_group_action(self):
        r_action = []
        t_action = []
        for q in range(self.L):
            a, b, c = self.index_dec[3][q]
            r_action.append(self.index_enc[3][c,a,b])
            t_action.append(self.index_enc[3][a,c,b])
            
        self.action = dihedral_group_action(D3, r_action=r_action, t_action=t_action)

        # self.action_dict = {}
        # for g in D3:
        #     _dict = []
        #     for q in range(self.L):
        #         a,b,c = self.index_dec[3][q]
        #         for x in g.word[::-1]:
        #             if x == 'r':
        #                 a,b,c = c,a,b
        #             elif x == 't':
        #                 a,b,c = a,c,b
        #             else:
        #                 raise ValueError(f'Invalid D3 generator: {x}')
        #         q_new = self.index_enc[3][a,b,c]
        #         _dict.append(q_new)
        #    self.action_dict[g] = np.array(_dict, dtype=np.int64)

    @property
    def points(self):
        # e1 = np.array([1,0])
        # e2 = np.array([1,np.sqrt(3)])/2
        e1 = np.array([-1/2,-np.sqrt(3)/2])
        e2 = np.array([1/2,-np.sqrt(3)/2])
        return self.get_points_from_basis([e1, e2])

    def get_pixel_polygons(self, radius_eps=0.08):
        points = self.points
        r = 1 / np.sqrt(3) * (1+radius_eps)
        rot = 1
        theta = np.linspace(.5*rot/6*np.pi*2, (6+.5*rot)/6*np.pi*2, 7)
        verts = np.array(
            [points[:,None,0] + r*np.cos(theta), 
             points[:,None,1] + r*np.sin(theta)]
            ).transpose(1,2,0)

        return verts
