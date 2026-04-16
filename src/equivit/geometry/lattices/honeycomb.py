import numpy as np
import torch
from matplotlib.collections import PolyCollection

from ..groups import D6, find_irrep_components, decompose_set_action, dihedral_group_action

from typing import Union, cast, TYPE_CHECKING

from .base import Lattice

from .advanced import AdvancedLattice


class Honeycomb(Lattice):
    """
    A honeycomb lattice consists of two types of sites - a and b.

    There are four indexing schemes:

    Args:
        N: number of layers of the honeycomb
    """
    symmetry_group = D6

    N: int
    """number of layers of the honeycomb"""
    def __init__(self, N: int):
        self.N = N

        self._setup_indices()
        self._setup_group_action()

    def _setup_indices(self):
        self.index_enc = {1: dict(), 2: dict(), 3: dict(), 4: dict()}
        self.index_dec = {1: dict(), 2: dict(), 3: dict(), 4: dict()}

        N = self.N

        _q = 0
        _qa = 0
        for n in range(-N, N+1):
            for m in range(-N, N+1):
                i = 3*n + 2
                j = 3*m + 2
                if  (-3*N<= i + j <= 3*N) and (-3*N <= i <= 3*N) and (-3*N <= j <= 3*N):
                    self.index_enc[2][i,j] = _q
                    self.index_dec[2][_q] = (i,j)
                    self.index_enc[4][0,_qa] = _q
                    self.index_dec[4][_q] = (0,_qa)
                    _q += 1
                    _qa += 1

        _qb = 0
        for n in range(-N, N+1):
            for m in range(-N, N+1):
                k = 3*n + 1
                l = 3*m + 1
                if  (-3*N<= k + l <= 3*N) and (-3*N <= k <= 3*N) and (-3*N <= l <= 3*N):
                    self.index_enc[2][k,l] = _q
                    self.index_dec[2][_q] = (k,l)
                    self.index_enc[4][1,_qb] = _q
                    self.index_dec[4][_q] = (1,_qb)
                    _q += 1
                    _qb += 1

        self.L = _q
        assert self.L % 2 == 0
        # self.La = _q // 2 # this is still used somewhere else, maybe we can fix it

        for q in range(self.L):
            self.index_enc[1][q] = q
            self.index_dec[1][q] = q
        

        for a in range(-3*N, 3*N+1):
            for b in range(-3*N, 3*N+1):
                for c in range(-3*N, 3*N+1):
                    i, j = a-b, b-c
                    if ((i-j)%3==0) and (i%3 != 0):
                        if -3*N <= i + j <= 3*N and -3*N <= i <= 3*N and -3*N <= j <= 3*N:
                            self.index_enc[3][a,b,c] = self.index_enc[2][i,j]

        for q in range(self.L):
            i,j = self.index_dec[2][q]
            self.index_dec[3][q] = (i+j,j,0)

    def _setup_group_action(self):
        r_action = []
        t_action = []
        for q in range(self.L):
            a, b, c = self.index_dec[3][q]
            r_action.append(self.index_enc[3][-b,-c,-a])
            t_action.append(self.index_enc[3][-a,-c,-b])
            
        self.action = dihedral_group_action(D6, r_action=r_action, t_action=t_action)

    @property
    def points(self):
        e1 = np.array([1,0])
        e2 = np.array([1,np.sqrt(3)])/2

        return self.get_points_from_basis([e1, e2])

    def get_pixel_polygons(self, radius_eps=None):
        """
        Triangular pixels centered at the honeycomb lattice sites.

        Args:
            radius_eps: a small number to adjust the size of the pixels. If None, it will be set to -0.01, which means the pixels will be slightly smaller than the default size.
        """
        if radius_eps is None: radius_eps = -0.01
        r = np.sqrt(3) * (1+radius_eps)
        theta = np.linspace(np.pi/2, np.pi*2+np.pi/2, 4)

        # there are two types of sites - a and b, which have opposite orientations of the triangular pixels
        triverts_a = r*np.array([np.cos(theta), np.sin(theta)]).T
        triverts_b = r*np.array([np.cos(theta), -np.sin(theta)]).T

        site_types = np.array([self.index_dec[4][q][0] for q in range(self.L)])

        # vertex coordinates of the triangular pixels releative to the center of each site
        # shape: (L, 4, 2)
        triverts = triverts_a * site_types[:,None,None] + triverts_b * (1-site_types[:,None,None])

        # absolute vertex coordinates of the triangular pixels
        verts = self.points[:,None,:] + triverts 
        return verts



class AdvancedHoneycomb(Honeycomb, AdvancedLattice):
    """
    Experimental.
    """
    def __init__(self, N: int):
        super().__init__(N) 

        self.subgroup_incl = self.symmetry_group.subgroup('D', 3, 0)
        self.subgroup = self.subgroup_incl.source

        self.coset_representatives = [self.symmetry_group.from_value((0,0)),  # identity
                                      self.symmetry_group.from_value((0,3))   # r^3
                                      ]

        # For each G-orbit O, we need to choose a specific H-orbit
        # This is done by choosing a specific base point x_0 in O


        self.orbits = self.action.orbits()
        self.base_points = []

        for orbit in self.orbits:
            q = orbit[0]

            # This should be taken care of by the orbits() method
            assert self.index_dec[4][q][0] == 0, "The base point must be of type a"
            self.base_points.append(q)




