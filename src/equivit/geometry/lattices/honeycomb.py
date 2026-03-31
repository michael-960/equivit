import numpy as np
import torch
from matplotlib.collections import PolyCollection

from ..groups import D6, get_set_action_rep_matrices, find_irrep_components, decompose_set_action

from typing import Union, cast, TYPE_CHECKING

from .base import Lattice


class Honeycomb(Lattice):
    """
    A honeycomb lattice consists of two types of sites - a and b.

    There are four indexing schemes:
    - 
    """
    symmetry_group = D6
    def __init__(self, N: int):
        """
        :param N: number of layers of the honeycomb
        :type N: int
        """
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
        e1 = np.array([1,0])
        e2 = np.array([1,np.sqrt(3)])/2

        return self.get_points_from_basis([e1, e2])

    def get_pixel_polygons(self, radius_eps=None):
        """
        Triangular pixels centered at the honeycomb lattice sites.
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


# class HoneyCombGrid:
#     """
#     A honeycomb lattice
#     """
#     def __init__(self, N: int, div: int=1):
#         e1 = np.array([1, 0])
#         e2 = np.array([1, np.sqrt(3)])/2
#         assert N % div == 0
#         assert (N//div) % 3 == 0
#         f = (N // div) // 3
        
#         points = []
#         points2 = []

#         self.abs_2inds_1 = []
#         self.abs_2inds_2 = []

#         _q1 = 0
#         _q2 = 0

#         for n in range(-N, N+1):
#             for m in range(-N, N+1):
#                 i = (3*n+2)*f
#                 j = (3*m+2)*f
#                 if  (-N<= i + j <= N) and (-N <= i <= N) and (-N <= j <= N):
#                     points.append(i*e1 + j*e2)
#                     self.abs_2inds_1.append([i,j])
#                     _q1 += 1
                    
#                 k = (3*n+1)*f
#                 l = (3*m+1)*f
#                 if  (-N<= k + l <= N) and (-N <= k <= N) and (-N <= l <= N):
#                     points2.append(k*e1 + l*e2)
#                     self.abs_2inds_2.append([k,l])
#                     _q2 += 1

#         self.points = np.array(points)
#         self.delta_y = (2*N - N*np.sqrt(3)) / 2
#         self.points = (np.array(points) + np.array([N, np.sqrt(3) / 2 * N])) + np.array([0,self.delta_y])

#         self.points2 = np.array(points2)
#         self.points2 = (np.array(points2) + np.array([N, np.sqrt(3) / 2 * N])) + np.array([0,self.delta_y])
        
        
        
