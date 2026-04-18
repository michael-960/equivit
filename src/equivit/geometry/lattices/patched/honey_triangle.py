import numpy as np
from .base import PatchedLattice
from ..honeycomb import Honeycomb
from ..triangle import Triangle

from ...groups import dihedral_group_action, D6

from ...._core import FunctionDict

def _exact_div(m, n):
    assert m % n == 0, f'{m} is not divisible by {n}'
    return m // n

class HoneyTriangle(PatchedLattice):
    """
    A patched lattice obtained by taking the union of a collection of triangular patches 
    whose centers are lattice sites of a honeycomb lattice.

    Args:
        N_honey: The size of the honeycomb lattice, i.e. the number of hexagons along each edge.
        N_triangle: The side length of the triangular patch, i.e. the number of sites (pixels) along each edge **minus one**.
        glue: Whether to identify the sites on the boundary of adjacent patches. 
            If True, the resulting lattice will have fewer sites, but the patchification will be less efficient.
    """
    def __init__(self, N_honey: int, N_triangle: int, glue: bool=True):
        self.honey = Honeycomb(N_honey)
        self.triangle = Triangle(N_triangle)
        self.glue = glue

        self._setup_indices()

    def _setup_indices(self):
        self.index_dec = {2: {}}
        self.index_enc = {2: {}}

        _patch_inds = []

        N2 = self.triangle.N

        _q = 0
        for p in range(self.honey.L):
            i0, j0 = self.honey.index_dec[2][p]
            w, _ = self.honey.index_dec[4][p]

            _patch = []
            for q in range(self.triangle.L):
                r, s = self.triangle.index_dec[2][q]

                if w == 0:
                    a, b, c = _exact_div((i0+j0-1) * N2, 3), _exact_div((j0-2) * N2, 3) + s, -r
                else:
                    a, b, c = _exact_div((i0+j0+1) * N2, 3), _exact_div((j0+2) * N2, 3) - s, r

                u, v = a-b, b-c

                if ((u,v) not in self.index_enc[2].keys()) or (not self.glue):
                    self.index_dec[2][_q] = (u, v)
                    self.index_enc[2][u, v] = _q
                    _q += 1

                _patch.append(self.index_enc[2][u, v])
            _patch_inds.append(_patch)

        self.L = _q
        self.patch_inds = np.array(_patch_inds, dtype=np.int64)

        def _3t1(abc):
            a, b, c = abc
            i, j = a - b, b - c
            if (i,j) in self.index_enc[2].keys():
                return self.index_enc[2][i, j]
            else:
                raise ValueError(f'Invalid hexagon 3-index: {abc}') 

        self.index_enc[3] = FunctionDict(_3t1)

        def _1t3(q):
            assert q in range(self.L), f'Invalid hexagon flattened index: {q}'
            i, j = self.index_dec[2][q]
            return (i+j, j, 0)
        
        self.index_dec[3] = FunctionDict(_1t3)

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
        e1 = np.array([np.sqrt(3),1.])/2
        e2 = np.array([0,1.])
        return self.get_points_from_basis([e1, e2])



