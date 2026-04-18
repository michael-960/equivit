from .base import PatchedLattice
import numpy as np

from ...._core import FunctionDict

from ..hexagon import Hexagon
from ... import dihedral_group_action, D6

class Hexhex(PatchedLattice):
    def __init__(self, N1: int, N2: int, patch_orientation='x', glue: bool=True):

        assert patch_orientation in ['x', 'y'], f'Invalid patch orientation: {patch_orientation}'

        if patch_orientation == 'y':
            self.hex1 = Hexagon(N1, orientation='x')
            self.hex2 = Hexagon(N2, orientation='y')
        else:
            self.hex1 = Hexagon(N1, orientation='y')
            self.hex2 = Hexagon(N2, orientation='x')

        self._setup_indices()
        self._setup_group_action()

    def _setup_indices(self):
        
        self.index_dec = {2: {}}
        self.index_enc = {2: {}}

        _patch_inds = []

        _q = 0
        for p in range(self.hex1.L):
            i, j = self.hex1.index_dec[2][p]

            _patch = []
            for q in range(self.hex2.L):
                k, l = self.hex2.index_dec[2][q]

                u = i *  self.hex2.N - j * self.hex2.N + k
                v = i *  self.hex2.N + j * 2 * self.hex2.N + l

                if (u,v) not in self.index_enc[2].keys():
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
            a = i + j
            b = j
            c = 0
            return (a, b, c)
        
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

 