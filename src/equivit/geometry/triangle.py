import numpy as np
import torch


class Triangle:
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
    def __init__(self, N: int):
        self.N = N # side length of the triangle
        self.L = (N+1)*(N+2)//2 # number of lattice sites

        self.index_3t1 = dict()
        self.index_1t3 = dict()

        self.index_2t3 = dict()
        self.index_3t2 = dict()

        self.index_1t2 = dict()
        self.index_2t1 = dict()

        verts_3d = []
        _q = 0
         
        for r in range(N+1):
            for s in range(N+1):
                if r + s <= N:
                    barycoord = (N-r-s,r,s)
                    self.index_3t1[barycoord] = _q
                    self.index_1t3[_q] = barycoord

                    self.index_2t3[r,s] = barycoord
                    self.index_3t2[barycoord] = (r,s)

                    self.index_2t1[r,s] = _q
                    self.index_1t2[_q] = (r,s)
                    _q += 1
                    verts_3d.append(barycoord)
        self.verts_3d = np.array(verts_3d)

        self.compute_irrep_projections()


    def compute_irrep_projections(self):

        # Compute D3 irrep spaces

        _dots = set(range(self.L))

        # get the orbits of the D3-action
        self.orbits = []
        while len(_dots) > 0:
            dot = next(iter(_dots))
            a,b,c = self.index_1t3[dot]

            orbit = set([self.index_3t1[P] for P in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]])
            self.orbits.append(list(orbit))
            _dots = _dots.difference(orbit)


        # A1 projections
        # shape: (Norbits, L)
        self.filters_a1 = torch.zeros((len(self.orbits), self.L), dtype=torch.float32)
        for i, orbit in enumerate(self.orbits):
            self.filters_a1[i, orbit] = 1


        # A2 projections
        orbits_with_a2 = []

        for orbit in self.orbits:
            if len(orbit) == 6:
                orbits_with_a2.append(orbit)
                
        self.filters_a2 = torch.zeros((len(orbits_with_a2), self.L), dtype=torch.float32)

        for i, orbit in enumerate(orbits_with_a2):
            a, b, c = self.index_1t3[orbit[0]]
            
            self.filters_a2[i,self.index_3t1[a,b,c]] = 1
            self.filters_a2[i,self.index_3t1[c,a,b]] = 1
            self.filters_a2[i,self.index_3t1[b,c,a]] = 1
            
            self.filters_a2[i,self.index_3t1[b,a,c]] = -1
            self.filters_a2[i,self.index_3t1[c,b,a]] = -1
            self.filters_a2[i,self.index_3t1[a,c,b]] = -1


        # E projections
        # an orbit with 3 elements contains 1 copy of E
        # an orbit with 6 elements contains 2 copies of E

        orbits_with_1e = []
        orbits_with_2e = []
        for orbit in self.orbits:
            if len(orbit) == 3:
                orbits_with_1e.append(orbit)
            if len(orbit) == 6:
                orbits_with_2e.append(orbit)


        self.filters_1e = torch.zeros((len(orbits_with_1e), 2, self.L), dtype=torch.float32)

        for i, orbit in enumerate(orbits_with_1e):
            a, b, c = self.index_1t3[orbit[0]]

            if b != c:
                if a == c:
                    a, b, c = b, a, c
                elif a == b:
                    a, b, c = c, a, b
            
            self.filters_1e[i,0,self.index_3t1[a,b,c]] = 2/np.sqrt(6)
            self.filters_1e[i,0,self.index_3t1[c,a,b]] = -1/np.sqrt(6)
            self.filters_1e[i,0,self.index_3t1[b,c,a]] = -1/np.sqrt(6)
            
            self.filters_1e[i,1,self.index_3t1[a,b,c]] = 0
            self.filters_1e[i,1,self.index_3t1[c,a,b]] = 1/np.sqrt(2)
            self.filters_1e[i,1,self.index_3t1[b,c,a]] = -1/np.sqrt(2)
                

        e2_6dimrep_basis_vectors = torch.tensor(np.array([
            np.array([2,-1,-1,-1,-1,2])/np.sqrt(12),
            np.array([0,1,1,-1,-1,0])/np.sqrt(4),
            np.array([0,1,-1,-1,1,0])/np.sqrt(4),
            -np.array([-2,-1,1,-1,1,2])/np.sqrt(12),
        ]).reshape(2,2,6)).to(torch.float32)

        _3dt2d_projection = np.array([
            [0,1/np.sqrt(2),-1/np.sqrt(2)],
            [2/np.sqrt(6),-1/np.sqrt(6),-1/np.sqrt(6)]
        ])

        self.filters_2e = torch.zeros((len(orbits_with_2e), 2, 2, self.L), dtype=torch.float32)

        for i, orbit in enumerate(orbits_with_2e):
            orbit_3inds = [np.array(self.index_1t3[o]) for o in orbit]
            orbit_3inds = np.array(sorted(orbit_3inds, key=lambda o: np.arctan2(*(_3dt2d_projection @ o).T))).tolist()
            orbit_1inds = [self.index_3t1[tuple(_k)] for _k in orbit_3inds]

            self.filters_2e[i,:,:,orbit_1inds] = e2_6dimrep_basis_vectors

        self.filters_e = torch.concat([self.filters_1e, 
                                       self.filters_2e.flatten(0,1)], dim=0)



    def d3_action(self, g: str, x: torch.Tensor):
        """
        :param g: group element of D3
        :type g: str
        :param x: tensor of shape (*, L)
        :type x: torch.Tensor
        """

        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        if g == 'r':
            for _q in range(self.L):
                a,b,c = self.index_1t3[_q]
                y[...,_q] = x[..., self.index_3t1[c,a,b]]

        elif g == 'rr':
            for _q in range(self.L):
                a,b,c = self.index_1t3[_q]
                y[...,_q] = x[..., self.index_3t1[b,c,a]]

        elif g == 't':
            for _q in range(self.L):
                a,b,c = self.index_1t3[_q]
                y[...,_q] = x[..., self.index_3t1[a,c,b]]
        else:
            raise ValueError(f'{g} is not a valid D6 group element')

        return y

    def fourier_space_features(self, x: torch.Tensor):
        """
        Basis transform to irrep spaces

        Implementation is not optimal. Only for testing purposes.
        
        :param x: tensor of shape (..., L)
        :type x: torch.Tensor

        :return: 
        :rtype:

        """
        f_a1 = torch.matmul(x, self.filters_a1.T)
        f_a2 = torch.matmul(x, self.filters_a2.T)
        f_e = torch.einsum('fjl,...l->...fj', self.filters_1e, x)
        f_2e = torch.einsum('fijl,...l->...fij', self.filters_2e, x).flatten(-3,-2)

        f_e_tot = torch.concat([f_e, f_2e], dim=-2)
        return f_a1, f_a2, f_e_tot

    def points(self):
        # e1 = np.array([1,0])
        # e2 = np.array([1,np.sqrt(3)])/2
        e1 = np.array([-1/2,-np.sqrt(3)/2])
        e2 = np.array([1/2,-np.sqrt(3)/2])

        E = np.array([e1, e2])

        points = np.einsum(
                        'ni,ij->nj', 
                        np.array([self.index_1t2[q] for q in range(self.L)]),
                        E
                    )

        return points

    def get_pixel_polygons(self, hex_radius_epsilon=0.08):
        points = self.points()
        r = 1 / np.sqrt(3) * (1+hex_radius_epsilon)
        rot = 1
        theta = np.linspace(.5*rot/6*np.pi*2, (6+.5*rot)/6*np.pi*2, 7)
        verts = np.array(
            [points[:,None,0] + r*np.cos(theta), 
             points[:,None,1] + r*np.sin(theta)]
            ).transpose(1,2,0)

        return verts

