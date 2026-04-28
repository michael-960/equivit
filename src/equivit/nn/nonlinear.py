from matplotlib import projections
import torch
import torch.nn as nn
from typing import List, Optional, Callable

from ..geometry import GroupAction, decompose_set_action, Group, IrrepType
from .utils import assert_all_not_quaternionic



class EquivariantNonlinear(nn.Module):
    r"""
    Let :math:`G` be a group. Let :math:`(\rho_i, V_i)_{i=0}^{M-1}` be the real
    irreps of :math:`G`. 

    Let :math:`X_0, X_1, \dotsb, X_{N-1}` be the homogeneous spaces of :math:`G`.
    That is, each :math:`X_a` is a set with a transitive :math:`G`-action, and 
    the stabilizer groups of the :math:`X_a` are all distinct (up to conjugation).

    For each :math:`a`, let :math:`C(X_a, \mathbb{R})` be the vector space of
    real-valued functions on :math:`X_a`. The natural :math:`G`-representation on :math:`C(X_a, \mathbb{R})`
    decomposes into irreps as :math:`\Phi_{a}: C(X_a, \mathbb{R}) \rightarrow
    \bigoplus_{i=0}^{M-1} \mathbb{R}^{\nu_i^a}\otimes V_i`, where
    :math:`\nu_i^a` is the multiplicity of the irrep :math:`V_i` in the
    decomposition of :math:`C(X_a, \mathbb{R})`. Fix one such isomorphism
    :math:`\Phi_{a}` for each :math:`a`.

    Fix a list of integers :math:`\mu_0, \mu_1, \dotsc, \mu_{N-1}` specifying
    the number of copies of each homogeneous space to use in the nonlinearity.

    Suppose :math:`x = (x_0, x_1, \dots, x_{M-1}) \in \bigoplus_{i=0}^{M-1} \mathbb{R}^{C_i}\otimes V_i` is
    an input feature vector with multiplicity (number of channels) :math:`C_i` for the irrep :math:`V_i`.

    Given an activation function :math:`\sigma: \mathbb{R} \to \mathbb{R}`, this 
    layer applies the following nonlinearity to :math:`x` and returns :math:`x'` : 

    .. math::
        \begin{aligned}
        & y_a = \Phi_a^{-1}\otimes \mathbb{1}_{\mu_a}\left(\bigoplus_{i=0}^{M-1} x_i\left[\sum_{b=0}^{a-1} \mu_b\nu_i^b : \sum_{b=0}^{a-1} \mu_b\nu_i^b + \mu_a\nu_i^a\right]\right) \in C(X_a, \mathbb{R})\otimes \mathbb{R}^{\mu_a} \\ 
        & y_a' = \sigma(y_a) \in C(X_a, \mathbb{R})\otimes \mathbb{R}^{\mu_a} \; \text{(applied entrywise)}\\
        & x_i' = \bigoplus_{a=0}^{N-1} [\Phi_a\otimes \mathbb{1}_{\mu_a}(y_a')]_i  \in \mathbb{R}^{C_i}\otimes V_i.
        \end{aligned}

    The map :math:`x \mapsto x'` is equivariant.
    
    Note: 
        We need :math:`\sum_{a=0}^{N-1} \mu_a\nu_i^a = C_i` for each
        :math:`i`, so that the input feature vector has enough channels to be split
        according to the multiplicities of the irreps in the homogeneous space
        decompositions.
        Thus, the multiplicities :math:`C_i` are computed automatically
        once the :math:`\mu_a` are specified. (Note that the matrix
        :math:`(\nu_i^a)_{i,a}` is determined entirely by the group :math:`G`.

    Args:
        group: the group :math:`G` for which the equivariant nonlinearity is defined
        homogeneous_space_copies: list of nonnegative integers :math:`\mu_0, \mu_1, \dotsc, \mu_{N-1}` specifying the number of copies for each homogeneous space
        activation: activation function to use in the pointwise nonlinearity
    """
    def __init__(self,
        group: Group,
        homogeneous_space_copies: List[int],
        activation: Callable=nn.ReLU(),
    ):
        super().__init__()
        self.group = group
        self.num_irreps = len(group.real_irreps())
        self.homogeneous_space_copies = homogeneous_space_copies
        self.num_homog_spaces = len(homogeneous_space_copies)
        
        self.activation = activation

        self.homog_actions = self.group.all_homogeneous_space_actions()
        assert len(homogeneous_space_copies) == len(self.homog_actions), "Length of homogeneous_space_copies must match the number of homogeneous spaces of the group."   

        self.fouriers = nn.ModuleList([Fourier(action) for action in self.homog_actions])

        # the i-th item of this list is the list of irrep multiplicities of the i-th homogeneous spac
        self.multiplicities = [
            fourier.irrep_multiplicities for fourier in self.fouriers
        ]

        # for the i-th irrep we need to allocate homogeneous_space_copies[n] * multiplicities[n][i]  channels of x[i]
        self.split_sizes = []
        for i in range(self.num_irreps):
            _ = []
            for n in range(len(self.homog_actions)):
                _.append(self.homogeneous_space_copies[n] * self.multiplicities[n][i])
            self.split_sizes.append(_)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)`, where :math:`d_i` is the (complex) dimension of the :math:`i`-th irrep

        Returns: list of tensors, each of shape :math:`(*, C_i, d_i)`
        """

        # TODO: too many for loops! optimize this

        # first, decompose each entry in x into chunks 
        x = [
            torch.split(y, self.split_sizes[i], dim=-2)
            for i, y in enumerate(x)
        ]

        # "transpose"
        x = list(zip(*x))

        outs = [None for _ in range(self.num_homog_spaces)]

        for n in range(len(self.homog_actions)):
            # transform to homogeneous space basis

            # each entry in x[n] has shape (*, homogeneous_space_copies[n]*multiplicities[n][i], di)
            x_homog = self.fouriers[n].inverse_transform([
                z.unflatten(-2, (self.homogeneous_space_copies[n], self.multiplicities[n][i])) for i, z in enumerate(x[n])
            ])
            # the output of inverse_transform has shape (*, homogeneous_space_copies[n], |X_n|) where X_n is the n-th homogeneous space

            # apply pointwise nonlinearity in homogeneous space
            x_homog = self.activation(x_homog)

            # transform back to real space
            # we get a list of tensors, each of shape (*, homogeneous_space_copies[n], multiplicities[n][i], di)
            y = self.fouriers[n].transform(x_homog)
            outs[n] = [z.flatten(-3, -2) for z in y]

        # recombine
        # outs[n][i] has shape (*, homogeneous_space_copies[n]*multiplicities[n][i], di)

        return [torch.cat([outs[n][i] for n in range(self.num_homog_spaces)], dim=-2) 
                for i in range(self.num_irreps)]


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=0)

class Fourier(nn.Module):
    """
    Given an action of a group G on a set X, 
    decompose the representation of G on functions on X into irreps, 
    and implement the corresponding Fourier transform and inverse Fourier transform.

    Note: this module should be used for small X (say |X| < 100), since the Fourier transform is
    implemented as a dense matrix multiplication.

    Note: this module assumes that all real irreps of the symmetry group (action.group) of complex type have 
    matrices that commute with the standard complex structure.
    """
    def __init__(self, action: GroupAction):
        super().__init__()
        assert_all_not_quaternionic(action.group)

        self.action = action

        irreps = action.group.real_irreps()

        self.dtypes = [torch.float32 if irrep.rep_type is IrrepType.REAL else torch.complex64 for irrep in irreps.values()]
        self.is_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in irreps.values()]
 
        projections = decompose_set_action(action)
        self.irrep_multiplicities = [len(p) for p in projections.values()]

        self.irrep_dims = [irrep.dim for irrep in irreps.values()]
        self.irrep_complex_dims = [irrep.dim if irrep.rep_type is IrrepType.REAL else irrep.dim//2 for irrep in irreps.values()]


        assert len(self.irrep_multiplicities) == len(self.irrep_dims), "Number of irreps in the decomposition does not match the number of irreps of the group. Something went wrong."
        self.split_sizes = [m*d for m, d in zip(self.irrep_multiplicities, self.irrep_dims)]
    
        all_projections = []
        for irrep_name, proj in projections.items():
            all_projections.extend([normalize_columns(p.to_dense()) for p in proj])
    
        matrix = torch.cat(all_projections, dim=-1)

        # (|X|, |X|)
        self.register_buffer('matrix', matrix.to(torch.float32))

    def transform(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        real space to frequency (or momentum or irrep) space
        x: (*, |X|)
        return: list of tensors, each of shape (*, Ri, di), where Ri is the multiplicity of the i-th irrep 
        and di is the complex dimension of the i-th irrep.
        """
        y = torch.matmul(x, self.matrix)
        chunks = [chunk.contiguous() 
                  for chunk in torch.split(y, self.split_sizes, dim=-1)]
        # return [
        #         chunk.view(dtype).view(*chunk.shape[:-1], m, d)
        #         for chunk, dtype, m, d in zip(chunks, self.dtypes, self.irrep_multiplicities, self.irrep_complex_dims)
        #         ]

        return [
            (
                torch.view_as_complex(chunk.view(*chunk.shape[:-1], m, d, 2)) 
                if m > 0 
                else torch.empty((*chunk.shape[:-1], 0, d), dtype=torch.complex64)
            )
            if is_complex
            else chunk.view(*chunk.shape[:-1], m, d)
            for chunk, is_complex, m, d in zip(chunks, self.is_complex, self.irrep_multiplicities, self.irrep_complex_dims)
        ]

    def inverse_transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        frequency space to real space
        x: list of tensors, each of shape (*, Ri, di), where Ri is the multiplicity of the i-th irrep and di is the dimension of the i-th irrep.

        Note: 
        - if the i-th irrep is of real type, then x[i] should be of real dtype 
        - if the i-th irrep is of complex type, then x[i] should be of complex dtype
        - di is the complex dimension of the i-th irrep
        """
        return torch.matmul(torch.cat(
                [z.view(torch.float32).flatten(-2, -1) for z in x], 
                # note: if z is complex, then z.view(torch.float32) will have shape (*, Ri, di*2)
                dim=-1),
                self.matrix.t()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(action={self.action})"
