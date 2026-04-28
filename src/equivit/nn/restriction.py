import torch.nn as nn
import torch
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..geometry import Group, GroupHomomorphism, find_irrep_components, IrrepType

from .utils import assert_all_not_quaternionic



class SymmetryRestriction(nn.Module):
    r"""
    Let :math:`\phi: H\rightarrow G` be a group homomorphism.

    If :math:`V` is a representation of :math:`G`, it is automatically a
    representation of :math:`H` via the pullback along :math:`\phi`.


    For each irrep :math:`V_i` of :math:`G`, we can decompose the pullback representation into irreps of :math:`H`: 

    .. math:: 
        
        \mathrm{Res}^G_H V_i \cong \bigoplus_j \mathbb{R}^{\mu_{ij}}\otimes W_j

    where :math:`W_j` are the irreps of :math:`H` and :math:`\mu_{ij}` are the multiplicities
    (number of times :math:`W_j` appears in :math:`V_i`).

    The isomorphism is not canonical, but we simply choose one and stick to it.
    Once we have fixed such an isomorphism for each :math:`V_i`, 
    we know how to map any feature vector :math:`x\in \bigoplus_i\mathbb{R}^{C_i}\otimes V_i` 
    transforming under the irreps to :math:`G`
    to a feature vector transforming under the irreps of :math:`H`:

    .. math::
    
        \bigoplus_i \mathbb{R}^{C_i}\otimes V_i
        \cong \bigoplus_i \bigoplus_j \mathbb{R}^{C_i}\otimes \mathbb{R}^{\mu_{ij}}\otimes W_j
        \cong \bigoplus_j \mathbb{R}^{\sum_i C_i\mu_{ij}}\otimes W_j

    This module takes in an element of :math:`\bigoplus_i\mathbb{R}^{C_i}\otimes V_i` as input 
    and outputs the corresponding element of :math:`\bigoplus_j \mathbb{R}^{\sum_i C_i\mu_{ij}}\otimes W_j`
    according to this isomorphism.


    For example, if :math:`H` is the trivial group, then the 
    above isomorphism for each irrep becomes :math:`V_i \cong \mathbb{R}^{\dim V_i}\otimes \mathbb{R}`,
    i.e., an input feature vector
    :math:`x\in \bigoplus_i\mathbb{R}^{C_i}\otimes V_i` is mapped to a feature
    vector in :math:`\bigoplus_i \mathbb{R}^{C_i\dim V_i}\otimes \mathbb{R}` by
    flattening the irrep dimensions into the channel dimension.
    """
    def __init__(self, homomorphism: GroupHomomorphism):
        super().__init__()
        self.homomorphism = homomorphism

        assert_all_not_quaternionic(homomorphism.source)
        assert_all_not_quaternionic(homomorphism.target)

        # irreps of H
        H_irreps = homomorphism.source.real_irreps()
        self.H_irrep_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in H_irreps.values()]
        self.H_irrep_dims = [irrep.dim for irrep in H_irreps.values()]
        self.num_H_irreps = len(H_irreps)

        # irreps of G
        G_irreps = homomorphism.target.real_irreps()
        self.G_irrep_dims = [irrep.dim for irrep in G_irreps.values()]
        self.num_G_irreps = len(G_irreps)



        for i, irrep in enumerate(G_irreps.values()):
            rep = irrep.pullback(homomorphism)
            # decompose into irreps of H
            for j, source_irrep in enumerate(H_irreps.values()):

                # notation: 
                # - D_j' is the dimension of the j-th irrep of H
                # - D_i is the dimension of the i-th irrep of G
                # - mu_ij is the multiplicity of the j-th irrep of H in (the pullback of) the i-th irrep of G

                # (mu_ij, D_j', D_i)
                res = find_irrep_components(rep, source_irrep)

                # we need to normalize this

                # (mu_ij, D_j', D_j')
                k = res @ res.transpose(0,2,1)

                # (mu_ij, D_j', D_i)
                res = res / np.sqrt(k[:,0,0][:,None,None])

                # this is a tensor of shape (D_i, mu_ij * D_j')
                self.register_buffer(f'res_{i}_{j}', torch.from_numpy(res.astype(np.float32)).flatten(0,1).permute(1,0))

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            x: a list of tensors. The :math:`i`-th tensor has shape :math:`(*, C_i, d_i)`, 
            where :math:`d_i` is the complex dimension of the irrep :math:`V_i` of :math:`G`.

        Returns:
            a list of tensors, where the :math:`j`-th tensor has shape :math:`(*, \sum_i C_i\mu_{ij}, d_j')`.
            :math:`d_j'` is the complex dimension of the irrep :math:`W_j` of :math:`H`.
        """

        common_shape = x[0].shape[:-2]

        # first, make everything real
        x = [z.view(torch.float32) for z in x]

        y = [None] * self.num_H_irreps

        for j, H_irrep_dim in enumerate(self.H_irrep_dims):
            _ = [None] * self.num_G_irreps
            for i, G_irrep_dim in enumerate(self.G_irrep_dims):
                # x[i] has shape (*, C_i, D_i) where D_i is the real dimension of the irrep V_i of G

                # (mu_ij * D_j', D_i)
                # (*, C_i, mu_ij * D_j')

                # (*, C_i, D_i) @ (D_i, mu_ij * D_j') -> (*, C_i, mu_ij * D_j')
                # then reshape to (*, C_i * mu_ij, D_j')
                _[i] =  (x[i] @ self.get_buffer(f'res_{i}_{j}')).view(*common_shape, -1, H_irrep_dim)

            y[j] = torch.cat(_, dim=-2)
            if self.H_irrep_complex[j]:
                y[j] = y[j].view(torch.complex64)

        return y

    def __repr__(self) -> str:
        return f'SymmetryRestriction({self.homomorphism})'