import torch.nn as nn
import torch
from typing import List
import numpy as np

from ..geometry import GroupHomomorphism, find_irrep_components, IrrepType

from .utils import assert_all_not_quaternionic

from . import functional as EF


class SymmetryCorestriction(nn.Module):
    r"""
    Adjoint of :class:`equivit.nn.SymmetryRestriction`.

    Let :math:`\phi: H\rightarrow G` be a group homomorphism.
    :class:`SymmetryRestriction` fixes, for each irrep :math:`V_i` of :math:`G`, an isomorphism

    .. math::

        \mathrm{Res}^G_H V_i \cong \bigoplus_j \mathbb{R}^{\mu_{ij}}\otimes W_j

    onto the irreps :math:`W_j` of :math:`H`, and uses it to send a feature vector in
    :math:`\bigoplus_i\mathbb{R}^{C_i}\otimes V_i` to one in
    :math:`\bigoplus_j \mathbb{R}^{\sum_i C_i\mu_{ij}}\otimes W_j`.

    This module implements the inverse of that map, which -- because each block of the chosen
    isomorphism is normalised to an isometry -- is also its adjoint: for every
    :math:`x \in \bigoplus_i\mathbb{R}^{C_i}\otimes V_i` and every
    :math:`y \in \bigoplus_j \mathbb{R}^{\sum_i C_i\mu_{ij}}\otimes W_j`,

    .. math::

        \langle \mathrm{cores}(y), x\rangle = \langle y, \mathrm{res}(x)\rangle .

    Since :math:`\sum_j \mu_{ij}\dim W_j = \dim V_i`, the map is a bijection, so it is both the
    left and the right inverse of :class:`SymmetryRestriction` built from the same homomorphism.
    It is :math:`H`-equivariant, not :math:`G`-equivariant: only :math:`H` acts on both sides.

    The channel widths :math:`C_i` cannot be recovered from the input, because
    :class:`SymmetryRestriction` concatenates the contributions of all :math:`V_i` into a single
    channel axis for each :math:`W_j`. They must therefore be supplied.

    Args:
        homomorphism: the homomorphism :math:`\phi: H\rightarrow G`, i.e. the same object passed
            to the corresponding :class:`SymmetryRestriction`
        dims: the channel widths :math:`C_0, C_1, \dotsb` on the :math:`G` side, one per irrep of
            :math:`G` -- the widths of the tensors this module *outputs*

    Note:
        The buffers are rebuilt here rather than shared with a :class:`SymmetryRestriction`
        instance, so that the two modules stay independent for checkpointing. They agree because
        :func:`find_irrep_components` is deterministic; ``test_corestriction.py`` checks the
        round trip against an independently constructed :class:`SymmetryRestriction`.
    """
    def __init__(self, homomorphism: GroupHomomorphism, dims: List[int]):
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
        self.G_irrep_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in G_irreps.values()]
        self.G_irrep_dims = [irrep.dim for irrep in G_irreps.values()]
        self.num_G_irreps = len(G_irreps)

        assert len(dims) == self.num_G_irreps, \
            f"dims must have one entry per irrep of the target group: expected {self.num_G_irreps}, got {len(dims)}"
        self.dims = list(dims)

        # multiplicities mu_ij, needed to split the input channel axis back up
        self.multiplicities = [[0] * self.num_H_irreps for _ in range(self.num_G_irreps)]

        for i, irrep in enumerate(G_irreps.values()):
            rep = irrep.pullback(homomorphism)
            for j, source_irrep in enumerate(H_irreps.values()):

                # (mu_ij, D_j', D_i)
                res = find_irrep_components(rep, source_irrep)

                # normalize each component to an isometry, exactly as SymmetryRestriction does
                k = res @ res.transpose(0, 2, 1)
                res = res / np.sqrt(k[:, 0, 0][:, None, None])

                self.multiplicities[i][j] = res.shape[0]

                # (mu_ij * D_j', D_i) -- the transpose of SymmetryRestriction's buffer
                self.register_buffer(f'cores_{i}_{j}',
                                     torch.from_numpy(res.astype(np.float32)).flatten(0, 1))

    def forward(self, y: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            y: a list of tensors. The :math:`j`-th tensor has shape
                :math:`(*, \sum_i C_i\mu_{ij}, d_j')`, where :math:`d_j'` is the complex dimension
                of the irrep :math:`W_j` of :math:`H`. This is the output format of
                :class:`SymmetryRestriction`.

        Returns:
            a list of tensors, where the :math:`i`-th tensor has shape :math:`(*, C_i, d_i)` and
            :math:`d_i` is the complex dimension of the irrep :math:`V_i` of :math:`G`.
        """
        common_shape = y[0].shape[:-2]

        # make everything real
        y = [EF.to_real(z).flatten(-2, -1) if self.H_irrep_complex[j] else z for j, z in enumerate(y)]

        x = [None] * self.num_G_irreps

        for i, C_i in enumerate(self.dims):
            acc = None
            for j, H_irrep_dim in enumerate(self.H_irrep_dims):
                mu = self.multiplicities[i][j]

                # the block of y[j]'s channel axis contributed by V_i, in the order
                # SymmetryRestriction concatenated them
                offset = sum(self.dims[k] * self.multiplicities[k][j] for k in range(i))
                chunk = y[j][..., offset:offset + C_i * mu, :]

                # (*, C_i * mu_ij, D_j') -> (*, C_i, mu_ij * D_j')
                chunk = chunk.reshape(*common_shape, C_i, mu * H_irrep_dim)

                # (*, C_i, mu_ij * D_j') @ (mu_ij * D_j', D_i) -> (*, C_i, D_i)
                term = chunk @ self.get_buffer(f'cores_{i}_{j}')
                acc = term if acc is None else acc + term

            if self.G_irrep_complex[i]:
                acc = EF.to_complex(acc.unflatten(-1, (-1, 2)))

            x[i] = acc

        return x

    def __repr__(self) -> str:
        return f'SymmetryCorestriction({self.homomorphism}, dims={self.dims})'
