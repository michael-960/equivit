from __future__ import annotations
from tokenize import group
from typing import List, overload, TYPE_CHECKING
import numpy as np

import torch

if TYPE_CHECKING:
    from .representations.base import GroupElement
    from .action import GroupAction
    from .representations import GroupRepresentation



class EquivariantPullbackBundle:
    r"""
    Given:
     - a group :math:`G` acting on a set :math:`X`
     - a choice of basepoint :math:`x_O` in each :math:`G`-orbit :math:`O \subset X`
     - a subgroup :math:`H \subset G` such that :math:`\mathrm{Stab}_G(x_O) \subset H` for each basepoint :math:`x_O`

    consider the following construction:

    Let :math:`\phi: X \rightarrow G/H` be the map sending each :math:`x \in X`
    to the coset :math:`gH` such that :math:`gx_O = x`, where :math:`O` is the
    :math:`G`-orbit containing :math:`x`. 

    This is well-defined: if :math:`g` is another element such that :math:`gx_O = x`, then :math:`g^{-1}g' \in Stab_G(x_O) \subset H`.

    Now, pull back the principal :math:`H`-bundle :math:`G \rightarrow G/H` along
    :math:`\phi` to get a principal :math:`H`-bundle over :math:`X`:

    .. math::
        \begin{CD}
            P = \phi^*G @>>> G \\
            @VVV @VVV \\
            X @>\phi>> G/H
        \end{CD}


    If :math:`V` is any :math:`H`-representation, then one can construct an associated vector bundle over :math:`X` with typical fiber :math:`V`.
    The space of sections of this bundle carries a natural :math:`G`-representation.

    Args:
        action: a group action of :math:`G` on :math:`X`
        subgroup_args: arguments specifying the subgroup :math:`H` (see Group.subgroup for details)
        representatives: a list of representatives for the left cosets of
            :math:`H` in :math:`G` (the order of the representatives should match
            the order of left cosets returned by Group.left_cosets)
        basepoints: a list of integers specifying the index of the basepoint in
            each :math:`G`-orbit (the order should match the order of orbits
            returned by GroupAction.orbits)
    """
    def __init__(self, action: GroupAction, subgroup_args: tuple, representatives: List[GroupElement], basepoints: List[int]):
        self.action = action
        self.group = action.group

        self.subgroup_args = subgroup_args
        self.subgroup_incl = self.group.subgroup(*subgroup_args)
        self.subgroup = self.subgroup_incl.source

        self.representatives = representatives
        self.basepoints = basepoints

        assert len(self.basepoints) == len(self.action.orbits())

        orbits = self.action.orbits()

        _subgroup_image = [self.subgroup_incl(h) for h in self.subgroup]
        for basepoint, orbit in zip(self.basepoints, orbits):
            assert basepoint in range(len(orbit)), f"Each basepoint must be an integer in the range [0, |O|), where O is the corresponding G-orbit. Got {basepoint} for orbit of size {len(orbit)}."

            # assert that the stabilizer of the basepoint is contained in the subgroup
            restricted_action = self.action.restrict_action(orbit)
            stabilizer = [g for g in self.group if restricted_action(g)[basepoint] == basepoint]
            for s in stabilizer:
                assert s in _subgroup_image, f"The subgroup must contain the stabilizer of the base point. Got stabilizer element {s} which is not in the subgroup image."

        # set up the principal H-bundle: 
        cosets = self.group.left_cosets(self.subgroup_incl)
        assert len(representatives) == len(cosets), f"The number of representatives must match the number of cosets of the subgroup in the group. Got {len(representatives)} representatives and {len(cosets)} cosets."

        for r, coset in zip(representatives, cosets):
            assert r in coset, f"Each representative must be an element of the corresponding coset. Got representative {r} which is not in the coset {coset}."

        # self.fibers[i] is the union of the fibers over the points in X which map to the i-th coset of H in G.
        self.fibers = []
        for k in representatives:
            fiber = []
            for i, orbit in zip(basepoints, orbits):
                kx0 = self.action(k)[orbit[i]]
                h_orbit = []
                for h in self.subgroup:
                    hkx0 = self.action(self.subgroup_incl(h))[kx0]
                    h_orbit.append(hkx0)
                fiber.extend(h_orbit) 
            self.fibers.append(fiber)

        # lookup table for the index (0,1,...) of each coset
        self.coset_inds = dict()
        for i, coset in enumerate(cosets):
            for k in coset:
                self.coset_inds[k] = i

        # lookup table for the element of H corresponding to each element of the subgroup image in G
        self.GtoH = dict()
        for h in self.subgroup:
            self.GtoH[self.subgroup_incl(h)] = h

    @overload
    def act_on_section(self, g: GroupElement, x: torch.Tensor, repr: GroupRepresentation, action_dim: int=0) -> torch.Tensor: ...
    @overload
    def act_on_section(self, g: GroupElement, x: np.ndarray, repr: GroupRepresentation, action_dim: int=0) -> np.ndarray: ...

    def act_on_section(
        self, g: GroupElement, x, repr: GroupRepresentation,
        action_dim: int=0
    ):
        r"""
        Given an :math:`H`-representation :math:`V`, there is a natural :math:`G`-action on the space of
        sections of the associated vector bundle :math:`P` \times_H V` over :math:`X` with typical fiber :math:`V`.
        """
        tensor_type = 'numpy'
        dtype = x.dtype
        if isinstance(x, torch.Tensor):
            tensor_type = 'torch'
            x = x.detach().cpu().numpy()

        if action_dim < 0:
            action_dim = x.ndim + action_dim
        assert action_dim < x.ndim, f"action_dim must be less than the number of dimensions of x. Got action_dim={action_dim} and x.ndim={x.ndim}."


        y = np.zeros_like(x)
        for i, k in enumerate(self.representatives):
            fiber = self.fibers[i]
            gfiber = [self.action(g)[t] for t in fiber]
        
            b = self.representatives[self.coset_inds[g*k]]
            h = self.GtoH[b.inv() * g * k]


            _slices_g = tuple(slice(None) if i != action_dim else gfiber for i in range(x.ndim))
            _slices = tuple(slice(None) if i != action_dim else fiber for i in range(x.ndim))

            y[_slices_g] = np.einsum('ij, ...j -> ...i', repr(h), x[_slices])

        if tensor_type == 'torch':
            y = torch.from_numpy(y).to(x.device).to(dtype)
        else:
            y = y.astype(dtype)
        return y


    def find_invariant_subspace(self, repr: GroupRepresentation) -> List[torch.Tensor]:
        r"""
        Find a basis for the invariant subspace of the induced representation (see Action.induce_from for details of the construction).

        Args:
            repr: a representation of H specified by subgroup_args
        Note: 
            each basepoint is an integer in :math:`\{0, 1, \dotsb, |O|-1\}`, where O is the corresponding G-orbit.
        """
        from .utils import find_irrep_components

        orbits = self.action.orbits()
        subgroup_incl = self.action.group.subgroup(*self.subgroup_args)
        subgroup = subgroup_incl.source

        dim = repr(subgroup.identity()).shape[0]

        trivial_rep = list(self.action.group.real_irreps().values())[0]

        invariant_vectors = []

        for orbit, basepoint in zip(orbits, self.basepoints):
            restricted_action = self.action.restrict_action(orbit)
            induced_rep = restricted_action.induce_from(self.subgroup_args, repr, self.representatives, basepoint)

            # shape: (number of invariant vectors, induced_rep_dim)
            inv_vecs = find_irrep_components(induced_rep, trivial_rep, clip_small_values=1e-9)[:,0,:].reshape(-1, len(orbit), dim)

            for i in range(inv_vecs.shape[0]):
                invariant_vectors.append(
                    torch.sparse_coo_tensor(
                        indices=torch.tensor(orbit).unsqueeze(0),
                        values=inv_vecs[i],
                        size=(self.action.num_elements, dim)
                    ).coalesce()
                )

        return invariant_vectors


    def twisted_product(self, action: GroupAction) -> GroupAction:
        r"""
        Given an :math:`H`-action on a set :math:`Y`, we can construct a twisted
        product action of :math:`G` on the set :math:`X \times Y` as follows:


        Note:
            - The flattened index of the resulting action is given by
              :math:`i_{X \times Y} = i_X \cdot |Y| + i_Y`, where :math:`i_X` and
              :math:`i_Y` are the flattened indices of the input actions on
              :math:`X` and :math:`Y` respectively.
        """
        from .action import GroupAction

        # TODO: we can relax this by allowing a homomorphism from H to the group acting on Y
        assert action.group is self.subgroup, f"The input action must be an action of the subgroup H. Got action of group {action.group} and subgroup {self.subgroup}."

        action_dict = dict()

        def _flatten(x, y):
            return x * action.num_elements + y

        for g in self.group:
            _dict = dict()
            for i, k in enumerate(self.representatives):

                b = self.representatives[self.coset_inds[g*k]]
                h = self.GtoH[b.inv() * g * k]

                for x in self.fibers[i]:
                    gx = self.action(g)[x]

                    for y in range(action.num_elements):
                        _dict[(x,y)] = (gx, action(h)[y])

            action_dict[g] = [_flatten(*_dict[x,y]) for x in range(self.action.num_elements) for y in range(action.num_elements)]

        return GroupAction(group=self.group, action_dict=action_dict)