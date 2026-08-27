from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from equivit.geometry.groups.base import Group, GroupElement, GroupHomomorphism
from equivit.geometry.groups.action import GroupAction
from equivit.nn._core import resolve_dims

from equivit.nn.linear import EquivariantLinear
from equivit.nn.restriction import SymmetryRestriction, SymmetryCorestriction

from equivit.geometry import IrrepType
from equivit.nn import functional as EF





SubgroupSpec = Union[tuple, GroupHomomorphism]

@dataclass(frozen=True)
class HeadSet:
    r"""
    A finite :math:`G`-set :math:`H = \bigsqcup_i G/K_i`, presented in two ways at once.

    The heads are indexed **flatly** by :math:`\{0, \dots, |H|-1\}`, which is what
    :class:`GroupAction` requires, and **structurally** by a pair :math:`(i, p)`: the orbit
    index :math:`i` and the position :math:`p` of the head inside :math:`G/K_i`. Orbit
    :math:`i` occupies the contiguous block ``blocks[i]``, so the two indexings differ by a
    per-orbit offset and nothing else.

    Attributes:
        group: the group :math:`G`.
        inclusions: the inclusions :math:`K_i \hookrightarrow G`, one per orbit, in the order
            given. ``inclusions[i].source`` is :math:`K_i` as a group in its own right, which
            is what its irreps are indexed by.
        action: the :math:`G`-action on the flat index set :math:`\{0,\dots,|H|-1\}`.
        blocks: ``blocks[i][p]`` is the flat index of head :math:`(i, p)`. Each block is a
            contiguous range and is exactly one orbit.
    """
    group: Group
    inclusions: Tuple[GroupHomomorphism, ...]
    action: GroupAction
    blocks: Tuple[Tuple[int, ...], ...]

    def __len__(self) -> int:
        return self.action.num_elements

    @property
    def num_orbits(self) -> int:
        return len(self.blocks)

    def flat(self, i: int, p: int) -> int:
        r"""Flat index of head :math:`(i, p)`."""
        return self.blocks[i][p]

    def structured(self, h: int) -> Tuple[int, int]:
        r"""Orbit index and position of the head with flat index :math:`h`."""
        for i, block in enumerate(self.blocks):
            if block[0] <= h <= block[-1]:
                return i, h - block[0]
        raise IndexError(f"head index {h} out of range for |H| = {len(self)}")

    def base_point(self, i: int) -> int:
        r"""Flat index of the base point of orbit :math:`i`, whose stabilizer is :math:`K_i`."""
        return self.blocks[i][0]

    def orbit_action(self, i: int) -> GroupAction:
        r"""The :math:`G`-action on :math:`G/K_i` alone, indexed by :math:`p`."""
        return self.action.restrict_action(list(self.blocks[i]))


def build_head_set(group: Group, subgroups: Sequence[SubgroupSpec]) -> HeadSet:
    r"""
    Build :math:`H = \bigsqcup_i G/K_i` from a list of stabilizer subgroups.

    Args:
        group: the group :math:`G`.
        subgroups: one entry per orbit. Each is either a tuple of subgroup arguments, as
            accepted by :meth:`Group.subgroup` (e.g. ``('D', 2, 0)`` or ``('C', 2)`` for
            :math:`D_4`), or an inclusion :math:`K_i \hookrightarrow G` already constructed.
            Repeats are allowed: ``[('C', 2), ('C', 2)]`` is two copies of the same orbit.

    Returns:
        The :class:`HeadSet`. Its size is :math:`|H| = \sum_i |G|/|K_i|`.
    """
    assert group.is_finite(), "head_set is only implemented for finite groups."
    assert len(subgroups) > 0, "at least one subgroup is required"

    inclusions = []
    for spec in subgroups:
        if isinstance(spec, GroupHomomorphism):
            assert spec.target is group, "the inclusion must have the group as its target"
            assert spec.is_injective(), "the subgroup inclusion must be injective"
            inclusions.append(spec)
        else:
            inclusions.append(group.subgroup(*spec))

    orbit_actions = [group.homogeneous_space_action(incl) for incl in inclusions]

    offsets, total = [], 0
    for a in orbit_actions:
        offsets.append(total)
        total += a.num_elements

    action_dict: Dict[GroupElement, List[int]] = {}
    for g in group:
        row: List[int] = []
        for a, off in zip(orbit_actions, offsets):
            row.extend(off + q for q in a(g))
        action_dict[g] = row

    blocks = tuple(
        tuple(range(off, off + a.num_elements)) for a, off in zip(orbit_actions, offsets)
    )

    return HeadSet(
        group=group,
        inclusions=tuple(inclusions),
        action=GroupAction(group, action_dict),
        blocks=blocks,
    )


def head_transporters(head_set: HeadSet) -> Tuple[Tuple[GroupElement, ...], ...]:
    r"""
    A transversal of :math:`G/K_i` for each orbit, indexed by head.

    ``out[i][p]`` is an element :math:`g_{i,p} \in G` with :math:`g_{i,p} \cdot p_0^{(i)} = (i, p)`,
    where :math:`p_0^{(i)}` is ``head_set.base_point(i)``. It is defined only up to right
    multiplication by :math:`K_i`, which is exactly the redundancy that makes

    .. math::
        M_{(i,p)} := \rho_V(g_{i,p})\, M_{K_i}\, \rho_V(g_{i,p})^{-1}

    well defined for a :math:`K_i`-invariant :math:`M_{K_i}`, and :math:`g_{i,0}` is in
    :math:`K_i`, so head :math:`(i, 0)` gets :math:`M_{K_i}` back unchanged.

    The representatives are the ones :meth:`Group.left_cosets` already returns; each is matched to
    the head it actually reaches, so nothing depends on the order in which cosets come out.
    """
    group = head_set.group
    out = []

    for i, inclusion in enumerate(head_set.inclusions):
        base = head_set.base_point(i)
        reps: list = [None] * len(head_set.blocks[i])

        for coset in group.left_cosets(inclusion):
            g = coset[0]
            _, p = head_set.structured(head_set.action(g)[base])
            assert reps[p] is None, f"two cosets of K_{i} reach head ({i}, {p})"
            reps[p] = g

        assert all(g is not None for g in reps), f"orbit {i} is not covered by the cosets of K_{i}"
        out.append(tuple(reps))

    return tuple(out)



DimSpec = Union[List[int], Dict[str, int]]

class FiberMap(nn.Module):
    r"""
    A :math:`K`-equivariant map between :math:`V` and a fiber :math:`W`, in one or both directions.

    Let :math:`\iota: K \hookrightarrow G` be the stabilizer of one head's base point, and let
    :math:`W` be a :math:`K`-representation given by its irrep multiplicities. With
    ``directions='both'`` this module holds

    .. math::
        \mathrm{Hom}_K(W, V) \cdot \mathrm{Hom}_K(V, W)
        \ \ni\ M_K = \mathrm{up} \circ \mathrm{down},

    which is the space Theorem 1 puts the value/output form :math:`R_K` in. The attention form
    :math:`M_K` lives in the same space but is never applied as an operator: attention needs
    :math:`q = \phi_q x` and :math:`k = \phi_k y` separately, with
    :math:`\phi_q, \phi_k \in \mathrm{Hom}_K(V, W)` and :math:`M_K = \phi_q^\top \phi_k`. That is
    what ``directions='down'`` is for -- two of them, rather than one of each direction.

    Each half is a composition of modules that already exist:

    .. math::
        V \xrightarrow{\ \mathrm{res}\ } \mathrm{Res}^G_K V \xrightarrow{\ L\ } W
        \qquad
        W \xrightarrow{\ L'\ } \mathrm{Res}^G_K V \xrightarrow{\ \mathrm{cores}\ } V

    with :math:`\mathrm{res}`, :math:`\mathrm{cores}` the fixed isometries of
    :class:`SymmetryRestriction` / :class:`SymmetryCorestriction` and :math:`L, L'` learnable
    :class:`EquivariantLinear` layers instantiated on :math:`K` rather than on :math:`G`. Because
    :math:`\mathrm{res}` is an isomorphism, the parameterisation is onto: every element of
    :math:`\mathrm{Hom}_K(V, W)` is some :math:`L \circ \mathrm{res}`. Transposition is a bijection
    :math:`\mathrm{Hom}_K(V, W) \to \mathrm{Hom}_K(W, V)`, so two ``'down'`` halves reach exactly
    the same set of :math:`M_K` as one of each.

    Args:
        inclusion: :math:`\iota: K \hookrightarrow G`, e.g. ``G.subgroup('D', 2, 0)``.
        dims_V: the channel widths :math:`C_0, C_1, \dotsb` of :math:`V`, one per irrep of
            :math:`G`. Also accepted as a name-keyed dict.
        dims_W: the multiplicities of :math:`W`, one per irrep of :math:`K`. A dict keyed by
            :math:`K`'s irrep names -- ``{'A1': 1, 'A2': 3}`` -- is the preferred form, since a
            wrong key raises where a mis-ordered list silently means something else. Names left
            out mean zero. Note the keys are :math:`K`'s, not :math:`G`'s.
        directions: ``'both'`` (default), ``'down'`` for :math:`V \to W` alone, or ``'up'`` for
            :math:`W \to V` alone. Only the requested halves are built, so an unused
            :class:`EquivariantLinear` never reaches the optimiser or the checkpoint.
        trivial_rep_bias: off by default. A per-channel bias on the trivial irrep is
            :math:`K`-equivariant and therefore legal, but the forms of Theorem 1 are linear, so it
            has no place in :math:`M` or :math:`R`.

    Note:
        :math:`W` is meant to be a *sub*bundle fiber, i.e. a subrepresentation of
        :math:`\mathrm{Res}^G_K V`. Nothing here enforces that: asking for more copies of an irrep
        of :math:`K` than :attr:`restricted_dims` has is allowed, and by Schur simply wastes
        parameters without enlarging the set of expressible :math:`M`.
    """

    DIRECTIONS = ('both', 'down', 'up')

    def __init__(
        self,
        inclusion: GroupHomomorphism,
        dims_V: DimSpec,
        dims_W: DimSpec,
        directions: str = 'both',
        trivial_rep_bias: bool = False,
    ):
        super().__init__()
        assert directions in self.DIRECTIONS, \
            f"directions must be one of {self.DIRECTIONS}, got {directions!r}"

        self.inclusion = inclusion
        self.directions = directions

        G, K = inclusion.target, inclusion.source
        self.dims_V = resolve_dims(G, dims_V)
        self.dims_W = resolve_dims(K, dims_W)

        self.has_down = directions in ('both', 'down')
        self.has_up = directions in ('both', 'up')

        self.restriction = SymmetryRestriction(inclusion) if self.has_down else None
        self.corestriction = SymmetryCorestriction(inclusion, self.dims_V) if self.has_up else None

        # widths of Res^G_K V, one per irrep of K: C'_j = sum_i C_i mu_ij
        mu = self._multiplicities()
        self.restricted_dims = [
            sum(self.dims_V[i] * mu[i][j] for i in range(len(self.dims_V)))
            for j in range(len(self.dims_W))
        ]

        self.down_linear = (
            EquivariantLinear(K, self.restricted_dims, self.dims_W, trivial_rep_bias)
            if self.has_down else None
        )
        self.up_linear = (
            EquivariantLinear(K, self.dims_W, self.restricted_dims, trivial_rep_bias)
            if self.has_up else None
        )

    def _multiplicities(self) -> List[List[int]]:
        r""":math:`\mu_{ij}`, from whichever of the two isometries was built."""
        if self.corestriction is not None:
            return self.corestriction.multiplicities
        res = self.restriction
        return [[res.get_buffer(f'res_{i}_{j}').shape[1] // res.H_irrep_dims[j]
                 for j in range(res.num_H_irreps)]
                for i in range(res.num_G_irreps)]

    def down(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""":math:`V \to W`, in :math:`\mathrm{Hom}_K(V, W)`. Input is indexed by :math:`G`'s
        irreps, output by :math:`K`'s."""
        assert self.has_down, f"this FiberMap was built with directions={self.directions!r}"
        return self.down_linear(self.restriction(x))

    def up(self, w: List[torch.Tensor]) -> List[torch.Tensor]:
        r""":math:`W \to V`, in :math:`\mathrm{Hom}_K(W, V)`."""
        assert self.has_up, f"this FiberMap was built with directions={self.directions!r}"
        return self.corestriction(self.up_linear(w))

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""The endomorphism :math:`M_K = \mathrm{up}\circ\mathrm{down}` of :math:`V`, of rank at
        most :math:`\dim W`. Requires ``directions='both'``."""
        assert self.directions == 'both', \
            f"forward needs both halves; this FiberMap was built with directions={self.directions!r}"
        return self.up(self.down(x))

    def __repr__(self) -> str:
        return (f'FiberMap({self.inclusion}, dims_V={self.dims_V}, dims_W={self.dims_W}, '
                f'directions={self.directions!r}, restricted_dims={self.restricted_dims})')


class OrbitTransport(nn.Module):
    r"""
    The action of a fixed list of transporters :math:`g_0, \dotsc, g_{P-1}` on a feature list,
    stacked along a new head axis.

    Theorem 1 fixes the forms of one orbit by conjugation,
    :math:`M_p = \rho_V(g_p) M_{K} \rho_V(g_p)^{-1}`. Applying that literally would mean building
    :math:`P` different operators. Since

    .. math::
        M_p x = \rho_V(g_p)\, M_K\, \big(\rho_V(g_p)^{-1} x\big),

    it is cheaper to conjugate the *data*: :meth:`pull_back` produces the whole stack
    :math:`\{\rho_V(g_p)^{-1}x\}_p`, one shared :class:`FiberMap` runs on it with the head axis
    batched in, and :meth:`push_forward` carries the result back. Only heads in the same orbit can
    be batched this way, since :math:`W` -- and hence the shape of :math:`M_K` -- varies between
    orbits.

    Args:
        group: the group :math:`G`.
        transporters: the transporters of one orbit, i.e. one entry of the tuple returned by
            :func:`head_transporters`.

    Note:
        The buffers are :math:`\rho_i(g_p)` and :math:`\rho_i(g_p^{-1})` per irrep, stacked over
        :math:`p`. The inverse is taken in the group, not by transposing the matrix, so nothing
        here assumes the irreps are orthogonal.
    """

    def __init__(self, group: Group, transporters: Sequence[GroupElement]):
        super().__init__()
        self.group = group
        self.transporters = tuple(transporters)
        self.num_heads = len(self.transporters)

        irreps = group.real_irreps()
        self.irrep_complex = [irrep.rep_type is IrrepType.COMPLEX for irrep in irreps.values()]
        self.num_irreps = len(irreps)

        for i, irrep in enumerate(irreps.values()):
            assert irrep.rep_type is not IrrepType.QUATERNIONIC, "quaternionic irreps are not supported"
            fwd = np.stack([np.asarray(irrep(g)) for g in self.transporters])
            inv = np.stack([np.asarray(irrep(g.inv())) for g in self.transporters])
            self.register_buffer(f'rho_{i}', torch.from_numpy(fwd.astype(np.float32)))
            self.register_buffer(f'rho_inv_{i}', torch.from_numpy(inv.astype(np.float32)))

    def _apply_stack(self, x: List[torch.Tensor], prefix: str, stacked: bool) -> List[torch.Tensor]:
        out = []
        for i, z in enumerate(x):
            cplx = self.irrep_complex[i]
            r = EF.to_real(z).flatten(-2, -1) if cplx else z
            if not stacked:
                r = r.unsqueeze(-3)                       # (*, 1, C_i, D_i), broadcast over heads
            rho = self.get_buffer(f'{prefix}_{i}')        # (P, D_i, D_i)
            r = torch.einsum('pij,...pcj->...pci', rho, r.expand(*r.shape[:-3], self.num_heads, *r.shape[-2:]))
            if cplx:
                r = EF.to_complex(r.unflatten(-1, (-1, 2)))
            out.append(r)
        return out

    def pull_back(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""":math:`x \mapsto \{\rho_V(g_p)^{-1}x\}_p`. Each tensor gains a head axis in front of
        the channel axis: :math:`(*, C_i, d_i) \to (*, P, C_i, d_i)`."""
        return self._apply_stack(x, 'rho_inv', stacked=False)

    def push_forward(self, y: List[torch.Tensor]) -> List[torch.Tensor]:
        r""":math:`\{y_p\}_p \mapsto \{\rho_V(g_p) y_p\}_p`, head axis in, head axis out. Summing
        over heads is the caller's job."""
        return self._apply_stack(y, 'rho', stacked=True)

    def conjugate(self, base_form, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""The whole orbit's forms applied at once: :math:`\{M_p x\}_p` for
        ``base_form`` :math:`= M_K`, e.g. a :class:`FiberMap`."""
        return self.push_forward(base_form(self.pull_back(x)))

    def __repr__(self) -> str:
        return f'OrbitTransport({self.group}, num_heads={self.num_heads})'






def _fiber_layout(group: Group, dims: Sequence[int]):
    r"""``(multiplicity, real dimension, is complex)`` per irrep, and :math:`\dim W`."""
    layout = [(m, irrep.dim, irrep.rep_type is IrrepType.COMPLEX)
              for irrep, m in zip(group.real_irreps().values(), dims)]
    return layout, sum(m * d for m, d, _ in layout)


def _flatten_fiber(w: List[torch.Tensor], layout) -> torch.Tensor:
    r"""A list of :math:`K`-irrep tensors :math:`(*, m_j, d_j)` to one real :math:`(*, \dim W)`."""
    parts = []
    for z, (_, _, cplx) in zip(w, layout):
        r = EF.to_real(z).flatten(-2, -1) if cplx else z
        parts.append(r.flatten(-2, -1))
    return torch.cat(parts, dim=-1)


def _unflatten_fiber(t: torch.Tensor, layout) -> List[torch.Tensor]:
    """The inverse of :func:`_flatten_fiber`."""
    out, offset = [], 0
    for m, d, cplx in layout:
        chunk = t[..., offset:offset + m * d].unflatten(-1, (m, d))
        offset += m * d
        out.append(EF.to_complex(chunk.unflatten(-1, (-1, 2))) if cplx else chunk)
    return out


class BundleAttention(nn.Module):
    r"""
    :math:`G`-equivariant multi-head self-attention on the head set
    :math:`H = \bigsqcup_i G/K_i`, in the form Theorem 1 predicts.

    The architecture is fixed by: an orthogonal representation :math:`V`, a list of stabilizer
    subgroups :math:`K_1, K_2, \dotsc` -- so :math:`|H| = \sum_i |G|/|K_i|` is inferred -- and, per
    orbit, two :math:`K_i`-representations: the attention fiber :math:`W_i` and the value fiber
    :math:`W_i'`. Heads within one orbit share parameters; the head at :math:`gK_i` gets

    .. math::
        M_{gK_i} = \rho_V(g)\, M_{K_i}\, \rho_V(g)^{-1}, \qquad
        R_{gK_i} = \rho_V(g)\, R_{K_i}\, \rho_V(g)^{-1},

    with :math:`M_{K_i} = \phi_q^\top\phi_k` and :math:`R_{K_i} = \phi_o\phi_v` the base-head forms,
    which is exactly the equivariance condition on a bundle-attention configuration
    :math:`(\chi = \mathrm{Id}, E, E')`. The output is summed over all of :math:`H`.

    Args:
        group: the group :math:`G`.
        dims_V: channel widths of :math:`V`, one per irrep of :math:`G` (list or name-keyed dict).
        subgroups: one entry per orbit, each a tuple of arguments for :meth:`Group.subgroup`
            (``('D', 2, 0)`` for :math:`D_4`, ``(2,)`` for :math:`C_n`) or an inclusion.
        attn_fibers: the :math:`W_i`, one per orbit. A dict keyed by :math:`K_i`'s irrep names is
            the preferred form; names left out mean zero.
        value_fibers: the :math:`W_i'`, one per orbit, same format.

    Shape:
        input and output are lists of tensors :math:`(*, L, C_j, d_j)`, one per irrep of :math:`G`,
        the convention the rest of ``equivit.nn`` uses.

    Note:
        Each orbit runs its own :func:`torch.nn.functional.scaled_dot_product_attention`, so the
        temperature is the default :math:`1/\sqrt{\dim W_i}` and genuinely differs between orbits,
        because their head dimensions do. An empty :math:`W_i` gives uniform attention weights
        (:math:`M_{K_i} = 0`), and an empty :math:`W_i'` makes the orbit contribute nothing
        (:math:`R_{K_i} = 0`) -- the :math:`X_0` heads of the theorem.
    """

    def __init__(
        self,
        group: Group,
        dims_V: DimSpec,
        subgroups: Sequence,
        attn_fibers: Sequence[DimSpec],
        value_fibers: Sequence[DimSpec],
    ):
        super().__init__()
        assert len(subgroups) == len(attn_fibers) == len(value_fibers), \
            (f"one attention fiber and one value fiber per orbit: got {len(subgroups)} subgroups, "
             f"{len(attn_fibers)} attention fibers, {len(value_fibers)} value fibers")

        self.group = group
        self.head_set = build_head_set(group, subgroups)
        transporters = head_transporters(self.head_set)

        self.transports = nn.ModuleList(
            OrbitTransport(group, t) for t in transporters
        )
        self.queries = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.values = nn.ModuleList()
        self.attn_layouts, self.value_layouts = [], []

        for inclusion, w_attn, w_value in zip(self.head_set.inclusions, attn_fibers, value_fibers):
            self.queries.append(FiberMap(inclusion, dims_V, w_attn, 'down'))
            self.keys.append(FiberMap(inclusion, dims_V, w_attn, 'down'))
            self.values.append(FiberMap(inclusion, dims_V, w_value, 'both'))

            K = inclusion.source
            self.attn_layouts.append(_fiber_layout(K, self.queries[-1].dims_W))
            self.value_layouts.append(_fiber_layout(K, self.values[-1].dims_W))

        self.dims_V = self.values[0].dims_V if self.values else None

    @property
    def num_heads(self) -> int:
        return len(self.head_set)

    def head_dims(self):
        r""":math:`(\dim W_i, \dim W_i')` per orbit."""
        return [(a[1], v[1]) for a, v in zip(self.attn_layouts, self.value_layouts)]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            x: one tensor per irrep of :math:`G`, of shape :math:`(*, L, C_j, d_j)`.

        Returns:
            the same shapes, summed over every head of every orbit.
        """
        out = None

        for i in range(self.head_set.num_orbits):
            (attn_layout, attn_dim) = self.attn_layouts[i]
            (value_layout, value_dim) = self.value_layouts[i]
            if value_dim == 0:
                continue                                  # R = 0: this orbit contributes nothing

            transport = self.transports[i]

            # (*, L, C_j, d_j) -> (*, L, P, C_j, d_j), the head's own frame
            pulled = transport.pull_back(x)

            # into the fibers, then to a flat head vector: (*, L, P, dim W)
            q = _flatten_fiber(self.queries[i].down(pulled), attn_layout)
            k = _flatten_fiber(self.keys[i].down(pulled), attn_layout)
            v = _flatten_fiber(self.values[i].down(pulled), value_layout)

            # attention runs per head, so the head axis has to precede the token axis
            y = F.scaled_dot_product_attention(
                q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)
            ).transpose(-3, -2)                           # (*, L, P, dim W')

            # back up to V, then out of the head's frame, then sum over the orbit's heads
            y = self.values[i].up(_unflatten_fiber(y, value_layout))
            y = [z.sum(dim=-3) for z in transport.push_forward(y)]

            out = y if out is None else [a + b for a, b in zip(out, y)]

        if out is None:
            return [torch.zeros_like(z) for z in x]
        return out

    def __repr__(self) -> str:
        return (f'BundleAttention({self.group}, dims_V={self.dims_V}, '
                f'num_heads={self.num_heads}, head_dims={self.head_dims()})')
