Guide - Groups and Representations
===================================

The module :doc:`equivit.geometry`
provides a collection of classes and functions 
for working with groups and group representations.


A group is an instance of the :class:`equivit.geometry.Group` abstract base class. 
Currently, there are two families of groups that are implemented: cyclic groups
:math:`C_n` (:class:`equivit.geometry.CyclicGroup`) and dihedral groups 
:math:`D_n` (:class:`equivit.geometry.DihedralGroup`).

For convenience, there are some predefined group instances 
exported in the top-level :mod:`equivit` module, such as :attr:`equivit.C4` and :attr:`equivit.D4`.
Note that the cyclic group and dihedral group 
classes do not create a new instance each time the constructor is called,
so, e.g., ``equivit.geometry.CyclicGroup(4) is equivit.C4`` evaluates to ``True``.


Basic Usage
------------

Group elements are accessed via ``__getitem__`` on the group instance:

.. code-block:: python

    import equivit

    group = equivit.D4
    print(group['']) # identity element
    print(group['r']) # rotation by 90 degrees

Groups (at least the built-in ones) can be iterated over:

.. code-block:: python

    for g in group:
        print(g)


The group operation is implemented via the ``*`` operator, and the inverse of a
group element can be obtained via the ``inv()`` method:

.. code-block:: python

    print(group['r'] * group['r']) # rotation by 180 degrees
    print(group['r'].inv()) # rotation by 270 degrees



Group Representations
----------------------

A group representation is an instance of the :class:`equivit.geometry.GroupRepresentation` class.
Unlike the :class:`equivit.geometry.Group` class, one can directly instantiate a
representation object by specifying the representation matrix for each group element.
For example, the following creates the trivial representation of :math:`D_4`:

.. code-block:: python

    import numpy as np
    rep = equivit.geometry.GroupRepresentation(group, {g: np.array([[1.]]) for g in group})


Note:
    The homomorphism property (:math:`\rho(g)\rho(h)=\rho(gh)`) is not automatically checked.


For cyclic and dihedral groups, there are 
factory functions that create representations given 
the images on the generators (:math:`r` for cyclic groups, and :math:`r,t` for dihedral groups).


.. code-block:: python

    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    rep = equivit.geometry.dihedral_group_representation(group, r_matrix=rotation_matrix(np.pi/2), t_matrix=np.array([[1., 0], [0, -1.]]))

    print(rep(group['r'])) # should be the rotation matrix


Finally, we can also get a list of real irreducible representations:

.. code-block:: python

    irreps = equivit.geometry.real_irreducible_representations(group)


This is a dictionary mapping the name of each irrep to the corresponding representation object.

