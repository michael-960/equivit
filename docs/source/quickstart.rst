Quickstart
==========


Installation
------------

First, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/michael-960/equivit

.. code-block:: bash

    cd equivit


Then, install the package locally using pip:

.. code-block:: bash

    pip install -e .




Building a model
--------------------------

While EquiViT provides a modular API for building custom equivariant vision transformer models (see :doc:`modules`, 
in particular :doc:`equivit.nn`), 
we also provide pre-defined architectures for quick experimentation. 

For example, we will show in the following how to build a :math:`D_4`-equivariant
vision transformer (`arXiv:2505.15441 <https://arxiv.org/abs/2505.15441>`_).

First, we set up the backbone of the vision transformer.
This is a module that takes as input a tensor of shape :math:`(B, C, L)`, where
:math:`L` is the number of pixels (i.e., ``img_size**2``),
and outputs a list of tensors, each of shape :math:`(B, L, C_i, d_i)`, where
:math:`C_i` is the number of channels for each irrep (specifie by ``dims``), and :math:`d_i` is the complex dimension of each irrep.
(See :doc:`models/octic` for details on the model architecture and the meaning of each configuration parameter.)

.. code-block:: python

    import equivit

    group = equivit.D4 # get a reference to the D4 group

    config = equivit.models.OcticViTBackboneConfig(
        img_size=32,
        patch_size=8,
        dims=[64, 64, 64, 64, 128], # D4 has 5 irreps: A1, A2, B1, B2, E1
        subgroup=('D', 4, 0), # for full D4 symmetry
        depth=12, # number of transformer blocks
        transformer_block_config=equivit.nn.EquivariantTransformerBlockConfig(
            homogeneous_space_copies=[16]*len(group.all_homogeneous_space_actions()),
            num_heads=[8, 8, 8, 8, 8], # number of attention heads for each irrep
            attn_type="irrepwise" # see equivit.nn for details
        )
    )

    backbone = equivit.models.OcticViTBackbone(config)


Next, we build the full model by adding a classification head on top of the
backbone.

.. code-block:: python

    import torch

    head = equivit.nn.InvariantClassificationHead(
        dim=sum(config.dims), num_logits=10
    )

    model = torch.nn.Sequential(backbone, head)
    



Testing Invariance
--------------------

We will check that the output of ``model`` is invariant under the :math:`D_4` group action.

To do this, we first create an instance of :class:`equivit.Square`, which represents
the square grid of ``config.img_size * config.img_size`` pixels (in our case, ``32*32``),
and encodes the :math:`D_4` action.

Here, ``square.points`` is an ``np.ndarray`` of shape :math:`(L, 2)`, 
where :math:`L` is the number of pixels, and each row contains the 
coordinates of a pixel. ``square.action`` is a function that takes as input
a group element (e.g., ``group['r']`` for a 90 degree rotation) and returns a
list of integers representing how the group element permutes the pixel indices.

.. code-block:: python

    import matplotlib.pyplot as plt

    square = equivit.Square(config.img_size - 1) # create a square grid

    plt.scatter(*square.points.T)

    print(square.action(group['r']))



Finally, we create a random batch of images (as a tensor of shape :math:`(B, C, L)`) 
and apply a group element (here a rotation) to the input. 
We then check that the output of the model is the same for both the original and transformed inputs.

.. code-block:: python

    x = torch.randn(2, 3, config.img_size**2) # batch of 2 images, each with 3 channels and size 32x32
    g_x = x[..., square.action(group['r'].inv())] # apply a group element (here, a rotation) to the input


    print(model(x))
    print(model(g_x))

    print('Error:')
    print((model(x) - model(g_x)).abs().max().item()) # check that the outputs are the same (up to numerical precision)


The last line should print a very small number (:math:`\lesssim 10^{-6}`).
