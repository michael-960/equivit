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

While EquiViT provides a modular API for building custom equivariant vision transformer models (see :doc:`API Reference`, 
in particular :doc:`equivit.nn`), 
we also provide pre-defined model configurations for quick experimentation. 

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
        dims=[64, 64, 64, 64, 128], # D4 has 5 irreps
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
    
