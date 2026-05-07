Guide - Training an Equivariant ViT
=============================================


This guide provides a step-by-step walkthrough of how to train an equivariant
ViT model using the EquiViT library.

The :module:`equivit.training` module provides high-level tools and functions
for training an equivariant ViT model using 

- Hydra/Omegaconf for config management
- PyTorch Lightning for training loops, model checkpointing, and logging.


1. Generate model configs
--------------------------
If you cloned the EquiViT repository, you can find example configs in ``equivit/experiments/conf``.
Otherwise, you can also manually generate the predefined configs:

.. code-block:: bash

   python -m equivit.training gen-model-configs conf

This will populate the ``conf`` directory with example configs for building 
:math:`D_4`-equivariant and :math:`D_6`-equivariant ViTs (see
:doc:`models/octic` and :doc:`models/honey` for details on the model
architectures). For example, you should see the following 
in ``conf/model/octic/D4/a/base.yaml``:


.. code-block:: yaml

   _target_: equivit.models.build_named_sequential
   preprocess:
   _target_: torch.nn.Flatten
   start_dim: -2
   end_dim: -1
   backbone:
   _target_: equivit.models.OcticViTBackbone
   config:
      _target_: equivit.models.OcticViTBackboneConfig
      depth: 12
      dims:
         A1: 72
         A2: 72
         B1: 72
         B2: 72
         E1: 144
      subgroup: [D, 4, 0]
      tokenizer_config:
         _target_: equivit.models.OcticTokenizeConfig
         img_size: 256
         patch_size: 16
         in_channels: 3
      transformer_block_config:
         _target_: equivit.nn.EquivariantTransformerBlockConfig
         attn_type: coupled
         num_heads: 12
         homogeneous_space_copies: [288, 0, 0, 0, 0, 0, 0, 0]
         activation: gelu
   head:
   _target_: equivit.nn.InvariantClassificationHead
   dim: 432
   num_logits: ${oc.select:data.num_logits,10}


This specific config will create a :math:`D_4`-equivariant ViT with 12
:doc:`nn/EquivariantTransformerBlock` layers.
See :doc:`models/octic` for more details on the model architecture and config options.



2. Set up data config
----------------------

Next, we need to configure the data loading pipeline. This involves specifying
a ``train_loader`` and ``val_loader`` in the config, which should be instances of 
:class:`torch.utils.data.DataLoader`. 

For example, we can set up the data loaders for the PatternNet dataset 
by adding the following to ``conf/data/patternnet.yaml``
(:class:`equivit.data.PatternNet` is a wrapper around the PatternNet dataset, which is 
accessed via HuggingFace):

.. code-block:: yaml

   train_loader:
   _target_: torch.utils.data.DataLoader
   dataset: 
      _target_: equivit.data.PatternNet
      split: "train"

      transform: 
         _target_: torchvision.transforms.Compose
         transforms: 
         - _target_: torchvision.transforms.ToTensor
         - _target_: torchvision.transforms.RandomRotation
            degrees: [-30, 30]

   batch_size: 16
   num_workers: 4
   shuffle: true


   val_loader:
   _target_: torch.utils.data.DataLoader
   dataset: 
      _target_: equivit.data.PatternNet
      split: "val"

      transform: 
         _target_: torchvision.transforms.Compose
         transforms: 
         - _target_: torchvision.transforms.ToTensor

   batch_size: 16
   num_workers: 4
   shuffle: false

   num_logits: 38
   img_size: 256


Note: ``num_logits`` and ``img_size`` are technically redundant and 
not strictly necessary to specify in the data config, but they are used by the
model config to set the number of output logits and the image size for the
tokenizer, respectively.


3. Set up training config
--------------------------

Finally, place the following in ``conf/config.yaml`` to set up the training config:

.. code-block:: yaml

   defaults:
   - data: patternnet
   - model: octic/D4/a/base
   - _self_

   compile: true


   optimizer:
      _target_: torch.optim.AdamW
      _partial_: true
      lr: 0.001
      betas: [0.9, 0.999]

   loss_fn:
      _target_: torch.nn.CrossEntropyLoss


   trainer:
      _target_: lightning.Trainer
         max_epochs: 50
         accelerator: auto
         devices: auto
         enable_progress_bar: true


   callbacks:
      model_checkpoint:
         _target_: pytorch_lightning.callbacks.ModelCheckpoint
            dirpath:  "${hydra:runtime.output_dir}/checkpoints"
            monitor: val_loss
            mode: min
            save_top_k: 2
            save_last: true

   loggers:
      mlflow:
         _target_: pytorch_lightning.loggers.MLFlowLogger
         experiment_name: "equivit-test"
         tracking_uri: "file:${hydra:runtime.cwd}/mlruns"


Note:
   - ``comple: true`` means that the model will be compiled using ``torch.compile``.
   - ``callbacks`` and ``loggers`` are optional, but they provide useful functionality for 
     checkpointing and logging training metrics. You can customize these as needed.
     The keys under ``callbacks`` and ``loggers`` (e.g. ``model_checkpoint`` and ``mlflow``) are arbitrary.




4. Start training
------------------

Your working directory should now look something like this:

.. code-block:: bash

   conf/
   ├── config.yaml
   ├── data
   │   └── patternnet.yaml
   └── model
       └── octic
           └── D4
               └── a
                   └── base.yaml


You can start training the model by running the following command:

.. code-block:: bash

   python -m equivit.training train-classifer --config-dir=conf --config-name=config


Since we set up the MLFlow logger in our config, you should see training metrics
being logged to the ``mlruns`` directory in your current working directory. You
can view these metrics using the MLFlow UI by running:


.. code-block:: bash

   mlflow server --port 5000

You can then access the MLFlow UI at ``http://localhost:5000``.


Note: the default model config does not have any dropout. 
You can easily change this by overriding the relevant config options in your command. 
For example:

.. code-block:: bash

   python train.py +model.head.drop_rate=0.1 +model.backbone.config.transformer_block_config.attn_drop=0.1 +model.backbone.config.transformer_block_config.drop_path=0.05


