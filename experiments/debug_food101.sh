#!/bin/bash
set -e

source ./env.sh


subgroup=C4
depth=4
compile=false

args=(
	--config-dir=conf 
	--config-name=config

	+run_name=DEBUG
	loggers.mlflow.experiment_name=food101

	trainer.log_every_n_steps=50
	trainer.max_epochs=100
	+trainer.gradient_clip_val=1.0
	+trainer.limit_train_batches=1.0
	+trainer.limit_val_batches=10

	model=octic/"$subgroup"/a/small
	model.backbone.config.depth="$depth"
	model.backbone.config.tokenizer_config.patch_size=16
	+model.head.drop_rate=0.3
	+model.backbone.config.transformer_block_config.attn_drop=0.1
	+model.backbone.config.transformer_block_config.drop_path=0.1

	compile="$compile"

	optimizer.lr=0.0001

	data=food101_256
	loss_fn._target_=torch.nn.CrossEntropyLoss

	data.train_loader.batch_size=64
	data.val_loader.batch_size=64
)

python -m equivit.training train-classifier "${args[@]}"

