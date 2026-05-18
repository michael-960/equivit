#!/bin/bash
set -e

source ./env.sh


# for subgroup in C1 C2 C4 D1 D2 D4; do
for subgroup in C1 D1 D4; do

	if [[ "$subgroup" = "C4" ]]; then
		compile=false
	else
		compile=true
	fi

	for depth in 8 2; do

		args=(
			--config-dir=conf 
			--config-name=config
			loggers.mlflow.experiment_name=equivariance-significance

			trainer.log_every_n_steps=50

			model=octic/"$subgroup"/a/small
			model.backbone.config.depth="$depth"
			+model.head.drop_rate=0.1
			+model.backbone.config.transformer_block_config.attn_drop=0.1
			+model.backbone.config.transformer_block_config.drop_path=0.05

			trainer.max_epochs=100

			compile="$compile"

			optimizer.lr=0.001

			data=pcam
			loss_fn._target_=torch.nn.BCEWithLogitsLoss

			data.train_loader.batch_size=256
			data.val_loader.batch_size=256
		)

		python -m equivit.training train-classifier "${args[@]}"

	done

done
