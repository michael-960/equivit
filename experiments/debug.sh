#!/bin/bash
set -e

source ./env.sh


for subgroup in D4 D2 D1 C4 C2 C1; do

	if [[ "$subgroup" = "C4" ]]; then
		compile=false
	else
		compile=true
	fi

	python -m equivit.training train-classifier --config-dir=conf --config-name=config \
		+run_name=DEBUG \
		loggers.mlflow.experiment_name=test-exp \
		model=octic/"$subgroup"/a/extratiny \
		trainer.log_every_n_steps=2 \
		data=pcam_debug\
		trainer.max_epochs=10 \
		compile="$compile" \
		optimizer.lr=0.001 \
		model.backbone.config.depth=2 \
		+model.head.drop_rate=0.1 \
		+model.backbone.config.transformer_block_config.attn_drop=0.1 \
		+model.backbone.config.transformer_block_config.drop_path=0.05

done
