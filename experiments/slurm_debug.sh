#!/usr/bin/env bash
#SBATCH --job-name=debug
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5


set -e

source ./env.sh


subgroups=(C1 C2 C4 D1 D2 D4)

subgroup="${subgroups[$SLURM_ARRAY_TASK_ID]}"


# C4 has complex numbers
if [[ "$subgroup" = "C4" ]]; then
    compile=false
else
    compile=true
fi

srun python -m equivit.training train-classifier --config-dir=conf --config-name=config \
    +run_name=DEBUG \
    loggers.mlflow.experiment_name=test-exp \
    model=octic/"$subgroup"/a/extratiny \
    trainer.log_every_n_steps=2 \
    data=pcam_debug\
    loss_fn._target_=torch.nn.BCEWithLogitsLoss \
    trainer.max_epochs=10 \
    compile="$compile" \
    optimizer.lr=0.001 \
    model.backbone.config.depth=2 \
    +model.head.drop_rate=0.1 \
    +model.backbone.config.transformer_block_config.attn_drop=0.1 \
    +model.backbone.config.transformer_block_config.drop_path=0.05

