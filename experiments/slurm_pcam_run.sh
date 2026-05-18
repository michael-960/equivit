#!/usr/bin/env bash
#SBATCH --job-name=food-101
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-17

set -e

source ./env.sh


task_id=$SLURM_ARRAY_TASK_ID


subgroups=(C1 C2 C4 D1 D2 D4)

depths=(12 6 2)

num_subgroups=${#subgroups[@]}
num_depths=${#depths[@]}


subgroup_index=$(( task_id / num_depths ))
depth_index=$(( task_id % num_depths ))

subgroup="${subgroups[$subgroup_index]}"
depth="${depths[$depth_index]}"

# C4 has complex numbers
if [[ "$subgroup" = "C4" ]]; then
    compile=false
else
    compile=true
fi


args=(
    --config-dir=conf 
    --config-name=config
    loggers.mlflow.experiment_name=pcam

    trainer.log_every_n_steps=50

    model=octic/"$subgroup"/a/base
    model.backbone.config.depth="$depth"
    +model.head.drop_rate=0.5
    +model.backbone.config.transformer_block_config.attn_drop=0.2
    +model.backbone.config.transformer_block_config.drop_path=0.08

    trainer.max_epochs=100

    compile="$compile"

    optimizer.lr=0.0001

    data=pcam_aug
    loss_fn._target_=torch.nn.BCEWithLogitsLoss

    data.train_loader.batch_size=256
    data.val_loader.batch_size=256
)

srun python -m equivit.training train-classifier "${args[@]}"
