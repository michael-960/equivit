# EquiViT experiments on SLURM clusters


## 0. Prerequisites
Preferably, the python version should be 3.12.13 (where everything is tested).


## 1. Install equivit

At the root of the repository, run:

```bash
python -m pip install -e .
```



## 2. Set up `env.sh`

Go to the `experiments` directory. Then:

```bash
cp env_example.sh env.sh
```

Edit the values of the environment variables to specify api key etc.


## 3. Download data

Take a look at `download_data.sh`, make sure the SLURM parameters are set correctly. Also make sure that `EXPERIMENT_DATA_DIR` is set in `env.sh`.
Then submit the script:

```bash
sbatch download_data.sh
```

This will download the datasets to `$EXPERIMENT_DATA_DIR`.


## 4. Test run

The `slurm_debug.sh` script submits an array of 6 jobs, each training a tiny
equivariant transformer on a tiny dataset for 10 epochs.
Before submitting, make sure the SLURM parameters in `slurm_debug.sh`
are set correctly. Then run:

```bash
sbatch slurm_debug.sh
```


## 5. Start training
Now, start training on the PCAM dataset (as usual inspect the script and make necessary modifications):
```bash
sbatch slurm_pcam.sh
```
This submits an array of 18 jobs, training 
$G$-equivariant ViTs of depth $d$, where
$$

    G\in\{C_1, C_2, C_4, D_1, D_2, D_4\}
$$
$$
d\in \{12, 6, 2\}.
$$
The PCam dataset consists of $327680$ images
extracted from histopathologic scans of lymph node sections, each with a binary label. The image size is $96\times 96$. 




## Notes
For a smoke test, include or modify the following arguments:

```
args=(
    +run_name=DEBUG
    ...
    ...
    (other arguments)
    ...
    ...
    trainer.max_epochs=1

    ...
    +trainer.limit_train_batches=5
    +trainer.limit_val_batches=2
)
```