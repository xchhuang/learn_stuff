#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 00:05:00
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --gres gpu:1
#SBATCH -a 1-1

mkdir -p slurm_outputs

cuda_version=11.3
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/

# python tutorial.py

# python void_and_cluster_jax.py --seed=0
# python void_and_cluster_jax.py --seed=${SLURM_ARRAY_TASK_ID}
python sdf.py

