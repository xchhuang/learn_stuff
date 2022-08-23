#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o slurm_outputs/slurm-%j.out

cuda_version=10.2
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/

# cd build
# cmake ..
# make

nvcc add.cu -o add_cuda
./add_cuda


