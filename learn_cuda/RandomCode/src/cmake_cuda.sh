#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p gpu20
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -o slurm_outputs/slurm-%j.out

cuda_version=10.0
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/

mkdir build
cd build
cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_30” ..
make

