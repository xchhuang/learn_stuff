#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o slurm_outputs/slurm-%j.out

# rsync -a RandomCode xhuang@slurm-submit.mpi-inf.mpg.de:/HPS/PatternSynthesis/work/github/learn_stuff/learn_cuda
# srun -p gpu20 --gres gpu:1 ./cmake_cuda.sh

cuda_version=10.2
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/

# cd build
# cmake ..
# make


nvcc add.cu -o add_cuda
nvprof ./add_cuda


