#!/bin/bash

#SBATCH --job-name="cublas"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ICT25_MHPC_0
#SBATCH --output=logs/cublas_10000_%x_%j.out 
#SBATCH --error=logs/cublas_10000_%x_%j.err
#SBATCH --qos=boost_qos_dbg

mkdir -p logs

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cuda/12.2
module load nvhpc/24.5

export OMPI_CXX=g++
export MPICH_CXX=g++

echo "[MPI] which mpic++:"
which mpic++ && mpic++ --version | head -n 1
echo "[CUDA] nvcc version:"
nvcc --version | head -n 4

echo
echo "[BUILD] Compiling..."

mpic++ -std=c++20 -O3 -march=native \
    -Iinclude -I/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/compilers/lib/ \
    src/main.cpp -o app.x \
    -L/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/math_libs/lib64 -lcublas -lcudart  -lnvToolsExt

echo "[BUILD] done."


# export OPENBLAS_NUM_THREADS=1

PROCS=${SLURM_NTASKS}

echo "Nodes = $SLURM_JOB_NUM_NODES, Total Procs = $PROCS, GPUs per node = 4"
echo "===================================="
echo " Matrix Multiply (cuBLAS) Sweep"
echo " Start Time : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "===================================="

srun  ./app.x

echo
echo "===================================="
echo " End Time : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "===================================="