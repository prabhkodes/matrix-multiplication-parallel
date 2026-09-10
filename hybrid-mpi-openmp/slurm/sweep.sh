#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=14

#SBATCH --hint=nomultithread

#SBATCH --time=00:20:00
#SBATCH --job-name=mpi_test
#SBATCH --output=logs/%x_%j.out

cd "${SLURM_SUBMIT_DIR:-$PWD}"


module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load openblas/0.3.26--gcc--12.2.0

RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

which mpic++ && mpic++ --version | head -n1
mpic++ -std=c++20 -O3 -march=native -Wall -Wextra -I./include -fopenmp \
       src/main.cpp -lopenblas -o app_mpi_omp.x

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OPENBLAS_NUM_THREADS=1

NODES=${SLURM_NNODES:-1}
PROCS=$(( RANKS_PER_NODE * NODES ))

echo "Nodes=${NODES}  Ranks/node=${RANKS_PER_NODE}  MPI ranks=${PROCS}  OMP threads=${OMP_NUM_THREADS}"
echo "Matrix N = 224000"

echo "About to start srun"
srun --cpu-bind=cores ./app_mpi_omp.x


