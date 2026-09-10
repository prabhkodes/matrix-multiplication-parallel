#!/bin/bash
#SBATCH -A ICT25_MHPC
#SBATCH -p dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=112
#SBATCH --time=00:10:00
#SBATCH --job-name=mpi_test
#SBATCH --output=logs/%x_%j.out

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load openblas/0.3.26--gcc--12.2.0

# ---- QUICK SWITCHES ----
MODE=hybrid            # pure | hybrid
RANKS_PER_NODE=28      # pure: 1/2/4/8/14/16/28/56/112 ; hybrid: 56(2t),28(4t),14(8t)
OMP_NUM_THREADS=4      # pure: 1 ; hybrid: 2/4/8
# ------------------------

which mpic++ && mpic++ --version | head -n1
mpic++ -std=c++20 -O3 -march=native -Wall -Wextra -I./include -fopenmp \
       src/main.cpp -lopenblas -o app_mpi_omp.x

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

NODES=${SLURM_NNODES:-1}
PROCS=$(( RANKS_PER_NODE * NODES ))

echo "Nodes=${NODES}  Ranks/node=${RANKS_PER_NODE}  MPI ranks=${PROCS}  OMP threads=${OMP_NUM_THREADS}"

if [[ "$MODE" == "pure" ]]; then
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  mpirun --bind-to core --map-by ppr:${RANKS_PER_NODE}:node \
         -n ${PROCS} ./app_mpi_omp.x
else
  mpirun --bind-to core --map-by ppr:${RANKS_PER_NODE}:node:pe=${OMP_NUM_THREADS} \
         -n ${PROCS} ./app_mpi_omp.x
fi