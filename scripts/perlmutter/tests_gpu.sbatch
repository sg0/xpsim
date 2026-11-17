#!/bin/bash
# Functional tests on Perlmutter GPU nodes (port of scripts/run_tests.sh).
# Submit from the repository root with your GPU account (must end in _g):
#   sbatch -A m1234_g scripts/perlmutter/tests_gpu.sbatch
#SBATCH -J xpsim-tests-gpu
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh gpu
export XP_OUT=$XP_OUT/tests-gpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/tests_cases.sh
export XP_TPN=4   # one rank per A100 for the mpi transport
echo "xpsim GPU functional tests: ${SLURM_NNODES:-?} nodes, logs in $XP_OUT/logs"
xp_test_matrix
xp_summary
