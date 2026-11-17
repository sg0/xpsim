#!/bin/bash
# Functional tests on Perlmutter CPU nodes (port of scripts/run_tests.sh).
#   sbatch -A m1234 scripts/perlmutter/tests_cpu.sbatch
# XP_TPN ranks per node (default 4 here so that 16-worker cases span 4 nodes);
# each rank gets 128/XP_TPN cores of OpenMP threads.
#SBATCH -J xpsim-tests-cpu
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh cpu
export XP_OUT=$XP_OUT/tests-cpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/tests_cases.sh
export XP_TPN=${XP_TPN:-4}
echo "xpsim CPU functional tests: ${SLURM_NNODES:-?} nodes, $XP_TPN ranks/node, logs in $XP_OUT/logs"
xp_test_matrix
xp_summary
