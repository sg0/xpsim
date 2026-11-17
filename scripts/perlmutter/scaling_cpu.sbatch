#!/bin/bash
# Weak/strong scaling on Perlmutter CPU nodes (2x EPYC 7763, 128 cores, 8 NUMA domains).
# Default layout: XP_TPN=8 ranks (workers) per node = one per NUMA domain, 16 OpenMP
# threads each on cores; the hier transport runs 1 rank/node with 8 worker threads.
#   sbatch -A m1234 -N 16 scripts/perlmutter/scaling_cpu.sbatch
# Same knobs as scaling_gpu.sbatch (XP_NODES, XP_STRATEGIES, XP_TRANSPORTS, XP_MODEL, ...).
#SBATCH -J xpsim-scaling-cpu
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh cpu
export XP_OUT=$XP_OUT/scaling-cpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/scaling_cases.sh

: "${XP_TPN:=8}"
: "${XP_MODEL:=--preset small --micro-batch 2}"
: "${XP_MICRO:=2}"
: "${XP_MOE_MODEL:=--preset small --experts 16 --topk 2 --micro-batch 2}"
: "${XP_EXPERTS:=16}"
: "${XP_PREC:=fp32}"
: "${XP_MB:=4}" "${XP_PP_MB:=8}" "${XP_GB:=512}"
: "${XP_CP_EXTRA:=--seq 1024}"
: "${XP_ITERS:=5}" "${XP_WARMUP:=1}"
export XP_ITERS XP_WARMUP
echo "xpsim CPU scaling, job ${SLURM_JOB_ID:-?}: $XP_TPN workers/node, results in $XP_OUT"
xp_scaling "$XP_TPN"
xp_summary
