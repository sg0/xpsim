#!/bin/bash
# Pipeline-bubble study on Perlmutter CPU nodes: one pipeline stage per node,
# XP_TPN-way TP inside the node (default 8 ranks/node, one per NUMA domain).
#   sbatch -A m1234 -N 4 scripts/perlmutter/pp_bubble_cpu.sbatch
# Same knobs as pp_bubble_gpu.sbatch.
#SBATCH -J xpsim-pp-bubble-cpu
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh cpu
export XP_OUT=$XP_OUT/pp-bubble-cpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/pp_bubble_cases.sh
: "${XP_TPN:=8}"
: "${XP_PP_MODEL:=--preset small --micro-batch 2}"
: "${XP_PP_LAYERS:=16}" "${XP_PREC:=fp32}"
: "${XP_PP_V:=1 2 4}" "${XP_PP_M:=1 2 4 8}" "${XP_PP_SCHED:=1f1b gpipe}"
: "${XP_ITERS:=4}" "${XP_WARMUP:=1}"
export XP_ITERS XP_WARMUP
xp_pp_bubble "$XP_TPN"
xp_summary
