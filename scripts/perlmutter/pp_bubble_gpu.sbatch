#!/bin/bash
# Pipeline-bubble study on Perlmutter GPU nodes: one pipeline stage per node
# (TP=4 inside the node), sweeping interleaving v, microbatches m and the schedule.
#   sbatch -A m1234_g -N 4 scripts/perlmutter/pp_bubble_gpu.sbatch
# Knobs: XP_PP_V ("1 2 4"), XP_PP_M (m/p multipliers, "1 2 4 8"), XP_PP_SCHED ("1f1b gpipe"),
#        XP_REPLAY ("off graph" to compare with CUDA-graph replay), XP_TRANSPORTS ("mpi"),
#        XP_PP_MODEL / XP_PP_LAYERS (model; layers >= nodes * max v), XP_PREC, XP_ITERS.
#SBATCH -J xpsim-pp-bubble-gpu
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh gpu
export XP_OUT=$XP_OUT/pp-bubble-gpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/pp_bubble_cases.sh
: "${XP_PP_MODEL:=--preset gpt3-1.3b --seq 2048 --micro-batch 1}"
: "${XP_PP_LAYERS:=32}" "${XP_PREC:=bf16}"
: "${XP_PP_V:=1 2 4}" "${XP_PP_M:=1 2 4 8}" "${XP_PP_SCHED:=1f1b gpipe}"
: "${XP_ITERS:=6}" "${XP_WARMUP:=2}"
export XP_ITERS XP_WARMUP
xp_pp_bubble 4
xp_summary
