#!/bin/bash
# Weak/strong scaling on Perlmutter GPU nodes (4x A100 per node), both transports:
#   mpi : one rank per GPU, CUDA-aware Cray MPICH collectives
#   hier: one rank per node, VMM zero-copy intra-node + CUDA-aware MPI inter-node
# Submit from the repository root (GPU account ends in _g); change -N for the largest scale:
#   sbatch -A m1234_g -N 16 scripts/perlmutter/scaling_gpu.sbatch
# Knobs (export before sbatch, or edit):
#   XP_NODES="1 2 4 8 16"  XP_STRATEGIES="dp fsdp ..."  XP_TRANSPORTS="mpi hier"
#   XP_MODEL, XP_MOE_MODEL, XP_PREC (bf16), XP_MB (microbatches, weak), XP_GB (global batch, strong)
#   XP_ITERS / XP_WARMUP, XP_PEAK (TFLOP/s per GPU for MFU; default A100 dense peak)
#   80 GB A100 nodes: add  -C "gpu&hbm80g"  on the sbatch command line.
#SBATCH -J xpsim-scaling-gpu
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -o %x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/perlmutter/env.sh gpu
export XP_OUT=$XP_OUT/scaling-gpu-${SLURM_JOB_ID:-local}
source scripts/perlmutter/common.sh
source scripts/perlmutter/scaling_cases.sh

: "${XP_MODEL:=--preset gpt3-1.3b --layers 8 --seq 2048 --micro-batch 1}"
: "${XP_MICRO:=1}"                 # must match --micro-batch in XP_MODEL
: "${XP_MOE_MODEL:=--preset moe-64e --layers 4 --seq 2048 --micro-batch 1}"
: "${XP_EXPERTS:=64}"              # must match the MoE model
: "${XP_PREC:=bf16}"
: "${XP_MB:=4}" "${XP_PP_MB:=8}" "${XP_GB:=256}"
: "${XP_CP_EXTRA:=--seq 8192}"     # long-context case for cp_dp
: "${XP_ITERS:=8}" "${XP_WARMUP:=2}"
export XP_ITERS XP_WARMUP
echo "xpsim GPU scaling, job ${SLURM_JOB_ID:-?}: results in $XP_OUT"
xp_scaling 4
xp_summary
