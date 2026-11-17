#!/usr/bin/env bash
# Per-rank wrapper: with 4 ranks per GPU node and --gpu-bind=none, give local rank i
# GPU 3-i (the GPU closest to the cores Slurm binds rank i to on Perlmutter).
if [ "${XP_SELECT_GPU:-0}" = 1 ]; then export CUDA_VISIBLE_DEVICES=$((3 - SLURM_LOCALID)); fi
exec "$@"
