#!/usr/bin/env bash
# Build xpsim for Perlmutter on a login node:
#   scripts/perlmutter/build.sh          # both
#   scripts/perlmutter/build.sh gpu      # GPU only (CUDA, sm_80, CUDA-aware Cray MPICH, VMM)
#   scripts/perlmutter/build.sh cpu      # CPU only (Cray LibSci CBLAS + OpenMP)
set -euo pipefail
cd "$(dirname "$0")/../.."
what=${1:-all}
J=${J:-16}
if [ "$what" = all ] || [ "$what" = gpu ]; then
  ( source scripts/perlmutter/env.sh gpu
    make MPICXX=CC CUDA=1 SM=80 NVCC=nvcc CUDA_HOME="$CUDA_HOME" BLAS=libsci BIN=bin/xpsim-gpu -j"$J" )
fi
if [ "$what" = all ] || [ "$what" = cpu ]; then
  ( source scripts/perlmutter/env.sh cpu
    make MPICXX=CC CUDA=0 BLAS=libsci BIN=bin/xpsim-cpu -j"$J" )
fi
ls -l bin/
