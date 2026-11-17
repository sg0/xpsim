#!/usr/bin/env bash
# Example parallelism sweep -> one CSV row per configuration (results/sweep.csv).
# Edit LAUNCH / BIN / MODEL for your system, e.g.
#   LAUNCH="srun -N 2 --ntasks-per-node=4" BIN=bin/xpsim ./scripts/sweep.sh            (mpi transport)
#   LAUNCH="srun -N 2 --ntasks-per-node=1" EXTRA="--transport hier" ./scripts/sweep.sh  (VMM + CUDA-aware MPI)
set -eu
BIN=${BIN:-bin/xpsim}
LAUNCH=${LAUNCH:-mpirun -np 8}
MODEL=${MODEL:---preset gpt3-1.3b --layers 8 --seq 2048 --micro-batch 1 --microbatches 8}
DEV=${DEV:---device gpu --precision bf16}
EXTRA=${EXTRA:-}
OUT=${OUT:-results/sweep.csv}
mkdir -p "$(dirname "$OUT")"
run() { local label=$1; shift; echo "== $label"; $LAUNCH $BIN $MODEL $DEV $EXTRA --iters 5 --warmup 2 --label "$label" --csv "$OUT" "$@" | grep -E "iteration time|throughput|MFU"; }

run dp8            --dp 8
run dp8-overlap    --dp 8 --overlap
run fsdp8          --fsdp 8 --overlap
run zero1-8        --fsdp 8 --zero 1
run hsdp2x4        --dp 2 --fsdp 4 --overlap
run tp8-sp         --tp 8 --sp
run tp4-dp2        --tp 4 --sp --dp 2
run tp2-pp2-dp2    --tp 2 --sp --pp 2 --dp 2
run pp4-dp2-1f1b   --pp 4 --dp 2
run pp4-dp2-gpipe  --pp 4 --dp 2 --pp-schedule gpipe
run cp2-tp4        --cp 2 --tp 4 --sp
# MoE: same world, expert parallelism and router locality
MODEL="--preset moe-64e --layers 4 --seq 2048 --micro-batch 1 --microbatches 4"
for r in uniform "local --locality 0.9" "remote --locality 0.9" "node-limited --max-nodes 1" "zipf --zipf-alpha 1.2"; do
  run "moe-ep8-$(echo $r | tr ' ' '_')" --dp 8 --ep 8 --router $r
done
echo "results in $OUT"
