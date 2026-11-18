#!/usr/bin/env bash
# Functional tests: every parallelism (and mixes) on both transports.
# Usage: scripts/run_tests.sh [binary] [launcher]
#   binary   default bin/xpsim-shim (thread-based MPI shim build)
#   launcher default "" for the shim (ranks via -np), e.g. "mpirun -np" for real MPI
set -u
BIN=${1:-bin/xpsim-shim}
LAUNCH=${2:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
COMMON="--iters 2 --warmup 1 --peak-tflops 0.05"
pass=0; fail=0
run() {  # run <np> <name> <args...>
  local np=$1 name=$2; shift 2
  local cmd
  if [ -z "$LAUNCH" ]; then cmd="$BIN -np $np $COMMON $*"; else cmd="$LAUNCH $np $BIN $COMMON $*"; fi
  if out=$(timeout 300 $cmd 2>&1); then
    local it=$(echo "$out" | grep "iteration time" | awk '{print $4}')
    local mfu=$(echo "$out" | grep "MFU / HFU" | awk '{print $5}')
    printf "PASS  %-34s iter %8s s  MFU %6s %%\n" "$name" "$it" "$mfu"; pass=$((pass+1))
  else
    printf "FAIL  %-34s\n" "$name"; echo "$out" | tail -15; fail=$((fail+1))
  fi
}
run 4 "selftest mpi"            --selftest --ranks-per-node 2
run 2 "selftest hier 2x2"       --selftest --transport hier --workers-per-node 2
run 1 "single"                  --preset tiny
run 4 "DP4 (ZeRO-0)"            --preset tiny --dp 4
run 4 "DP4 overlap"             --preset tiny --dp 4 --overlap
run 4 "FSDP4 (ZeRO-3)"          --preset tiny --fsdp 4
run 4 "FSDP4 prefetch+overlap"  --preset tiny --fsdp 4 --overlap
run 4 "ZeRO-1 x4"               --preset tiny --fsdp 4 --zero 1
run 4 "ZeRO-2 x4"               --preset tiny --fsdp 4 --zero 2
run 4 "HSDP 2x2"                --preset tiny --dp 2 --fsdp 2
run 4 "TP4"                     --preset tiny --tp 4
run 4 "TP4+SP"                  --preset tiny --tp 4 --sp
run 4 "PP4 1f1b m4"             --preset tiny --layers 4 --pp 4 --microbatches 4
run 2 "PP2 gpipe"               --preset tiny --pp 2 --microbatches 4 --pp-schedule gpipe
run 4 "CP4 causal"              --preset tiny --cp 4
run 4 "CP2 x TP2 (GQA kv=2)"    --preset tiny --cp 2 --tp 2 --kv-heads 2
run 4 "EP4 MoE uniform"         --preset moe-tiny --dp 4 --ep 4
run 4 "EP2 x DP2 zipf cf1.25"   --preset moe-tiny --dp 4 --ep 2 --router zipf --capacity-factor 1.25
run 4 "MoE TP2+EP2 (no SP)"     --preset moe-tiny --tp 2 --dp 2 --ep 2
run 4 "MoE TP2+SP+EP2 FSDP"     --preset moe-tiny --tp 2 --sp --fsdp 2 --ep 2
run 8 "3D: TP2 PP2 DP2"         --preset tiny --tp 2 --pp 2 --dp 2 --microbatches 4
run 8 "4D: TP2 SP PP2 FSDP2 rc" --preset tiny --tp 2 --sp --pp 2 --fsdp 2 --recompute --microbatches 3
run 16 "5D: TP2 CP2 PP2 DP2 EP2 MoE" --preset moe-tiny --tp 2 --cp 2 --pp 2 --dp 2 --ep 2 --microbatches 2
run 4 "bf16 mixed, fp32 grads"  --preset tiny --dp 2 --tp 2 --precision bf16
run 4 "fp16 + loss scale"       --preset tiny --fsdp 4 --precision fp16 --grad-dtype compute
run 4 "fp64"                    --preset tiny --dp 2 --tp 2 --precision fp64
run 4 "sync-collectives"        --preset tiny --tp 2 --dp 2 --sync-collectives
run 2 "hier: TP2 x DP2 (2 nodes)"     --preset tiny --tp 2 --dp 2 --transport hier --workers-per-node 2
run 2 "hier: FSDP4 overlap"           --preset tiny --fsdp 4 --overlap --transport hier --workers-per-node 2
run 2 "hier: PP2 x TP2"               --preset tiny --pp 2 --tp 2 --microbatches 4 --transport hier --workers-per-node 2
run 2 "hier: EP4 local router"        --preset moe-tiny --dp 4 --ep 4 --router local --transport hier --workers-per-node 2
run 2 "hier: EP4 remote router"       --preset moe-tiny --dp 4 --ep 4 --router remote --transport hier --workers-per-node 2
run 4 "hier: EP8 node-limited M=1"    --preset moe-tiny --dp 8 --ep 8 --router node-limited --max-nodes 1 --transport hier --workers-per-node 2
run 4 "hier: EP8 distance ring b=2"   --preset moe-tiny --dp 8 --ep 8 --router distance --distance-beta 2 --topology ring --transport hier --workers-per-node 2
run 2 "hier: CP2 TP2 DP2 bf16 (2 nodes)" --preset tiny --cp 2 --tp 2 --dp 2 --precision bf16 --transport hier --workers-per-node 4
# trace record / in-process replay / save + reload round trips (scripts/replay_check.sh)
rt() {
  local name=$1; shift
  if out=$(BIN=$BIN LAUNCH=$LAUNCH "$(dirname "$0")/replay_check.sh" "$@" 2>&1); then
    printf "PASS  %-34s %s\n" "replay: $name" "$(echo "$out" | cut -c42- | cut -d'|' -f3)"; pass=$((pass+1))
  else printf "FAIL  %-34s\n" "replay: $name"; echo "$out" | tail -6; fail=$((fail+1)); fi
}
rt "TP4+SP"                 "replay TP4+SP" 4 --preset tiny --tp 4 --sp
rt "FSDP4 overlap"          "replay FSDP4" 4 --preset tiny --fsdp 4 --overlap
rt "PP4 1f1b"               "replay PP4" 4 --preset tiny --layers 4 --pp 4 --microbatches 4
rt "MoE TP2+SP+EP2 FSDP"    "replay MoE" 4 --preset moe-tiny --tp 2 --sp --fsdp 2 --ep 2
rt "4D rc bf16"             "replay 4D" 8 --preset tiny --tp 2 --sp --pp 2 --fsdp 2 --recompute --microbatches 3 --precision bf16
rt "fp16 loss scale"        "replay fp16" 4 --preset tiny --fsdp 4 --precision fp16 --grad-dtype compute
rt "hier EP4 local"         "replay hier EP4" 2 --preset moe-tiny --dp 4 --ep 4 --router local --transport hier --workers-per-node 2
echo "----"; echo "passed $pass, failed $fail"
[ $fail -eq 0 ]
