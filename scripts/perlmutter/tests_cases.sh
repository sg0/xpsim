# Functional test matrix (Perlmutter port of scripts/run_tests.sh). Sourced by
# tests_gpu.sbatch / tests_cpu.sbatch after env.sh and common.sh.
# Worker counts are mapped onto nodes with XP_TPN ranks per node (mpi transport)
# or <wpn> workers per rank (hier transport, one rank per node).

: "${XP_TPN:=4}"
# mpi transport: W workers -> ceil(W/TPN) nodes x min(W,TPN) ranks/node
t_mpi() { local name=$1 W=$2; shift 2
  local tpn=$(( W < XP_TPN ? W : XP_TPN )); local nodes=$(( (W + tpn - 1) / tpn ))
  xp_case "$name [mpi ${nodes}x${tpn}]" "$nodes" "$tpn" mpi 1 -- $PEAK "$@"; }
# hier transport: W workers, wpn per node -> W/wpn nodes x 1 rank
t_hier() { local name=$1 W=$2 wpn=$3; shift 3
  xp_case "$name [hier $((W / wpn))x${wpn}]" $((W / wpn)) 1 hier "$wpn" -- $PEAK "$@"; }

xp_test_matrix() {
  PEAK=$(xp_peak_args fp32)
  t_mpi  "selftest"                     8  --selftest
  t_hier "selftest"                     8 4 --selftest
  t_mpi  "single"                       1  --preset tiny
  t_mpi  "DP4 (ZeRO-0)"                 4  --preset tiny --dp 4
  t_mpi  "DP4 overlap"                  4  --preset tiny --dp 4 --overlap
  t_mpi  "FSDP4 (ZeRO-3)"               4  --preset tiny --fsdp 4
  t_mpi  "FSDP4 prefetch+overlap"       4  --preset tiny --fsdp 4 --overlap
  t_mpi  "ZeRO-1 x4"                    4  --preset tiny --fsdp 4 --zero 1
  t_mpi  "ZeRO-2 x4"                    4  --preset tiny --fsdp 4 --zero 2
  t_mpi  "HSDP 2x2"                     4  --preset tiny --dp 2 --fsdp 2
  t_mpi  "TP4"                          4  --preset tiny --tp 4
  t_mpi  "TP4+SP"                       4  --preset tiny --tp 4 --sp
  t_mpi  "PP4 1f1b m4"                  4  --preset tiny --layers 4 --pp 4 --microbatches 4
  t_mpi  "PP2 gpipe"                    2  --preset tiny --pp 2 --microbatches 4 --pp-schedule gpipe
  t_mpi  "PP4 interleaved v2 m8"        4  --preset tiny --layers 8 --pp 4 --pp-virtual 2 --microbatches 8
  t_mpi  "PP4 interleaved v2 across nodes" 16 --preset tiny --layers 16 --tp 4 --pp 4 --pp-virtual 2 --microbatches 8
  t_hier "PP4 interleaved v2 timeline"  16 4 --preset tiny --layers 16 --tp 4 --pp 4 --pp-virtual 2 --microbatches 8 --timeline "$XP_OUT/timeline_pp4v2"
  t_mpi  "CP4 causal"                   4  --preset tiny --cp 4
  t_mpi  "CP2 x TP2 (GQA kv=2)"         4  --preset tiny --cp 2 --tp 2 --kv-heads 2
  t_mpi  "EP4 MoE uniform"              4  --preset moe-tiny --dp 4 --ep 4
  t_mpi  "EP2 x DP2 zipf cf1.25"        4  --preset moe-tiny --dp 4 --ep 2 --router zipf --capacity-factor 1.25
  t_mpi  "MoE TP2+EP2 (no SP)"          4  --preset moe-tiny --tp 2 --dp 2 --ep 2
  t_mpi  "MoE TP2+SP+EP2 FSDP"          4  --preset moe-tiny --tp 2 --sp --fsdp 2 --ep 2
  t_mpi  "3D: TP2 PP2 DP2"              8  --preset tiny --tp 2 --pp 2 --dp 2 --microbatches 4
  t_mpi  "4D: TP2 SP PP2 FSDP2 rc"      8  --preset tiny --tp 2 --sp --pp 2 --fsdp 2 --recompute --microbatches 3
  t_mpi  "5D: TP2 CP2 PP2 DP2 EP2 MoE" 16  --preset moe-tiny --tp 2 --cp 2 --pp 2 --dp 2 --ep 2 --microbatches 2
  t_mpi  "bf16 mixed, fp32 grads"       4  --preset tiny --dp 2 --tp 2 --precision bf16
  t_mpi  "fp16 + loss scale"            4  --preset tiny --fsdp 4 --precision fp16 --grad-dtype compute
  t_mpi  "fp64"                         4  --preset tiny --dp 2 --tp 2 --precision fp64
  t_mpi  "sync-collectives"             4  --preset tiny --tp 2 --dp 2 --sync-collectives
  # real inter-node traffic over Slingshot
  t_mpi  "DP8 across nodes"             8  --preset tiny --dp 8 --overlap
  t_mpi  "FSDP16 across nodes"         16  --preset tiny --fsdp 16 --overlap
  t_mpi  "EP16 MoE across nodes"       16  --preset moe-tiny --experts 16 --dp 16 --ep 16
  # hierarchical transport (GPU: VMM intra-node + CUDA-aware MPI inter-node)
  t_hier "TP2 x DP2"                    4 2 --preset tiny --tp 2 --dp 2
  t_hier "FSDP4 overlap"                4 2 --preset tiny --fsdp 4 --overlap
  t_hier "PP2 x TP2"                    4 2 --preset tiny --pp 2 --tp 2 --microbatches 4
  t_hier "EP4 local router"             4 2 --preset moe-tiny --dp 4 --ep 4 --router local
  t_hier "EP4 remote router"            4 2 --preset moe-tiny --dp 4 --ep 4 --router remote
  t_hier "EP8 node-limited M=1"         8 2 --preset moe-tiny --dp 8 --ep 8 --router node-limited --max-nodes 1
  t_hier "EP8 distance ring b=2"        8 2 --preset moe-tiny --dp 8 --ep 8 --router distance --distance-beta 2 --topology ring
  t_hier "CP2 TP2 DP2 bf16"             8 4 --preset tiny --cp 2 --tp 2 --dp 2 --precision bf16
  t_hier "TP4+SP x DP4 bf16"           16 4 --preset tiny --tp 4 --sp --dp 4 --precision bf16
  # trace record / replay (GPU: CUDA graphs per compute segment; CPU: eager re-issue)
  local RM; RM=$(xp_replay_mode)
  t_mpi  "replay-$RM TP4+SP"            4  --preset tiny --tp 4 --sp --replay "$RM"
  t_hier "replay-$RM EP8 MoE"           8 4 --preset moe-tiny --dp 8 --ep 8 --replay "$RM"
  t_roundtrip "FSDP16 overlap"         16  --preset tiny --fsdp 16 --overlap
  t_roundtrip "3D TP2 PP2 DP4 bf16"    16  --preset tiny --tp 2 --pp 2 --dp 4 --microbatches 4 --precision bf16
  t_roundtrip "MoE TP2+SP EP4 FSDP2"   16  --preset moe-tiny --tp 2 --sp --fsdp 2 --dp 4 --ep 4
  t_roundtrip "TP4 PP4 v2 interleaved" 16  --preset tiny --layers 16 --tp 4 --pp 4 --pp-virtual 2 --microbatches 8
}

# Save a trace (mpi transport, in-process replay), rebuild + replay it in new jobs on
# the mpi and on the hier transport; grad norms of record and reload must agree.
t_roundtrip() { local name=$1 W=$2; shift 2
  local RM; RM=$(xp_replay_mode)
  local tpn=$(( W < XP_TPN ? W : XP_TPN )); local nodes=$(( (W + tpn - 1) / tpn ))
  local pre="$XP_OUT/traces/$(echo "$name" | tr ' /+:()' '______')"; mkdir -p "$XP_OUT/traces"
  local rec="trace-save $name [mpi ${nodes}x${tpn}]" lm="trace-load $name [mpi ${nodes}x${tpn}]"
  local lh="trace-load $name [hier ${nodes}x${tpn}]"
  xp_case "$rec" "$nodes" "$tpn" mpi 1 -- $PEAK --verbose --replay "$RM" --trace-save "$pre" "$@"
  xp_case "$lm"  "$nodes" "$tpn" mpi 1 -- $PEAK --verbose --replay "$RM" --trace-load "$pre"
  xp_case "$lh"  "$nodes" 1 hier "$tpn" -- $PEAK --verbose --replay "$RM" --trace-load "$pre"
  xp_same_norms "round trip $name (record vs reload)" "$rec" "$lm"
}
