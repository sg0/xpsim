# Scaling experiments (sourced by scaling_gpu.sbatch / scaling_cpu.sbatch).
# For every node count n in XP_NODES, every strategy in XP_STRATEGIES and every
# transport in XP_TRANSPORTS one case runs inside the same allocation (srun -N n)
# and appends a row to $XP_OUT/scaling_$XP_PART.csv. Labels are
# "strategy|transport|nodes|workers|weak|strong", parsed by summarize.py.
#
#   G  workers per node: 4 on GPU nodes (one per A100), XP_TPN on CPU nodes
#   W  = n * G workers
# Weak scaling keeps the per-worker work fixed (micro-batch x XP_MB microbatches);
# strong scaling keeps the global batch XP_GB fixed.
#
# Strategies:
#   dp        --dp W                      (DDP / ZeRO-0, overlapped all-reduce)
#   fsdp      --fsdp W                    (ZeRO-3, prefetch + overlapped reduce-scatter)
#   zero1     --fsdp W --zero 1
#   hsdp      --fsdp G --dp n             (shard inside a node, replicate across nodes)
#   tp_dp     --tp G --sp --dp n          (Megatron TP+SP inside a node)
#   tp_pp_dp  --tp G --sp --pp 2 --dp n/2 (3-D, even n only)
#   cp_dp     --cp G --dp n               (context parallel inside a node, long sequence)
#   fsdp_strong  fsdp at a fixed global batch (strong scaling)
#   moe_ep    MoE model, --dp W --ep gcd(W, experts)
#   moe_router   MoE at the largest n with each router locality policy

xp_default_nodes() {
  local n=1 out=""; local max=${SLURM_NNODES:-1}
  while [ "$n" -le "$max" ]; do out="$out $n"; n=$((n * 2)); done
  echo $out
}

xp_scaling() {
  local G=$1
  local nodes_list=${XP_NODES:-$(xp_default_nodes)}
  local strategies=${XP_STRATEGIES:-"dp fsdp zero1 hsdp tp_dp tp_pp_dp cp_dp fsdp_strong moe_ep moe_router"}
  local transports=${XP_TRANSPORTS:-"mpi hier"}
  local csv="$XP_OUT/scaling_${XP_PART}.csv"
  local peak; peak=$(xp_peak_args "$XP_PREC")
  local base="$XP_MODEL --precision $XP_PREC $peak --csv $csv"
  local nmax=0; for n in $nodes_list; do [ "$n" -gt "$nmax" ] && nmax=$n; done
  echo "nodes: $nodes_list | strategies: $strategies | transports: $transports | workers/node: $G"
  echo "model: $XP_MODEL --precision $XP_PREC | MoE: $XP_MOE_MODEL | csv: $csv"
  for n in $nodes_list; do
    local W=$((n * G))
    for s in $strategies; do
      local args="" mode=weak routers="_"
      case $s in
        dp)          args="--dp $W --overlap --microbatches $XP_MB" ;;
        fsdp)        args="--fsdp $W --overlap --microbatches $XP_MB" ;;
        zero1)       args="--fsdp $W --zero 1 --overlap --microbatches $XP_MB" ;;
        hsdp)        args="--fsdp $G --dp $n --overlap --microbatches $XP_MB" ;;
        tp_dp)       args="--tp $G --sp --dp $n --overlap --microbatches $XP_MB" ;;
        tp_pp_dp)    [ $((n % 2)) -eq 0 ] || continue
                     args="--tp $G --sp --pp 2 --dp $((n / 2)) --overlap --microbatches $XP_PP_MB" ;;
        cp_dp)       args="--cp $G --dp $n --overlap --microbatches $XP_MB $XP_CP_EXTRA" ;;
        fsdp_strong) local per=$((XP_MICRO * W)); mode=strong
                     [ $((XP_GB % per)) -eq 0 ] && [ "$XP_GB" -ge "$per" ] || { echo "SKIP  fsdp_strong N=$n (global batch $XP_GB not divisible by $per)"; continue; }
                     args="--fsdp $W --overlap --microbatches $((XP_GB / per))" ;;
        moe_ep)      args="$XP_MOE_MODEL --dp $W --ep $(gcd "$W" "$XP_EXPERTS") --overlap --microbatches $XP_MB" ;;
        moe_router)  [ "$n" -eq "$nmax" ] || continue
                     routers="uniform local remote node-limited zipf"
                     args="$XP_MOE_MODEL --dp $W --ep $(gcd "$W" "$XP_EXPERTS") --overlap --microbatches $XP_MB" ;;
        *) echo "unknown strategy $s"; continue ;;
      esac
      for r in $routers; do
        local rargs="" sname=$s
        if [ "$r" != "_" ]; then
          sname="${s}_$r"; rargs="--router $r"
          [ "$r" = node-limited ] && rargs="$rargs --max-nodes 1"
          [ "$r" = local ] || [ "$r" = remote ] && rargs="$rargs --locality 0.9"
        fi
        for tr in $transports; do
          local label="$sname|$tr|$n|$W|$mode"
          local json="$XP_OUT/json/$(echo "$label" | tr '|' '_').json"
          mkdir -p "$XP_OUT/json"
          if [ "$tr" = mpi ]; then
            xp_case "$label" "$n" "$G" mpi 1 -- $base $args $rargs --label "$label" --json "$json"
          else
            xp_case "$label" "$n" 1 hier "$G" -- $base $args $rargs --label "$label" --json "$json"
          fi
        done
      done
    done
  done
  if [ "${XP_DRYRUN:-0}" != 1 ] && [ -f "$csv" ]; then
    python3 "$XP_ROOT/scripts/perlmutter/summarize.py" "$csv" | tee "$XP_OUT/scaling_${XP_PART}_summary.txt"
  fi
}
