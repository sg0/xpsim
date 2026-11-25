# Pipeline-bubble study (sourced by pp_bubble_gpu.sbatch / pp_bubble_cpu.sbatch).
# Fixed machine: XP_PP_NODES nodes = pipeline stages (one stage per node, TP inside
# the node, so stage-to-stage traffic crosses the network). Sweeps
#   v   in XP_PP_V      virtual stages per rank (1 = plain pipeline, >1 = interleaved)
#   m   in XP_PP_M      microbatches, as multiples of p (m = k*p)
#   sched in XP_PP_SCHED   1f1b | gpipe
#   replay in XP_REPLAY    off | graph (GPU: removes kernel-launch overhead)
#   transport in XP_TRANSPORTS
# Every case writes one CSV row (label "ppstudy|v|m|sched|transport+replay") and a
# timeline; pp_bubble_summary.py prints measured vs theoretical bubble and
# timeline_gantt.py draws any pair of timelines.

xp_pp_bubble() {
  local G=$1
  local p=${XP_PP_NODES:-${SLURM_NNODES:-2}}
  local csv="$XP_OUT/pp_bubble_${XP_PART}.csv" tdir="$XP_OUT/timelines"
  mkdir -p "$tdir" "$XP_OUT/json"
  local peak; peak=$(xp_peak_args "$XP_PREC")
  echo "pipeline bubble study: p=$p stages (one per node) x tp=$G | v: ${XP_PP_V} | m/p: ${XP_PP_M} | sched: ${XP_PP_SCHED} | replay: ${XP_REPLAY:-off}"
  for v in $XP_PP_V; do
    [ "$XP_PP_LAYERS" -ge $((p * v)) ] || { echo "SKIP  v=$v (needs >= $((p * v)) layers)"; continue; }
    for k in $XP_PP_M; do
      local m=$((k * p))
      for sch in $XP_PP_SCHED; do
        for tr in ${XP_TRANSPORTS:-mpi}; do
          for rp in ${XP_REPLAY:-off}; do
            local tl=$tr; [ "$rp" != off ] && tl="$tr+$rp"
            local label="ppstudy|$v|$m|$sch|$tl"
            local tag; tag=$(echo "$label" | tr '|+' '__')
            local args="$XP_PP_MODEL --layers $XP_PP_LAYERS --precision $XP_PREC $peak --tp $G --sp --pp $p --pp-virtual $v
                        --microbatches $m --pp-schedule $sch --replay $rp --csv $csv --label $label
                        --json $XP_OUT/json/$tag.json --timeline $tdir/$tag"
            if [ "$tr" = mpi ]; then xp_case "$label" "$p" "$G" mpi 1 -- $args
            else xp_case "$label" "$p" 1 hier "$G" -- $args; fi
          done
        done
      done
    done
  done
  if [ "${XP_DRYRUN:-0}" != 1 ] && [ -f "$csv" ]; then
    python3 "$XP_ROOT/scripts/perlmutter/pp_bubble_summary.py" "$csv" "$p" | tee "$XP_OUT/pp_bubble_${XP_PART}_summary.txt"
    echo "Gantt chart, e.g.: python3 scripts/timeline_gantt.py $tdir/ppstudy_1_$((2 * p))_1f1b_mpi.json $tdir/ppstudy_2_$((2 * p))_1f1b_mpi.json --labels v=1 v=2 --out pp_gantt.png"
  fi
}
