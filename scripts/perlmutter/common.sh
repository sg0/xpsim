# Shared helpers for the Perlmutter batch scripts (sourced, not executed).
#
# xp_launch <nodes> <tasks_per_node> <transport> <workers_per_node> -- <xpsim args>
#   Builds the srun line for the current partition (XP_PART = gpu|cpu):
#   GPU, transport mpi : <tpn> ranks/node (4 = one per A100), -c 128/tpn, --cpu-bind=cores,
#                        --gpus-per-node=4 --gpu-bind=none, CUDA_VISIBLE_DEVICES reversed per
#                        local rank (closest-GPU affinity on Perlmutter) when tpn == 4
#   GPU, transport hier: 1 rank/node driving <workers_per_node> GPUs (VMM intra-node,
#                        CUDA-aware MPI inter-node), -c 128
#   CPU, transport mpi : <tpn> ranks/node, -c 256/tpn, OpenMP 128/tpn threads per rank on cores
#   CPU, transport hier: 1 rank/node, <workers_per_node> worker threads x 128/wpn OpenMP threads
# XP_DRYRUN=1 prints the command instead of running it.

: "${XP_PART:=gpu}"
XP_PASS=0; XP_FAIL=0; XP_SKIP=0

xp_bin() { if [ "$XP_PART" = gpu ]; then echo "$XP_BIN_GPU"; else echo "$XP_BIN_CPU"; fi; }

xp_launch() {
  local nodes=$1 tpn=$2 tr=$3 wpn=$4; shift 4; [ "${1:-}" = "--" ] && shift
  local ntasks c envs=() extra=()
  if [ "$tr" = hier ]; then tpn=1; fi
  ntasks=$((nodes * tpn))
  if [ "$XP_PART" = gpu ]; then
    c=$((128 / tpn))
    envs+=(OMP_NUM_THREADS=1)
    [ "$tr" = mpi ] && [ "$tpn" -eq 4 ] && envs+=(XP_SELECT_GPU=1)
    extra+=(--gpus-per-node=4 --gpu-bind=none)
  else
    c=$((256 / tpn))
    if [ "$tr" = hier ]; then
      envs+=(OMP_NUM_THREADS=$((128 / wpn)) OMP_PROC_BIND=false)
    else
      envs+=(OMP_NUM_THREADS=$((128 / tpn)) OMP_PLACES=cores OMP_PROC_BIND=spread)
    fi
  fi
  local targs=(--transport "$tr")
  [ "$tr" = hier ] && targs+=(--workers-per-node "$wpn")
  local dev=cpu; [ "$XP_PART" = gpu ] && dev=gpu
  XP_CMD=(env "${envs[@]}" srun -N "$nodes" -n "$ntasks" --ntasks-per-node="$tpn" -c "$c" --cpu-bind=cores
          "${extra[@]}" "$XP_ROOT/scripts/perlmutter/bind.sh" "$(xp_bin)" --device "$dev" "${targs[@]}"
          --iters "${XP_ITERS:-3}" --warmup "${XP_WARMUP:-1}" "$@")
}

# xp_case <name> <nodes> <tasks_per_node> <transport> <workers_per_node> -- <args>
# Runs one case, prints PASS/FAIL with iteration time, tokens/s and MFU, logs to $XP_OUT/logs.
xp_case() {
  local name=$1 nodes=$2; shift 2
  if [ "$nodes" -gt "${SLURM_NNODES:-$nodes}" ]; then
    printf "SKIP  %-44s needs %d nodes\n" "$name" "$nodes"; XP_SKIP=$((XP_SKIP + 1)); return 0
  fi
  xp_launch "$nodes" "$@"
  mkdir -p "$XP_OUT/logs"
  local log; log=$(xp_log_of "$name")
  if [ "${XP_DRYRUN:-0}" = 1 ]; then
    printf "DRY   %-44s %s\n" "$name" "${XP_CMD[*]}"; XP_PASS=$((XP_PASS + 1)); return 0
  fi
  if timeout "${XP_CASE_TIMEOUT:-900}" "${XP_CMD[@]}" > "$log" 2>&1; then
    local it tps mfu
    it=$(grep "iteration time" "$log" | awk '{print $4}')
    tps=$(grep "^throughput" "$log" | awk '{print $3, $4}')
    mfu=$(grep "MFU / HFU" "$log" | awk '{print $5}')
    printf "PASS  %-44s iter %9s s  %-18s MFU %6s %%\n" "$name" "${it:--}" "${tps:--}" "${mfu:--}"
    XP_PASS=$((XP_PASS + 1))
  else
    printf "FAIL  %-44s (log: %s)\n" "$name" "$log"; tail -8 "$log" | sed 's/^/      /'
    XP_FAIL=$((XP_FAIL + 1))
  fi
}

xp_log_of() { echo "$XP_OUT/logs/$(echo "$1" | tr ' /:()+,|' '________').log"; }
xp_norms() { grep "grad-norm" "$1" 2>/dev/null | sed 's/.*grad-norm \([^ ]*\).*/\1/' | tr '\n' ' '; }
# xp_same_norms <name> <case A name> <case B name>: grad-norm sequences of two logged cases agree
xp_same_norms() {
  local a b; a=$(xp_norms "$(xp_log_of "$2")"); b=$(xp_norms "$(xp_log_of "$3")")
  if [ "${XP_DRYRUN:-0}" = 1 ]; then printf "DRY   %-44s compare grad norms\n" "$1"; return 0; fi
  if [ -n "$a" ] && [ "$a" = "$b" ]; then printf "PASS  %-44s grad norms identical (%s)\n" "$1" "$(echo $a | wc -w) iters"; XP_PASS=$((XP_PASS + 1))
  else printf "FAIL  %-44s\n      A: %s\n      B: %s\n" "$1" "$a" "$b"; XP_FAIL=$((XP_FAIL + 1)); fi
}
xp_replay_mode() { if [ "$XP_PART" = gpu ]; then echo graph; else echo eager; fi; }

xp_summary() { echo "----"; echo "passed $XP_PASS, failed $XP_FAIL, skipped $XP_SKIP"; [ "$XP_FAIL" -eq 0 ]; }

# Nominal per-worker peak (TFLOP/s) for MFU on A100 (dense tensor-core rates);
# CPU runs use the measured attainable GEMM rate unless XP_PEAK is set.
xp_peak_args() {
  local prec=$1
  if [ -n "${XP_PEAK:-}" ]; then echo "--peak-tflops $XP_PEAK"; return; fi
  [ "$XP_PART" = gpu ] || return 0
  case "$prec" in
    bf16|fp16) echo "--peak-tflops 312" ;;
    fp64)      echo "--peak-tflops 19.5" ;;
    fp32)      if [ "${XP_TF32:-0}" = 1 ]; then echo "--peak-tflops 156 --tf32"; else echo "--peak-tflops 19.5"; fi ;;
  esac
}
gcd() { local a=$1 b=$2 t; while [ "$b" -ne 0 ]; do t=$((a % b)); a=$b; b=$t; done; echo "$a"; }
