#!/usr/bin/env bash
# Trace round-trip check: run <args> live for one warm-up step, record it, replay the
# measured steps in-process (--replay eager|graph) and save the trace; then rebuild and
# replay the saved trace in a fresh process (--trace-load). Both runs must produce the
# same grad-norm sequence, counted FLOPs and communication volume.
#
# usage: scripts/replay_check.sh <name> <np> <xpsim args...>
# env:   BIN     binary (default bin/xpsim-shim)
#        LAUNCH  launcher prefix taking the rank count, e.g. "mpirun -np" or "srun -n"
#                (empty: thread-based MPI shim, ranks via -np)
#        MODE    in-process replay mode: eager (default) or graph (GPU)
#        LOAD_NP / LOADX   rank count / extra options for the load run (e.g. another transport)
#        TRACE_DIR         where traces go (default ${TMPDIR:-/tmp}/xpsim-traces)
set -u
BIN=${BIN:-bin/xpsim-shim}
LAUNCH=${LAUNCH:-}
MODE=${MODE:-eager}
TRACE_DIR=${TRACE_DIR:-${TMPDIR:-/tmp}/xpsim-traces}
mkdir -p "$TRACE_DIR"
name=$1; np=$2; shift 2
lnp=${LOAD_NP:-$np}
pre="$TRACE_DIR/$(echo "$name" | tr ' /+:()' '______')"
C="--iters 3 --warmup 1 --peak-tflops 0.05 --verbose"
run() { local n=$1; shift; if [ -z "$LAUNCH" ]; then timeout 600 "$BIN" -np "$n" "$@"; else timeout 600 $LAUNCH "$n" "$BIN" "$@"; fi; }
a=$(run "$np" $C "$@" --replay "$MODE" --trace-save "$pre" 2>&1) || { echo "FAIL  $name (record)"; echo "$a" | tail -6; exit 1; }
b=$(run "$lnp" $C --trace-load "$pre" --replay "$MODE" ${LOADX:-} 2>&1) || { echo "FAIL  $name (load)"; echo "$b" | tail -6; exit 1; }
norms() { echo "$1" | grep "grad-norm" | sed 's/.*grad-norm \([^ ]*\).*/\1/' | tr '\n' ' '; }
flops() { echo "$1" | grep "counted" | awk '{print $7 $8}'; }
vol()   { echo "$1" | grep "comm volume" | awk '{print $7}'; }
seg=$(echo "$b" | grep "recorded ops" | sed 's/^ *//' | cut -d, -f1-3)
if [ "$(norms "$a")" = "$(norms "$b")" ] && [ "$(flops "$a")" = "$(flops "$b")" ] && [ "$(vol "$a")" = "$(vol "$b")" ]; then
  printf "MATCH %-34s %s | %s MiB | %s\n" "$name" "$(flops "$b")" "$(vol "$b")" "$seg"; exit 0
fi
printf "DIFF  %-34s\n   in-process: %s | %s | %s\n   from file : %s | %s | %s\n" "$name" "$(norms "$a")" "$(flops "$a")" \
  "$(vol "$a")" "$(norms "$b")" "$(flops "$b")" "$(vol "$b")"
exit 1
