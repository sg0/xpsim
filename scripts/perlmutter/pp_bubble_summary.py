#!/usr/bin/env python3
"""Measured vs theoretical pipeline bubble from a pp_bubble CSV.

Labels are "ppstudy|v|m|sched|transport[+replay]" (written by pp_bubble_cases.sh).
measured bubble = exposed pipeline send/recv time (pp_p2p, incl. waiting for the
previous/next stage) / iteration time, averaged over workers; theoretical =
(p-1)/(v*m + p-1) for 1F1B/interleaved (equal F/B cost assumed).
usage: pp_bubble_summary.py pp_bubble_gpu.csv <p>
"""
import csv
import sys
from collections import defaultdict

path, p = sys.argv[1], int(sys.argv[2])
rows = [r for r in csv.DictReader(open(path)) if r["label"].startswith("ppstudy|")]
groups = defaultdict(list)
for r in rows:
    _, v, m, sch, tr = r["label"].split("|")
    groups[(sch, tr)].append((int(v), int(m), float(r["iter_s"]), float(r["tokens_per_s"]), float(r["pp_p2p_ms"]) / 1e3,
                              float(r["mfu"]), float(r["peak_mem_mib"])))
for (sch, tr), g in sorted(groups.items()):
    print(f"\n{sch}  [{tr}]  p={p}")
    print(f"{'v':>3} {'m':>5} {'iter s':>9} {'tokens/s':>11} {'bubble theo %':>14} {'pp wait %':>10} {'MFU %':>7} {'mem MiB':>8} {'speedup':>8}")
    base = {}
    for v, m, it, tps, pw, mfu, mem in sorted(g):
        base.setdefault(m, it)
        theo = (p - 1) / (v * m + p - 1)
        print(f"{v:>3} {m:>5} {it:>9.4f} {tps:>11.4g} {100 * theo:>14.1f} {100 * pw / it:>10.1f} {100 * mfu:>7.2f} {mem:>8.0f} "
              f"{base[m] / it:>8.2f}")
print("\nspeedup: relative to v=1 at the same m. pp wait includes the transfer time of the (v-times more) messages.")
