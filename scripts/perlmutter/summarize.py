#!/usr/bin/env python3
"""Scaling-efficiency tables from an xpsim CSV (scaling_gpu.csv / scaling_cpu.csv).

Labels are "strategy|transport|nodes|workers|weak|strong" (written by scaling_cases.sh).
For each (strategy, transport) the smallest worker count is the baseline:
  weak   efficiency = (tokens/s per worker) / (baseline tokens/s per worker)
  strong efficiency = (tokens/s / baseline tokens/s) / (workers / baseline workers)
                     (= speedup / worker ratio when the global batch is fixed)
Also writes <csv>_summary.csv next to the input.
usage: summarize.py scaling_gpu.csv
"""
import csv
import sys
from collections import defaultdict


def main(path):
    rows = list(csv.DictReader(open(path)))
    groups = defaultdict(list)
    for r in rows:
        parts = r["label"].split("|")
        if len(parts) != 5:
            continue
        strat, tr, nodes, workers, mode = parts
        groups[(strat, tr, mode)].append({
            "nodes": int(nodes), "workers": int(workers), "iter": float(r["iter_s"]),
            "tps": float(r["tokens_per_s"]), "mfu": float(r["mfu"]), "comm": float(r["comm_frac"]),
            "mem": float(r["peak_mem_mib"]),
        })
    out_rows = []
    for (strat, tr, mode), g in sorted(groups.items()):
        g.sort(key=lambda x: x["workers"])
        b = g[0]
        print(f"\n{strat}  [{tr}, {mode} scaling]")
        print(f"{'nodes':>5} {'workers':>7} {'iter s':>9} {'tokens/s':>12} {'tok/s/wkr':>10} {'MFU %':>6} "
              f"{'comm %':>6} {'mem MiB':>8} {'eff %':>6}")
        for x in g:
            if mode == "weak":
                eff = (x["tps"] / x["workers"]) / (b["tps"] / b["workers"])
            else:
                eff = (x["tps"] / b["tps"]) / (x["workers"] / b["workers"])
            print(f"{x['nodes']:>5} {x['workers']:>7} {x['iter']:>9.4f} {x['tps']:>12.4g} {x['tps']/x['workers']:>10.4g} "
                  f"{100*x['mfu']:>6.2f} {100*x['comm']:>6.1f} {x['mem']:>8.0f} {100*eff:>6.1f}")
            out_rows.append([strat, tr, mode, x["nodes"], x["workers"], x["iter"], x["tps"], x["mfu"], x["comm"],
                             x["mem"], eff])
    # execution-variant comparison at equal scale (mpi, hier, mpi+graph, hier+graph, ...)
    pairs = defaultdict(dict)
    for (strat, tr, mode), g in groups.items():
        for x in g:
            pairs[(strat, mode, x["workers"])][tr] = x["iter"]
    cmp = [(k, v) for k, v in sorted(pairs.items()) if len(v) > 1]
    if cmp:
        print("\nvariant comparison (iteration time; speedup relative to the first variant listed)")
        for (strat, mode, w), v in cmp:
            names = sorted(v, key=lambda t: (t != "mpi", t))
            base = v[names[0]]
            cols = "  ".join(f"{t} {v[t]:.4f}s (x{base / v[t]:.2f})" for t in names)
            print(f"  {strat:<24} W={w:<5} {cols}")
    out = path.rsplit(".", 1)[0] + "_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "transport", "mode", "nodes", "workers", "iter_s", "tokens_per_s", "mfu", "comm_frac",
                    "peak_mem_mib", "efficiency"])
        w.writerows(out_rows)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scaling_gpu.csv")
