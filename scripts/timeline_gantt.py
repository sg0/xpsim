#!/usr/bin/env python3
"""Pipeline Gantt chart + bubble table from xpsim timelines (--timeline P -> P.json).

One panel per timeline file (same time scale, so schedules can be compared side by
side), one row per worker sorted by pipeline stage. Bars: forward (F) and backward
(B) of each microbatch chunk, and exposed pipeline send/recv waits; blank space is
idle time (the bubble). Labels name the microbatch (and chunk) on bars wide enough.

usage: timeline_gantt.py run_v1.json run_v2.json [--labels "v=1" "v=2"]
                         [--iteration 0] [--out pipeline.png] [--csv stages.csv]
"""
import argparse
import csv
import json
import sys

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode):
# contrast < 3:1 for slots 2/3 -> bars carry text labels and a table is printed.
C_F, C_B, C_W = "#257de0", "#f47230", "#00b894"
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"


def load(path, iteration):
    ev = json.load(open(path))["traceEvents"]
    names, sort = {}, {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[e["tid"]] = e["args"]["name"]
        if e.get("ph") == "M" and e.get("name") == "thread_sort_index":
            sort[e["tid"]] = e["args"]["sort_index"]
    xs = [e for e in ev if e.get("ph") == "X"]
    its = [e for e in xs if e["name"] == f"iteration {iteration}"]
    if not its:
        sys.exit(f"{path}: iteration {iteration} not in the timeline (raise --timeline-iters)")
    t0 = min(e["ts"] for e in its)
    t1 = max(e["ts"] + e["dur"] for e in its)
    rows = {}
    for e in xs:
        if e["ts"] < t0 or e["ts"] > t1:
            continue
        r = rows.setdefault(e["tid"], {"F": [], "B": [], "W": []})
        if e.get("cat") == "pipeline" and e["name"][:2] in ("F ", "B "):
            label = e["name"][2:].replace("mb", "").replace(" c", ".")
            r[e["name"][0]].append((e["ts"] - t0, e["dur"], label))
        elif e.get("cat") == "comm" and e["name"].startswith("pp_p2p") and e.get("args", {}).get("phase") == "wait":
            r["W"].append((e["ts"] - t0, e["dur"], ""))
    order = sorted(rows, key=lambda t: sort.get(t, t))
    return {"rows": [(t, names.get(t, f"w{t}"), rows[t]) for t in order], "span": t1 - t0}


def stats(run):
    out = []
    for tid, name, r in run["rows"]:
        f = sum(d for _, d, _ in r["F"]); b = sum(d for _, d, _ in r["B"]); w = sum(d for _, d, _ in r["W"])
        out.append({"worker": name, "F_ms": f / 1e3, "B_ms": b / 1e3, "p2p_wait_ms": w / 1e3,
                    "busy_pct": 100 * (f + b - w) / run["span"], "idle_pct": 100 * (1 - (f + b - w) / run["span"])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--iteration", type=int, default=0, help="recorded iteration to draw (0 = first)")
    ap.add_argument("--out", default="pipeline_gantt.png")
    ap.add_argument("--csv", help="write the per-worker table here")
    a = ap.parse_args()
    labels = a.labels or a.files
    runs = [load(f, a.iteration) for f in a.files]

    table = []
    for lab, run in zip(labels, runs):
        print(f"\n{lab}: iteration {run['span'] / 1e3:.2f} ms")
        print(f"  {'worker':<28} {'F ms':>9} {'B ms':>9} {'p2p wait':>9} {'busy %':>7} {'idle %':>7}")
        for s in stats(run):
            print(f"  {s['worker']:<28} {s['F_ms']:>9.2f} {s['B_ms']:>9.2f} {s['p2p_wait_ms']:>9.2f} {s['busy_pct']:>7.1f} {s['idle_pct']:>7.1f}")
            table.append({"run": lab, **s})
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    xmax = max(r["span"] for r in runs) / 1e3
    nrows = [len(r["rows"]) for r in runs]
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 0.5 + sum(0.42 * n + 0.9 for n in nrows)), squeeze=False,
                             gridspec_kw={"height_ratios": [n + 1.5 for n in nrows]}, facecolor=SURFACE)
    for ax, lab, run in zip(axes[:, 0], labels, runs):
        ax.set_facecolor(SURFACE)
        names = []
        for i, (tid, name, r) in enumerate(run["rows"]):
            y = len(run["rows"]) - 1 - i
            names.append((y, name))
            # waits happen inside F/B spans (a stage waits for its input): drawn on top,
            # full height, so the F/B color shows compute only
            for key, col in (("F", C_F), ("B", C_B), ("W", C_W)):
                segs = [(s / 1e3, d / 1e3) for s, d, _ in r[key]]
                if segs:
                    ax.broken_barh(segs, (y - 0.35, 0.7), facecolors=col, edgecolor=SURFACE, linewidth=1.0)
            for key in ("F", "B"):  # direct labels on the compute part, where wide enough
                for s, d, txt in r[key]:
                    w = sum(wd for ws, wd, _ in r["W"] if ws >= s and ws + wd <= s + d)
                    cs, cd = s + w, d - w
                    if cd / 1e3 > xmax * 0.022:
                        ax.text((cs + cd / 2) / 1e3, y, txt, ha="center", va="center", fontsize=6.5, color=TEXT)
        ax.set_yticks([y for y, _ in names], [n for _, n in names], fontsize=8, color=TEXT)
        ax.set_ylim(-0.7, len(run["rows"]) - 0.3)
        ax.set_xlim(0, xmax * 1.01)
        busy = sum(s["busy_pct"] for s in stats(run)) / len(run["rows"])
        ax.set_title(f"{lab}  ·  iteration {run['span'] / 1e3:.1f} ms  ·  mean idle {100 - busy:.1f} %",
                     loc="left", fontsize=10, color=TEXT)
        ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=TEXT2, labelsize=8); ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel("time within the iteration (ms)", fontsize=9, color=TEXT2)
    handles = [Patch(color=C_F, label="forward (microbatch[.chunk])"), Patch(color=C_B, label="backward"),
               Patch(color=C_W, label="pipeline send/recv wait"), Patch(facecolor=SURFACE, edgecolor=GRID, label="idle (bubble)")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print(f"\nwritten {a.out}" + (f" and {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main()
