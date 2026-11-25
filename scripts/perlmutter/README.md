# xpsim on NERSC Perlmutter

Batch kit for functional tests and CPU/GPU scalability experiments on Perlmutter
(HPE Cray EX, Slingshot-11; GPU nodes: 4x A100 + 1x EPYC 7763; CPU nodes: 2x EPYC 7763,
128 cores, 8 NUMA domains). Run everything from the repository root.

| file | purpose |
|---|---|
| `env.sh gpu\|cpu` | modules (PrgEnv-gnu, cray-libsci, cudatoolkit, craype-accel-nvidia80), `MPICH_GPU_SUPPORT_ENABLED=1`, `MPICH_OFI_NIC_POLICY=GPU`, output dir (`$SCRATCH/xpsim-results`) |
| `build.sh [gpu\|cpu]` | builds `bin/xpsim-gpu` (CUDA sm_80, CUDA-aware Cray MPICH, VMM) and `bin/xpsim-cpu` (LibSci CBLAS + OpenMP) with the `CC` wrapper |
| `tests_gpu.sbatch`, `tests_cpu.sbatch` | functional matrix (port of `scripts/run_tests.sh`): 60 cases, every parallelism and mix, both transports, real inter-node DP/FSDP/EP cases, interleaved pipelines (incl. a merged multi-node timeline), in-process replay (CUDA graphs on GPU) and trace save -> reload round trips (incl. interleaved) on both transports with grad-norm comparison; 4 nodes, debug QOS |
| `scaling_gpu.sbatch`, `scaling_cpu.sbatch` | weak and strong scaling over 1, 2, 4, ... nodes inside one allocation, both transports, MoE router-locality sweep at the largest scale; pipeline strategies `pp`, `pp_v2`, `pp_v4` (one stage per node, interleaved v=2/4) |
| `pp_bubble_gpu.sbatch`, `pp_bubble_cpu.sbatch`, `pp_bubble_cases.sh`, `pp_bubble_summary.py` | pipeline-bubble study at a fixed node count (one stage per node): sweeps interleaving `v`, microbatches `m`, schedule (and optionally CUDA-graph replay); one timeline per case; table of measured vs theoretical bubble, MFU, memory and speedup over v=1 |
| `common.sh`, `tests_cases.sh`, `scaling_cases.sh` | launch logic and case tables (shared by CPU and GPU) |
| `bind.sh` | per-rank GPU affinity (local rank i -> GPU 3-i when 4 ranks/node, `--gpu-bind=none`) |
| `summarize.py` | efficiency tables + transport comparison from the CSV; also writes `*_summary.csv` |

## Quick start

```bash
scripts/perlmutter/build.sh                                   # login node
sbatch -A m1234_g scripts/perlmutter/tests_gpu.sbatch          # GPU account ends in _g
sbatch -A m1234   scripts/perlmutter/tests_cpu.sbatch
sbatch -A m1234_g -N 16 scripts/perlmutter/scaling_gpu.sbatch  # 1..16 nodes = 4..64 GPUs
sbatch -A m1234   -N 16 scripts/perlmutter/scaling_cpu.sbatch  # 1..16 nodes = 8..128 ranks
```

Results: `$SCRATCH/xpsim-results/<kind>-<jobid>/` holds `logs/` (one report per case),
`json/`, `scaling_<gpu|cpu>.csv`, `*_summary.csv` and `*_summary.txt`. Re-run the
summary at any time with `python3 scripts/perlmutter/summarize.py <csv>`.

## Launch layouts

| partition / transport | ranks per node | srun |
|---|---|---|
| GPU, `mpi` | 4 (one per A100) | `-c 32 --cpu-bind=cores --gpus-per-node=4 --gpu-bind=none`, `CUDA_VISIBLE_DEVICES=3-SLURM_LOCALID` |
| GPU, `hier` | 1, driving 4 GPUs | `-c 128 --cpu-bind=cores --gpus-per-node=4 --gpu-bind=none`; VMM zero copy inside the node, CUDA-aware MPI between nodes |
| CPU, `mpi` | `XP_TPN` (default 8 = one per NUMA domain) | `-c 256/XP_TPN --cpu-bind=cores`, `OMP_NUM_THREADS=128/XP_TPN`, `OMP_PLACES=cores`, `OMP_PROC_BIND=spread` |
| CPU, `hier` | 1, `XP_TPN` worker threads | `-c 256`, `OMP_NUM_THREADS=128/XP_TPN` per worker |

## Scaling knobs (export before `sbatch`)

| variable | default (GPU / CPU) | meaning |
|---|---|---|
| `XP_NODES` | powers of 2 up to `-N` | node counts to run |
| `XP_STRATEGIES` | `dp fsdp zero1 hsdp tp_dp tp_pp_dp cp_dp fsdp_strong moe_ep moe_router` | see `scaling_cases.sh` |
| `XP_TRANSPORTS` | `mpi hier` | compare rank-per-GPU MPI against VMM+MPI |
| `XP_MODEL` / `XP_MICRO` | gpt3-1.3b, 8 layers, seq 2048, micro-batch 1 / small, micro-batch 2 | dense model (`XP_MICRO` = its micro-batch) |
| `XP_MOE_MODEL` / `XP_EXPERTS` | moe-64e, 4 layers / small + 16 experts | MoE model; EP = gcd(workers, experts) |
| `XP_PREC` | bf16 / fp32 | precision |
| `XP_MB`, `XP_PP_MB` | 4, 8 | microbatches per step (weak scaling; pipeline cases) |
| `XP_GB` | 256 / 512 | global batch (sequences) for `fsdp_strong` |
| `XP_ITERS`, `XP_WARMUP` | 8, 2 / 5, 1 | measured / warm-up iterations |
| `XP_PEAK`, `XP_TF32` | A100 dense peak (312 bf16/fp16, 19.5 fp64/fp32, 156 TF32) / measured | MFU denominator per worker |
| `XP_REPLAY` | `off` | e.g. `"off graph"`: also run every case with in-process CUDA-graph replay (labels `mpi+graph`, `hier+graph`); the summary compares all variants |
| `XP_PP_LAYERS`, `XP_PP_MB` | 32 / 16, 8 | depth and minimum microbatches of the `pp*` strategies |
| `XP_DRYRUN=1` | | print the srun lines only |

Efficiency definitions: weak = tokens/s per worker relative to the smallest run;
strong = speedup / worker ratio. For 80 GB A100 nodes add `-C "gpu&hbm80g"`.

## Pipeline-bubble study

```bash
sbatch -A m1234_g -N 4 scripts/perlmutter/pp_bubble_gpu.sbatch                       # p = 4 stages
XP_REPLAY="off graph" sbatch -A m1234_g -N 8 scripts/perlmutter/pp_bubble_gpu.sbatch  # + CUDA-graph replay
```
Knobs:
- `XP_PP_V` (default `1 2 4`): interleaving degrees;
- `XP_PP_M` (`1 2 4 8`): microbatches as multiples of p;
- `XP_PP_SCHED` (`1f1b gpipe`);
- `XP_REPLAY`, `XP_TRANSPORTS`, `XP_PP_MODEL`, `XP_PP_LAYERS`.

Output: `pp_bubble_<gpu|cpu>.csv`, the summary table (measured pp wait vs (p−1)/(v·m+p−1),
MFU, memory, speedup over v=1) and `timelines/ppstudy_<v>_<m>_<sched>_<transport>.json`
for `scripts/timeline_gantt.py`.

## Notes and caveats

- **Tested:** the scripts were exercised end to end with a fake `srun` mapped onto the
  thread-based MPI shim: all 60 functional cases, the 44-case scaling run, the CSV and the
  summary. They have not been run under real Slurm on Perlmutter.
- **GPU build:** the CUDA backend and VMM transport have not been compiled yet. Submit
  `tests_gpu.sbatch` first; its two selftest cases check every collective on both transports.
- **hier transport on GPU:** needs a CUDA-aware Cray MPICH that accepts CUDA VMM
  (`cuMemCreate`/`cuMemMap`) device pointers for inter-node messages. If the hier
  selftest fails, rerun with `XP_TRANSPORTS=mpi`, or try `--gdr 0`.
- **NICs with hier:** with one rank per node, the inter-node traffic comes from a single
  process. Depending on the Cray MPICH version, that process may use only one of the four
  Slingshot NICs; check `man intro_mpi` for multi-NIC options.
- **Memory:** the GPU defaults fit 40 GB A100s. If you raise `--seq` or `--layers`, use
  FSDP or the 80 GB nodes, because attention stores the full score matrix.
