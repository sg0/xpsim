# xpsim: distributed-memory ML training simulator (MPI + C++, CPU or CUDA)

xpsim runs the *real* computation and communication of a transformer training
step (forward, backward, optimizer) on randomly generated parameters,
activations, gradients and optimizer states. The work is distributed over any
mix of **DP, FSDP/ZeRO-1/2/3, HSDP, TP (+sequence parallel), PP (GPipe/1F1B),
CP (context parallel) and EP (MoE expert parallel)**. It reports figures of merit
and a full time, communication and memory breakdown.

The numbers are random. The *shapes, FLOPs, bytes, communication patterns,
dependencies and memory footprints* are those of a real run, so iteration time,
MFU, exposed communication, bubbles and memory are what you would measure on
that machine.

```
make CUDA=1 SM=90 BLAS=mkl                 # GPU build (CPU fallback BLAS = MKL)
mpirun -np 16 ./bin/xpsim --device gpu --precision bf16 --preset llama3-8b --layers 8 \
       --tp 4 --sp --pp 2 --fsdp 2 --overlap --microbatches 8
# one rank per node, VMM intra-node + CUDA-aware MPI inter-node:
srun -N 2 --ntasks-per-node=1 ./bin/xpsim --transport hier --device gpu --precision bf16 ...
```

---------------------------------------------------------------------------------
## 1. Building (GNU make)

| command | result |
|---|---|
| `make` | CPU build with `mpicxx` |
| `make CUDA=1 SM=80` (or 90, 100) | + CUDA backend: cuBLAS tensor cores, VMM transport. Needs sm_80+ |
| `make BLAS=openblas\|mkl\|blis\|system` or `BLAS_LIBS="-L... -lfoo"` | CPU GEMMs through vendor CBLAS (`cblas_sgemm`/`cblas_dgemm`) |
| `make SHIM=1` | no MPI installed: links a thread-based MPI shim (`tools/mpishim`); ranks = threads, `-np N` |
| `make test` | builds the shim variant and runs `scripts/run_tests.sh` (35 configs) |
| `make DEBUG=1`, `OPENMP=0` | debug / no OpenMP |

Variables: `MPICXX`, `CXX`, `CUDA_HOME` (default `/usr/local/cuda`), `NVCC`, `SM`, `BLAS`, `BLAS_LIBS`.

Requirements: C++17 compiler, MPI (MPI-3 non-blocking collectives; CUDA-aware MPI
for GPU runs), optionally CUDA 11.8+ with cuBLAS and the driver API (`-lcuda`) for VMM.

---------------------------------------------------------------------------------
## 2. Running

`./bin/xpsim --help` lists every option. The important ones:

**Model.** `--preset tiny|small|gpt3-1.3b|gpt3-13b|llama2-7b|llama3-8b|llama3-70b|mixtral-8x7b|moe-64e|moe-tiny`;
override with `--layers --hidden --heads --kv-heads (GQA) --ffn --seq --vocab --micro-batch --microbatches`.

**Parallelism.** World size = `pp * dp * fsdp * cp * tp`:

| flag | meaning |
|---|---|
| `--tp N [--sp]` | Megatron tensor parallel (column/row-parallel QKV, out-proj, MLP; vocab-parallel embedding/head); `--sp` = sequence parallel (all-gather / reduce-scatter instead of all-reduce) |
| `--pp N --pp-schedule 1f1b\|gpipe --microbatches m` | pipeline parallel; layers split over stages |
| `--dp N` | data-parallel *replicate* dimension |
| `--fsdp N --zero 0..3` | data-parallel *shard* dimension; ZeRO stage (default 3 if fsdp>1). `--dp 2 --fsdp 4` = HSDP |
| `--cp N` | context parallel: sequence split, K/V all-gathered, dK/dV reduce-scattered; causal mask with global positions |
| `--ep N --experts E --topk k [--moe-every n] [--capacity-factor c]` | MoE expert parallel, folded into dp*fsdp |
| `--overlap` | async gradient reductions (bucketed per layer) and FSDP parameter prefetch |
| `--recompute` | full activation recomputation, including TP/CP/EP communication |

Rank layout: `worker = ((((pp*DP + dr)*FSDP + ds)*CP + c)*TP + t)`. TP is innermost
(intra-node) and PP outermost. The groups created on worker 0 are printed in
the report.

**Device / numerics.** `--device cpu|gpu`, `--precision fp32|bf16|fp16|fp64`:
- bf16/fp16 = mixed precision: fp32 master weights and AdamW states, and fp32 `main_grad`
  by default (`--grad-dtype compute` for low-precision grads).
- fp16 adds static loss scaling with overflow detection and step skipping.
- `--tf32` uses TF32 tensor cores for fp32 GEMMs.
- `--clip-grad` sets global-norm clipping (one all-reduce).

**MoE router.** You asked for locality control. Choices are random, but their
*placement* is controlled:

| `--router` | behaviour | knobs |
|---|---|---|
| `uniform` | k distinct experts uniformly | |
| `zipf` | popularity-skewed (hot experts, load imbalance) | `--zipf-alpha` |
| `local` | with prob. p pick from the near set, else elsewhere | `--locality p`, `--local-scope rank\|node` |
| `remote` / `far` | with prob. p pick experts on *other* nodes (or ranks) | same |
| `distance` | P(expert) ∝ exp(-β·hops(my node, expert node)); β<0 prefers far nodes | `--distance-beta`, `--topology flat\|ring\|fattree`, `--nodes-per-switch` |
| `node-limited` | DeepSeek-V3-style: each token's experts lie on ≤ M nodes | `--max-nodes M` |
| `round-robin` | perfectly balanced baseline | |

Add `--router-static` for routing that is fixed across iterations. The report shows
the self / intra-node / inter-node split, mean hops, distinct nodes per token,
drop rate, expert-load imbalance and the a2a bytes split.

**Transport.**
- `--transport mpi` (default): one rank per worker. `--cuda-aware 0|1` controls
  passing device buffers to MPI (otherwise pinned-host staging).
  `--ranks-per-node` overrides node detection.
- `--transport hier`: one rank per node, `--workers-per-node` (default = #GPUs),
  `--window-mb`, `--gdr`, `--allow-multi-rank-per-node`. See §3.
- `--sync-collectives` inserts a barrier before each collective, so load-imbalance
  skew (`sync_wait`) is separated from transfer time.

**Clusters.** NERSC Perlmutter batch scripts (functional tests, CPU/GPU weak and strong
scaling, both transports) are in `scripts/perlmutter/` (see its README).

**Replay.** `--replay off|eager|graph`, `--trace-save P`, `--trace-load P`, `--graph-dot P`,
`--arena-mb MB`: record one step and replay it as CUDA graphs, in-process or later from a
file (see §4).

**Output.** `--json f`, `--csv f` (appends one row: ideal for sweeps, see
`scripts/sweep.sh`), `--label`, `--verbose` (per-iteration times and grad norm),
`--selftest` (verifies all collectives / P2P / DistMatrix redistributions on the
active transport).

---------------------------------------------------------------------------------
## 3. Transports: VMM intra-node + CUDA-aware MPI inter-node

`--transport hier` integrates your `vmm::DistributedBuffer`
(`third_party/vmm/vmm_mpi.h`, `vmm_multigpu.h`, unchanged):

* One MPI rank per node drives all its GPUs. Each GPU is a *worker* thread with
  its own CUDA stream and cuBLAS handle.
* Every worker owns one chunk of the node's `MultiGpuBuffer`: a single VMM range,
  physically striped one chunk per GPU and mapped read/write on every GPU. This is
  the worker's **communication window**.
* A **progress engine** on the rank's main thread does the rendezvous, the MPI
  traffic and the intra-node moves (MPI_THREAD_FUNNELED: only this thread calls MPI).
  * **Intra-node:** peers read each other's window chunks directly. Reductions are one
    kernel per destination GPU that loads all local peers' chunks in place (zero copy
    over NVLink/PCIe P2P); copies are P2P DMA over the mappings.
  * **Inter-node:** `DistributedBuffer::isend/irecv` straight from/into the VMM chunks
    (CUDA-aware MPI, GPUDirect RDMA-capable allocations with `--gdr 1`). There are no
    host staging buffers.
  * **Algorithms:** AllGather and AllToAllv are direct exchanges. ReduceScatter has the
    block owner reduce all contributions. AllReduce = ReduceScatter + AllGather
    (2(p-1)/p volume). P2P is matched intra-node and uses MPI inter-node.
  * **Tags:** unique per (role, src GPU, dst GPU, sequence, phase). Messages are posted
    in (src, dst) order, and all collectives are truly asynchronous (`--overlap` works
    without MPI async-progress threads).
* On CPU builds the same engine uses a shared host window, which makes it testable
  anywhere.

`--transport mpi` uses MPI collectives on sub-communicators. Device pointers go
directly to a CUDA-aware MPI. 16-bit reductions stage through host memory, because
user-defined MPI ops are host-only: prefer `--grad-dtype fp32` (default) or `hier`.

---------------------------------------------------------------------------------
## 4. Trace record & replay (CUDA graphs)

xpsim can record one training step (every device op and every communication call)
and replay it, either in the same run or later from files:

```bash
# in-process: warm-up step 0 runs live and is recorded, measured steps replay it
mpirun -np 16 ./bin/xpsim --device gpu --precision bf16 --tp 4 --sp --fsdp 4 --replay graph
# record + save for later (P.manifest + P.w<worker>.xpt)
mpirun -np 16 ./bin/xpsim ... --replay graph --trace-save $SCRATCH/traces/llama8b-tp4
# later, any job: rebuild the CUDA graphs from the files and replay compute + communication
mpirun -np 16 ./bin/xpsim --trace-load $SCRATCH/traces/llama8b-tp4 --iters 50
srun -N 4 --ntasks-per-node=1 ./bin/xpsim --trace-load $SCRATCH/traces/llama8b-tp4 --transport hier
```

| option | meaning |
|---|---|
| `--replay off\|eager\|graph` | record the last warm-up step, then replay the measured steps. `graph`: one CUDA graph per compute segment (GPU); `eager`: re-issue the recorded ops (CPU or GPU) |
| `--trace-save P` | also write the recorded step: `P.manifest` (the full command line) and one `P.w<worker>.xpt` per worker |
| `--trace-load P` | build no model: take the configuration from `P.manifest`, allocate a fresh arena, run the recorded initialization, then replay the step every iteration (default mode: `graph` on GPU, `eager` on CPU) |
| `--graph-dot P` | dump every captured CUDA graph as Graphviz DOT (`P.w<k>.seg<i>.dot`) |
| `--arena-mb MB` | size of the per-worker replay arena (default 85% of free GPU memory; 16 GiB of virtual memory on CPU) |

**How it works.**
- **Arena.** With replay or tracing enabled, all model memory comes from one pre-reserved
  region per worker, and every op stores its buffers as *offsets* into it. Host-side
  inputs (token ids, targets, MoE routing indices and weights) are stored as constants.
  A trace therefore does not depend on addresses: a later process reserves its own arena
  and rebuilds the graphs against the new base address. CUDA cannot serialize a
  `cudaGraph_t` itself; this recipe is what gets saved.
- **Device-resident step.** The vocab-parallel loss statistics, the global gradient norm,
  clipping, the fp16 overflow skip and the AdamW step count all stay on the GPU
  (`dev::OptState`), so a step contains no host decision and no device-to-host copy.
- **Communication.** Collectives and point-to-point calls are recorded at the `Comm` front
  end with their communicator's creation index and request ids. Replay issues the same
  calls (overlap and non-blocking structure included), so timing, bytes and the
  intra/inter-node split are accounted exactly as in a live run. Communication cannot be
  stream-captured over MPI or the hierarchical engine, so the step is split at
  communication calls into compute segments; each segment is one CUDA graph,
  instantiated once and launched every iteration (thread-local capture, so the hier
  transport's GPU workers capture concurrently; fixed cuBLAS workspace).
- **Report.** Each segment's wall time is attributed to categories in proportion to the
  per-op times measured while recording; FLOPs are counted from the recorded ops. The
  report's `execution` line shows the mode, op count, segment count and graph count.

**What may differ between record and replay**: iterations, warm-up, replay mode, transport
(`mpi` <-> `hier`), workers per node, device (cpu <-> gpu), timing and output options.
Model and parallelism options cannot change; trying to raises an error.

**Semantics.** Every replayed step reuses the recorded step's data, i.e. the same tokens
and the same MoE routing (as with `--router-static`). Parameters and optimizer state
evolve normally. Replaying a saved trace from a fresh process reproduces the in-process
run bit for bit on CPU. `scripts/replay_check.sh` checks this, and `make test` runs it
for 7 configurations (TP+SP, FSDP, PP, MoE, 4-D with recompute in bf16, fp16 with loss
scaling, hier transport).

---------------------------------------------------------------------------------
## 5. Code map (Elemental-style matrices)

| file | content |
|---|---|
| `include/xpsim/matrix.hpp` | `Matrix<T>`: El::Matrix analogue (column-major, `Height/Width/LDim/Buffer/Attach/View`), `Gemm`, `Axpy`, `Copy` |
| `include/xpsim/dist_matrix.hpp` | `DistGrid` (El::Grid: MC/MR/VC comms) and `DistMatrix<T,U,V>` with `U,V ∈ {MC,MR,VC,STAR}` (block distribution). `Copy` = AllGather / local filter, `Contract` = ReduceScatter, `AllReduce` |
| `include/xpsim/flat_param.hpp` | FSDP flat parameter: ZeRO 0-3, HSDP, prefetch, async reduce-scatter/all-reduce, AdamW on the master shard |
| `include/xpsim/grid.hpp`, `src/grid.cpp` | 5-D process grid and all groups (tp, cp, fsdp, hsdp, dp, pp, ep, edp) |
| `src/trainer.cpp` | transformer layers (GQA attention, CP, TP/SP, MLP / MoE), embedding/LM head, 1F1B/GPipe schedules, step |
| `include/xpsim/moe_router.hpp`, `src/moe_router.cpp` | locality-aware MoE router |
| `include/xpsim/comm.hpp`, `src/comm.cpp`, `src/comm_hier.cpp`, `src/comm_vmm.cpp` | worker-level `Comm`, MPI transport, hierarchical engine, VMM windows |
| `include/xpsim/device.hpp`, `src/device*.{cpp,cu}` | backend: CPU (OpenMP blocked GEMM or CBLAS) and CUDA (cuBLAS GemmEx + kernels); allocation tracking |
| `src/metrics.cpp`, `src/report.cpp` | counters and report |
| `include/xpsim/trace.hpp`, `src/trace.cpp` | trace recording, file format, replayer (eager / CUDA graphs) |
| `src/selftest.cpp` | transport / redistribution verification |
| `tools/mpishim` | thread-based MPI subset for testing without MPI |

**GEMM engines.**
- GPU (sm_80+): `cublasGemmStridedBatchedEx`.
  - FP64 → `CUBLAS_COMPUTE_64F` (DMMA FP64 tensor cores).
  - FP32 → `CUBLAS_COMPUTE_32F_FAST_TF32` with `--tf32`.
  - BF16/FP16 → HMMA with FP32 accumulation; weight-gradient GEMMs accumulate directly into fp32 `main_grad`.
- CPU: vendor `cblas_?gemm` when built with `BLAS=`, else a blocked OpenMP kernel.
  16-bit operands are widened to fp32.

---------------------------------------------------------------------------------
## 6. Figures of merit and metrics

| metric | definition |
|---|---|
| iteration time | per iteration, max over workers (mean, std, min, max over measured iterations; warm-up excluded) |
| throughput | tokens/s, samples/s, tokens/s/worker |
| model FLOPs | GEMM + attention FLOPs actually executed (excl. recompute), summed over workers; checked against the analytic 6·N_active + 12·L·s·h_attn per token (ratio printed) |
| MFU / HFU | model (hardware, incl. recompute) FLOPs / (iter time · workers · peak). Peak = `--peak-tflops` or measured attainable GEMM rate |
| compute / comm / other | per-worker exposed time; comm fraction; compute imbalance (max/mean) |
| per category | 8 compute + 19 communication categories: mean/max ms, % of iteration, calls, bytes, algBW, NCCL-style busBW, intra- vs inter-node bytes |
| pipeline | theoretical bubble (p-1)/(m+p-1) vs measured P2P+bubble wait per stage |
| MoE | routing destination split, hops, nodes/token, drops, expert-load imbalance, a2a volume |
| memory | peak per worker by category (params, grads, optimizer, activations, comm buffers) |

The JSON contains all of the above plus per-iteration times. The CSV has one row
per run.

---------------------------------------------------------------------------------
## 7. Validation and limitations

- Try `--selftest` on both transports (fp32/fp64/bf16/fp16, AR/AG/RS/A2Av/P2P/DistMatrix).
- Gradient norms should be bit-identical between `mpi` and `hier` transports.
- Counted FLOPs must match the analytic model.

**Modelling simplifications:**
- No LayerNorm/bias/dropout.
- GELU 2-matrix MLP: for SwiGLU FLOP parity set `--ffn` to 1.5× the SwiGLU width.
- Attention materializes the score matrix (no flash attention) and computes the
  full (unmasked) score GEMMs.
- CP uses contiguous chunks (no arbitrary zig-zag load balancing).
- Experts are not tensor-parallel.
- No interleaved/virtual PP stages.
- Router weights are random; the gating gradient is synthetic (its cost and its
  TP all-reduce are real).
