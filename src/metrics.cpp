#include "xpsim/metrics.hpp"

namespace xpsim {

static const char* kNames[NCAT] = {
    "gemm_fwd", "gemm_bwd", "attn_fwd", "attn_bwd", "recompute", "elementwise", "moe_route", "optimizer",
    "tp_allreduce", "tp_allgather", "tp_reducescatter", "cp_allgather", "cp_reducescatter", "dp_allreduce",
    "fsdp_allgather", "fsdp_reducescatter", "hsdp_allreduce", "zero_param_allgather", "expert_allreduce",
    "expert_reducescatter", "expert_allgather", "ep_alltoall", "pp_p2p", "loss_allreduce", "gradnorm_allreduce",
    "misc_comm", "sync_wait"};
static const char* kDesc[NCAT] = {
    "forward GEMMs (QKV, out-proj, MLP, experts, head)",
    "backward GEMMs (dX and dW)",
    "attention core forward (QK^T, softmax, PV)",
    "attention core backward",
    "activation recomputation (forward re-run in backward)",
    "elementwise (GELU, residual, casts, embedding)",
    "MoE router + token permutation",
    "optimizer step (AdamW on master shard, casts)",
    "TP all-reduce of row-parallel outputs / input grads",
    "sequence-parallel all-gather (TP group)",
    "sequence-parallel reduce-scatter (TP group)",
    "context-parallel K/V all-gather",
    "context-parallel dK/dV reduce-scatter",
    "data-parallel gradient all-reduce (ZeRO-0)",
    "FSDP/ZeRO-3 parameter all-gather (per layer, fwd+bwd)",
    "FSDP/ZeRO-2/3 gradient reduce-scatter",
    "HSDP/outer-DP all-reduce of gradient shards",
    "ZeRO-1/2 updated-parameter all-gather",
    "expert-data-parallel gradient all-reduce",
    "expert gradient reduce-scatter (ZeRO>=2)",
    "expert parameter all-gather (ZeRO)",
    "expert-parallel token dispatch/combine all-to-all",
    "pipeline-parallel activation/gradient send+recv (incl. bubble wait)",
    "vocab-parallel cross-entropy statistics",
    "global gradient-norm / overflow check",
    "barriers and other small collectives",
    "pre-collective barrier wait (--sync-collectives: load imbalance/skew)"};

const char* cat_name(Cat c) { return kNames[int(c)]; }
const char* cat_desc(Cat c) { return kDesc[int(c)]; }

double busbw_factor(CollKind k, int p) {
  if (p <= 1) return 0.0;
  switch (k) {
    case CollKind::ALLREDUCE: return 2.0 * (p - 1) / p;
    case CollKind::ALLGATHER: case CollKind::REDUCESCATTER: case CollKind::ALLTOALL: return double(p - 1) / p;
    case CollKind::P2P: case CollKind::BCAST: return 1.0;
    default: return 0.0;
  }
}

void Metrics::reset() {
  cat = {}; flops_model = flops_hw = 0; moe = MoEStats{}; iter_times.clear(); skipped_steps = 0;
}

Metrics& metrics() { static thread_local Metrics m; return m; }

}  // namespace xpsim
