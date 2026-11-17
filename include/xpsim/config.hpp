// Simulation configuration (model, parallelism, MoE router, device, run).
#pragma once
#include <string>
#include <vector>
#include "xpsim/comm.hpp"
#include "xpsim/types.hpp"

namespace xpsim {

enum class RouterPolicy { UNIFORM, ZIPF, LOCAL, REMOTE, DISTANCE, NODE_LIMITED, ROUND_ROBIN };
enum class NetTopology { FLAT, RING, FATTREE };
enum class PPSchedule { GPIPE, ONE_F_ONE_B };

struct Config {
  // ---- model (decoder-only transformer, pre-LN, GQA attention, optional MoE)
  int layers = 4, hidden = 512, heads = 8, kv_heads = 0 /*0 = heads (MHA)*/, ffn = 0 /*0 = 4*hidden*/;
  int seq = 256, vocab = 8192, micro_batch = 2, microbatches = 2;
  bool causal = true;
  // ---- MoE
  int experts = 0, topk = 2, moe_every = 1, expert_ffn = 0 /*0 = ffn*/;
  double capacity_factor = 0.0;  // 0 = dropless
  // ---- MoE router (expert choice + locality control)
  RouterPolicy router = RouterPolicy::UNIFORM;
  double zipf_alpha = 1.0;      // ZIPF: expert popularity ~ 1/rank^alpha
  double locality = 0.8;        // LOCAL/REMOTE: probability a choice is drawn from the near/far set
  std::string local_scope = "node";  // LOCAL/REMOTE near set: "rank" (own experts) or "node"
  double distance_beta = 1.0;   // DISTANCE: P(expert) ~ exp(-beta * hops); beta<0 prefers far experts
  int max_nodes = 1;            // NODE_LIMITED: each token's experts span at most this many nodes
  NetTopology topology = NetTopology::FLAT;
  int nodes_per_switch = 4;     // FATTREE leaf size
  bool router_static = false;   // same routing every iteration (else re-drawn per iteration)
  // ---- parallelism
  int tp = 1, pp = 1, dp = 1 /*replicate*/, fsdp = 1 /*shard*/, cp = 1, ep = 1;
  int zero = -1;  // -1: 3 if fsdp>1 else 0
  bool sp = false, overlap = false, recompute = false;
  PPSchedule schedule = PPSchedule::ONE_F_ONE_B;
  // ---- device / numerics
  bool gpu = false;
  DType precision = DType::F32;   // compute/activation dtype (bf16/fp16 = mixed precision)
  bool grad_fp32 = true;          // fp32 main_grad buffers (Megatron default) vs compute dtype
  bool tf32 = false;
  double loss_scale = 0;          // fp16: static loss scale (0 = 65536 for fp16, 1 otherwise)
  double clip_grad = 1.0;         // global-norm clipping (0 disables; adds an all-reduce)
  double lr = 1e-4;
  // ---- transport
  RuntimeOptions rt;
  // ---- run / reporting
  int iters = 5, warmup = 2;
  unsigned long long seed = 1234;
  double peak_tflops = 0;   // 0 = measure attainable GEMM peak at start-up
  bool sync_timers = true, verbose = false, selftest = false;
  std::string json, csv, label;

  // derived
  int head_dim() const { return hidden / heads; }
  int dp_total() const { return dp * fsdp; }
  int world() const { return pp * dp * fsdp * cp * tp; }
  int zero_stage() const { return zero >= 0 ? zero : (fsdp > 1 ? 3 : 0); }
  bool is_moe_layer(int l) const { return experts > 0 && moe_every > 0 && (l % moe_every) == moe_every - 1; }
  long long global_batch() const { return (long long)micro_batch * microbatches * dp_total(); }
  long long tokens_per_iter() const { return global_batch() * seq; }
};

Config parse_config(int argc, char** argv);  // throws on error
void validate(const Config& c, int world);
std::string usage();
std::string describe(const Config& c);
const char* router_name(RouterPolicy r);

}  // namespace xpsim
