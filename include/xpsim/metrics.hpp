// Per-worker performance counters: time / bytes per compute and communication
// category, FLOP counters, MoE routing statistics, memory.
#pragma once
#include <array>
#include <chrono>
#include <string>
#include <vector>
#include "xpsim/device.hpp"

namespace xpsim {

enum class Cat : int {
  // compute
  GEMM_FWD, GEMM_BWD, ATTN_FWD, ATTN_BWD, RECOMPUTE, ELEMWISE, MOE_ROUTE, OPTIMIZER,
  // communication
  TP_AR, TP_AG, TP_RS, CP_AG, CP_RS, DP_AR, FSDP_AG, FSDP_RS, HSDP_AR, ZERO_AG,
  EXP_AR, EXP_RS, EXP_AG, EP_A2A, PP_P2P, LOSS_AR, GRADNORM_AR, MISC_COMM, SYNC_WAIT,
  COUNT
};
constexpr int NCAT = int(Cat::COUNT);
constexpr int FIRST_COMM = int(Cat::TP_AR);
inline bool is_comm(Cat c) { return int(c) >= FIRST_COMM; }
const char* cat_name(Cat c);
const char* cat_desc(Cat c);

// NCCL-tests style bus-bandwidth factor for a collective over p ranks.
enum class CollKind { ALLREDUCE, ALLGATHER, REDUCESCATTER, ALLTOALL, P2P, BCAST, BARRIER };
double busbw_factor(CollKind k, int p);

struct CatStats {
  double time = 0, calls = 0, bytes = 0, busbytes = 0, intra_bytes = 0, inter_bytes = 0;
};

struct MoEStats {
  double assignments = 0, to_self = 0, intra_node = 0, inter_node = 0, dropped = 0, distance = 0;
  double tokens = 0, token_nodes = 0;  // distinct destination nodes per token (node-limited routing)
  double max_expert_load = 0, mean_expert_load = 0, layers = 0;  // summed over (layer, mb) events
  double a2a_intra_bytes = 0, a2a_inter_bytes = 0;
};

struct Metrics {
  std::array<CatStats, NCAT> cat{};
  double flops_model = 0, flops_hw = 0;  // GEMM + attention FLOPs (model excludes recompute)
  MoEStats moe;
  std::vector<double> iter_times;
  double skipped_steps = 0;
  bool recording = false;   // false during warm-up
  bool sync_timers = true;  // device-sync around timed regions (accurate GPU timing)
  bool in_recompute = false;
  void reset();
};

Metrics& metrics();  // thread-local (one per worker)

inline double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class ScopedTimer {
 public:
  explicit ScopedTimer(Cat c) : c_(c) {
    if (metrics().sync_timers && dev::is_gpu()) dev::sync();
    t0_ = now_sec();
  }
  ~ScopedTimer() {
    auto& m = metrics();
    if (m.sync_timers && dev::is_gpu()) dev::sync();
    if (!m.recording) return;
    Cat c = (m.in_recompute && !is_comm(c_) && c_ != Cat::OPTIMIZER) ? Cat::RECOMPUTE : c_;
    m.cat[int(c)].time += now_sec() - t0_;
    m.cat[int(c)].calls += 1;
  }
 private:
  Cat c_; double t0_;
};

inline void count_flops(double f) {
  auto& m = metrics();
  if (!m.recording) return;
  m.flops_hw += f;
  if (!m.in_recompute) m.flops_model += f;
}
inline void count_bytes(Cat c, double bytes, double busbytes, double intra, double inter) {
  auto& m = metrics();
  if (!m.recording) return;
  auto& s = m.cat[int(c)];
  s.bytes += bytes; s.busbytes += busbytes; s.intra_bytes += intra; s.inter_bytes += inter;
}

}  // namespace xpsim
