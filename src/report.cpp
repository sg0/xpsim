// Gathers per-worker counters on worker 0 and prints figures of merit,
// time/communication breakdowns, pipeline/MoE/memory statistics; optional
// JSON and CSV output for sweeps.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include "xpsim/grid.hpp"
#include "xpsim/trainer.hpp"

namespace xpsim {

namespace {
const char* kTags[] = {"params", "grads", "optimizer", "activations", "comm_buffers", "workspace"};
constexpr int NTAG = 6, NMOE = 11, NCOORD = 6;
constexpr int OFF_CAT = 0, OFF_FLOP = 6 * NCAT, OFF_MOE = OFF_FLOP + 2, OFF_MEM = OFF_MOE + NMOE,
              OFF_SKIP = OFF_MEM + 2 + NTAG, OFF_COORD = OFF_SKIP + 1, OFF_IT = OFF_COORD + NCOORD;

std::string fmt(const char* f, double v) { char b[64]; std::snprintf(b, sizeof b, f, v); return b; }
std::string human(double v, const char* unit) {
  const char* p[] = {"", "K", "M", "G", "T", "P"};
  int i = 0; while (std::fabs(v) >= 1000 && i < 5) { v /= 1000; ++i; }
  char b[64]; std::snprintf(b, sizeof b, "%.3g %s%s", v, p[i], unit); return b;
}
std::string mib(double b) { return fmt("%.1f", b / (1024.0 * 1024.0)); }
}  // namespace

void report(const Config& cfg, const ProcessGrid& g, double peak, const ModelCounts& mc) {
  auto& m = metrics();
  const int iters = int(m.iter_times.size());
  const int n = OFF_IT + iters;
  std::vector<double> v(n, 0.0);
  for (int c = 0; c < NCAT; ++c) {
    const auto& s = m.cat[c];
    double* p = &v[OFF_CAT + 6 * c];
    p[0] = s.time; p[1] = s.calls; p[2] = s.bytes; p[3] = s.busbytes; p[4] = s.intra_bytes; p[5] = s.inter_bytes;
  }
  v[OFF_FLOP] = m.flops_model; v[OFF_FLOP + 1] = m.flops_hw;
  const auto& q = m.moe;
  double moe[NMOE] = {q.assignments, q.to_self, q.intra_node, q.inter_node, q.dropped, q.distance, q.max_expert_load, q.mean_expert_load, q.layers, q.tokens, q.token_nodes};
  std::copy(moe, moe + NMOE, &v[OFF_MOE]);
  v[OFF_MEM] = double(dev::mem_peak()); v[OFF_MEM + 1] = double(dev::mem_current());
  for (int t = 0; t < NTAG; ++t) {
    auto it = dev::mem_stats().find(kTags[t]);
    v[OFF_MEM + 2 + t] = it == dev::mem_stats().end() ? 0 : double(it->second.peak);
  }
  v[OFF_SKIP] = m.skipped_steps;
  double co[NCOORD] = {double(g.me.pp), double(g.me.dr), double(g.me.ds), double(g.me.c), double(g.me.t), double(node_of(my_worker()))};
  std::copy(co, co + NCOORD, &v[OFF_COORD]);
  for (int i = 0; i < iters; ++i) v[OFF_IT + i] = m.iter_times[i];

  auto& world = Comm::world();
  const int W = world.size();
  std::vector<double> all(size_t(n) * W);
  world.allgather_host(v.data(), all.data(), n, Cat::MISC_COMM);
  if (my_worker() != 0) return;
  auto at = [&](int w, int off) { return all[size_t(w) * n + off]; };

  // ---------------- iteration time ----------------
  std::vector<double> it_max(iters, 0.0);
  for (int i = 0; i < iters; ++i) for (int w = 0; w < W; ++w) it_max[i] = std::max(it_max[i], at(w, OFF_IT + i));
  const double t_mean = std::accumulate(it_max.begin(), it_max.end(), 0.0) / std::max(1, iters);
  double t_var = 0; for (double x : it_max) t_var += (x - t_mean) * (x - t_mean);
  const double t_std = iters > 1 ? std::sqrt(t_var / (iters - 1)) : 0;
  const double t_min = *std::min_element(it_max.begin(), it_max.end()), t_max = *std::max_element(it_max.begin(), it_max.end());

  double fm = 0, fh = 0;
  for (int w = 0; w < W; ++w) { fm += at(w, OFF_FLOP); fh += at(w, OFF_FLOP + 1); }
  fm /= iters; fh /= iters;  // per iteration, all workers
  const double tokens = double(cfg.tokens_per_iter());
  const double analytic = mc.model_flops_per_token * tokens;
  const double mfu = fm / (t_mean * W * peak * 1e12), hfu = fh / (t_mean * W * peak * 1e12);

  // ---------------- per-category stats ----------------
  struct Row { double tmean = 0, tmax = 0, calls = 0, bytes = 0, bus = 0, intra = 0, inter = 0; };
  std::vector<Row> rows(NCAT);
  std::vector<double> comp(W, 0), comm(W, 0), iter_w(W, 0);
  for (int w = 0; w < W; ++w) {
    for (int i = 0; i < iters; ++i) iter_w[w] += at(w, OFF_IT + i);
    iter_w[w] /= iters;
  }
  for (int c = 0; c < NCAT; ++c) {
    for (int w = 0; w < W; ++w) {
      const double t = at(w, OFF_CAT + 6 * c) / iters;
      rows[c].tmean += t / W; rows[c].tmax = std::max(rows[c].tmax, t);
      rows[c].calls += at(w, OFF_CAT + 6 * c + 1) / iters / W;
      rows[c].bytes += at(w, OFF_CAT + 6 * c + 2) / iters / W;
      rows[c].bus += at(w, OFF_CAT + 6 * c + 3) / iters / W;
      rows[c].intra += at(w, OFF_CAT + 6 * c + 4) / iters / W;
      rows[c].inter += at(w, OFF_CAT + 6 * c + 5) / iters / W;
      (is_comm(Cat(c)) ? comm[w] : comp[w]) += t;
    }
  }
  const double comp_mean = std::accumulate(comp.begin(), comp.end(), 0.0) / W;
  const double comp_max = *std::max_element(comp.begin(), comp.end());
  const double comm_mean = std::accumulate(comm.begin(), comm.end(), 0.0) / W;
  double other_mean = 0; for (int w = 0; w < W; ++w) other_mean += (iter_w[w] - comp[w] - comm[w]) / W;

  std::ostringstream o;
  o << "\n================================ xpsim report ================================\n";
  if (!cfg.label.empty()) o << "label: " << cfg.label << "\n";
  o << describe(cfg) << "\n";
  o << "workers: " << W << " (" << num_nodes() << " node(s) x " << workers_per_node() << ")  device: " << dev::device_name()
    << "  GEMM: " << dev::gemm_engine() << "\n";
  o << "groups (worker 0):\n" << g.describe_groups();
  o << "model: " << human(mc.params_total, "") << " params (" << human(mc.params_dense, "") << " dense, "
    << human(mc.params_expert, "") << " expert), " << human(mc.params_active, "") << " active/token; "
    << human(mc.model_flops_per_token, "FLOP") << "/token (6N + attention)\n";

  o << "\n---- figures of merit ---------------------------------------------------------\n";
  o << "iteration time      : " << fmt("%.4f", t_mean) << " s  (std " << fmt("%.4f", t_std) << ", min " << fmt("%.4f", t_min)
    << ", max " << fmt("%.4f", t_max) << ", " << iters << " iters, max over workers)\n";
  o << "throughput          : " << human(tokens / t_mean, "tokens/s") << "  |  " << fmt("%.2f", cfg.global_batch() / t_mean)
    << " samples/s  |  " << human(tokens / t_mean / W, "tokens/s/worker") << "\n";
  o << "model FLOPs / iter  : counted " << human(fm, "FLOP") << "  analytic " << human(analytic, "FLOP") << "  (ratio "
    << fmt("%.3f", fm / analytic) << ")\n";
  o << "achieved per worker : " << fmt("%.3f", fm / t_mean / W / 1e12) << " TFLOP/s model, " << fmt("%.3f", fh / t_mean / W / 1e12)
    << " TFLOP/s hardware (incl. recompute)\n";
  o << "peak per worker     : " << fmt("%.3f", peak) << " TFLOP/s " << (cfg.peak_tflops > 0 ? "(given)" : "(measured GEMM)") << "\n";
  o << "MFU / HFU           : " << fmt("%.2f", 100 * mfu) << " % / " << fmt("%.2f", 100 * hfu) << " %\n";
  o << "compute / comm / other (mean per worker): " << fmt("%.1f", 1e3 * comp_mean) << " / " << fmt("%.1f", 1e3 * comm_mean)
    << " / " << fmt("%.1f", 1e3 * other_mean) << " ms   comm fraction " << fmt("%.1f", 100 * comm_mean / t_mean) << " %\n";
  o << "compute imbalance   : max/mean " << fmt("%.3f", comp_mean > 0 ? comp_max / comp_mean : 1) << "\n";
  double skipped = 0; for (int w = 0; w < W; ++w) skipped = std::max(skipped, at(w, OFF_SKIP));
  if (skipped > 0) o << "skipped steps       : " << skipped << " (fp16 overflow)\n";

  o << "\n---- time breakdown per iteration ---------------------------------------------\n";
  o << "category                 mean ms    max ms  %iter  calls    MB/iter  algBW GB/s busBW GB/s  intra MB  inter MB\n";
  for (int c = 0; c < NCAT; ++c) {
    const Row& r = rows[c];
    if (r.calls == 0 && r.tmean == 0) continue;
    char b[256];
    const bool cm = is_comm(Cat(c));
    std::snprintf(b, sizeof b, "%-22s %9.2f %9.2f %6.1f %6.0f %10s %11s %10s %9s %9s\n", cat_name(Cat(c)), 1e3 * r.tmean,
                  1e3 * r.tmax, 100 * r.tmean / t_mean, r.calls, cm ? mib(r.bytes).c_str() : "",
                  cm && r.tmean > 0 ? fmt("%.2f", r.bytes / r.tmean / 1e9).c_str() : "",
                  cm && r.tmean > 0 ? fmt("%.2f", r.bus / r.tmean / 1e9).c_str() : "", cm ? mib(r.intra).c_str() : "",
                  cm ? mib(r.inter).c_str() : "");
    o << b;
  }
  o << "(comm times are exposed wall time on the worker incl. waiting for peers; with --overlap they exclude hidden\n"
       " transfer time. --sync-collectives separates skew (sync_wait) from transfer time.)\n";

  double vb = 0, vi = 0, vx = 0;
  for (int c = FIRST_COMM; c < NCAT; ++c) { vb += rows[c].bytes; vi += rows[c].intra; vx += rows[c].inter; }
  o << "comm volume / worker / iter: " << mib(vb) << " MiB  (intra-node " << mib(vi) << " MiB, inter-node " << mib(vx) << " MiB)\n";

  if (cfg.pp > 1) {
    const double mth = double(cfg.pp - 1) / (cfg.microbatches + cfg.pp - 1);
    o << "\n---- pipeline ------------------------------------------------------------------\n";
    o << "theoretical bubble (p-1)/(m+p-1): " << fmt("%.1f", 100 * mth) << " %\n";
    for (int s = 0; s < cfg.pp; ++s) {
      double t = 0, cpt = 0; int k = 0;
      for (int w = 0; w < W; ++w) if (int(at(w, OFF_COORD)) == s) {
        t += at(w, OFF_CAT + 6 * int(Cat::PP_P2P)) / iters; cpt += comp[w]; ++k; }
      o << "  stage " << s << ": p2p+bubble wait " << fmt("%.2f", 1e3 * t / k) << " ms (" << fmt("%.1f", 100 * t / k / t_mean)
        << " %), compute " << fmt("%.2f", 1e3 * cpt / k) << " ms\n";
    }
  }
  if (cfg.experts > 0) {
    double a[NMOE] = {0}; double li = 0;
    for (int w = 0; w < W; ++w) {
      for (int k = 0; k < NMOE; ++k) a[k] += at(w, OFF_MOE + k);
      double mx = at(w, OFF_MOE + 6), mn = at(w, OFF_MOE + 7);
      if (mn > 0) li = std::max(li, mx / mn);
    }
    const double kept = a[1] + a[2] + a[3];
    o << "\n---- MoE / expert parallelism -------------------------------------------------\n";
    o << "router: " << router_name(cfg.router) << "  assignments/iter " << human(a[0] / iters, "") << "  dropped "
      << fmt("%.2f", 100 * a[4] / std::max(1.0, a[0])) << " %\n";
    o << "destination: self " << fmt("%.1f", 100 * a[1] / std::max(1.0, kept)) << " %, intra-node "
      << fmt("%.1f", 100 * a[2] / std::max(1.0, kept)) << " %, inter-node " << fmt("%.1f", 100 * a[3] / std::max(1.0, kept))
      << " %   mean hops " << fmt("%.3f", a[5] / std::max(1.0, kept)) << "   distinct nodes/token "
      << fmt("%.2f", a[10] / std::max(1.0, a[9])) << "\n";
    o << "expert load imbalance (max over workers of max/mean tokens per local expert): " << fmt("%.3f", li) << "\n";
    const Row& r = rows[int(Cat::EP_A2A)];
    o << "all-to-all: " << mib(r.bytes) << " MiB/iter/worker (self " << mib(r.bytes - r.intra - r.inter) << ", intra " << mib(r.intra) << ", inter " << mib(r.inter) << "), "
      << fmt("%.2f", 1e3 * r.tmean) << " ms\n";
  }
  o << "\n---- memory (peak per worker, max over workers) -------------------------------\n";
  double mpk = 0; for (int w = 0; w < W; ++w) mpk = std::max(mpk, at(w, OFF_MEM));
  o << "total " << mib(mpk) << " MiB :";
  for (int t = 0; t < NTAG; ++t) {
    double x = 0; for (int w = 0; w < W; ++w) x = std::max(x, at(w, OFF_MEM + 2 + t));
    o << " " << kTags[t] << " " << mib(x);
  }
  o << "\n===============================================================================\n";
  std::fputs(o.str().c_str(), stdout);
  std::fflush(stdout);

  if (!cfg.json.empty()) {
    std::ofstream j(cfg.json);
    j << "{\n  \"label\": \"" << cfg.label << "\",\n  \"workers\": " << W << ", \"nodes\": " << num_nodes()
      << ",\n  \"config\": {\"tp\": " << cfg.tp << ", \"pp\": " << cfg.pp << ", \"dp\": " << cfg.dp << ", \"fsdp\": " << cfg.fsdp
      << ", \"cp\": " << cfg.cp << ", \"ep\": " << cfg.ep << ", \"zero\": " << cfg.zero_stage() << ", \"sp\": " << cfg.sp
      << ", \"overlap\": " << cfg.overlap << ", \"recompute\": " << cfg.recompute << ", \"precision\": \"" << dtype_name(cfg.precision)
      << "\", \"device\": \"" << (cfg.gpu ? "gpu" : "cpu") << "\", \"transport\": \"" << (cfg.rt.transport == Transport::MPI ? "mpi" : "hier")
      << "\", \"router\": \"" << router_name(cfg.router) << "\", \"layers\": " << cfg.layers << ", \"hidden\": " << cfg.hidden
      << ", \"seq\": " << cfg.seq << ", \"experts\": " << cfg.experts << ", \"global_batch\": " << cfg.global_batch() << "},\n";
    j << "  \"fom\": {\"iter_time_s\": " << t_mean << ", \"iter_time_std_s\": " << t_std << ", \"tokens_per_s\": " << tokens / t_mean
      << ", \"tflops_per_worker\": " << fm / t_mean / W / 1e12 << ", \"mfu\": " << mfu << ", \"hfu\": " << hfu
      << ", \"peak_tflops\": " << peak << ", \"comm_fraction\": " << comm_mean / t_mean << ", \"compute_imbalance\": "
      << (comp_mean > 0 ? comp_max / comp_mean : 1) << ", \"peak_mem_bytes\": " << mpk << "},\n  \"categories\": {";
    bool first = true;
    for (int c = 0; c < NCAT; ++c) {
      const Row& r = rows[c];
      if (r.calls == 0 && r.tmean == 0) continue;
      j << (first ? "\n" : ",\n") << "    \"" << cat_name(Cat(c)) << "\": {\"mean_s\": " << r.tmean << ", \"max_s\": " << r.tmax
        << ", \"calls\": " << r.calls << ", \"bytes\": " << r.bytes << ", \"busbytes\": " << r.bus << ", \"intra_bytes\": "
        << r.intra << ", \"inter_bytes\": " << r.inter << "}";
      first = false;
    }
    j << "\n  },\n  \"iter_times_s\": [";
    for (int i = 0; i < iters; ++i) j << (i ? ", " : "") << it_max[i];
    j << "]\n}\n";
  }
  if (!cfg.csv.empty()) {
    bool fresh = !std::ifstream(cfg.csv).good();
    std::ofstream c(cfg.csv, std::ios::app);
    if (fresh) {
      c << "label,workers,nodes,tp,pp,dp,fsdp,cp,ep,zero,sp,overlap,recompute,precision,device,transport,router,iter_s,tokens_per_s,"
           "tflops_per_worker,mfu,comm_frac,peak_mem_mib";
      for (int k = 0; k < NCAT; ++k) c << "," << cat_name(Cat(k)) << "_ms";
      c << "\n";
    }
    c << cfg.label << "," << W << "," << num_nodes() << "," << cfg.tp << "," << cfg.pp << "," << cfg.dp << "," << cfg.fsdp << ","
      << cfg.cp << "," << cfg.ep << "," << cfg.zero_stage() << "," << cfg.sp << "," << cfg.overlap << "," << cfg.recompute << ","
      << dtype_name(cfg.precision) << "," << (cfg.gpu ? "gpu" : "cpu") << "," << (cfg.rt.transport == Transport::MPI ? "mpi" : "hier")
      << "," << router_name(cfg.router) << "," << t_mean << "," << tokens / t_mean << "," << fm / t_mean / W / 1e12 << "," << mfu
      << "," << comm_mean / t_mean << "," << mpk / 1048576.0;
    for (int k = 0; k < NCAT; ++k) c << "," << 1e3 * rows[k].tmean;
    c << "\n";
  }
}

}  // namespace xpsim
