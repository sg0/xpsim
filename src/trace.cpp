// Trace recording, serialization and replay (see include/xpsim/trace.hpp).
#include "xpsim/trace.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include "xpsim/device.hpp"

namespace xpsim {

// =================================================================== recorder
namespace {
struct Rec {
  Trace* t = nullptr;
  std::vector<TOp>* list = nullptr;
  int next_req = 0;
};
thread_local Rec R;
}  // namespace

namespace trace {
bool active() { return R.list != nullptr && !dev::suspended(); }
void rec(TOpKind k, std::initializer_list<int64_t> a, std::initializer_list<double> f) {
  rec_vec(k, std::vector<int64_t>(a), std::vector<double>(f));
}
void rec_vec(TOpKind k, std::vector<int64_t> a, std::vector<double> f) {
  TOp op;
  op.kind = k;
  op.recompute = metrics().in_recompute ? 1 : 0;
  op.a = std::move(a);
  op.f = std::move(f);
  R.list->push_back(std::move(op));
}
int64_t ptr(const void* p) {
  if (!p) return -1;
  long long off = dev::arena_offset(p);
  if (off < 0) fail("trace: device buffer outside the arena (allocated before arena_begin or while suspended)");
  return off;
}
int64_t cst(const void* host, size_t bytes) {
  auto& c = R.t->consts;
  const uint8_t* b = static_cast<const uint8_t*>(host);
  c.emplace_back(b, b + bytes);
  return -(2 + int64_t(c.size() - 1));
}
int next_request_id() { return R.next_req++; }
void begin(Trace* t, std::vector<TOp>* list) { R.t = t; R.list = list; R.next_req = 0; }
void end() { R.list = nullptr; }
Trace* current() { return R.t; }
}  // namespace trace

static const char* kSpanNames[] = {"F", "B", "grad sync", "optimizer step"};
int span_name_id(const char* name) {
  for (int i = 0; i < int(sizeof(kSpanNames) / sizeof(kSpanNames[0])); ++i)
    if (std::strcmp(kSpanNames[i], name) == 0) return i;
  return -1;
}
const char* span_name(int id) { return kSpanNames[id]; }
void trace_mark_span(const char* name, bool begin, int mb, int chunk) {
  if (!trace::active()) return;
  const int id = span_name_id(name);
  if (id >= 0) trace::rec(TOpKind::C_MARK, {id, begin ? 1 : 0, mb, chunk});
}

long trace_mark() { return R.list ? long(R.list->size()) : -1; }
void trace_attribute(long mark, double seconds, Cat c) {
  if (!R.list || mark < 0) return;
  std::vector<TOp*> fresh;
  for (size_t i = size_t(mark); i < R.list->size(); ++i) {
    TOp& op = (*R.list)[i];
    if (!is_comm_op(op.kind) && op.cat == 255) fresh.push_back(&op);
  }
  for (TOp* op : fresh) { op->cat = uint8_t(c); op->time = seconds / fresh.size(); }
}
void trace_add_flops(double f) {
  if (!R.list || R.list->empty() || dev::suspended()) return;
  R.list->back().flops += f;
}

// =================================================================== helpers
uint64_t hash_args(const std::vector<std::string>& args) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const std::string& s) { for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ull; } h ^= 0xff; h *= 1099511628211ull; };
  for (size_t i = 0; i < args.size(); ++i) {
    std::string a = args[i];
    if (a.rfind("--", 0) != 0) continue;
    std::string k = a.substr(2), v;
    auto eq = k.find('=');
    if (eq != std::string::npos) { v = k.substr(eq + 1); k = k.substr(0, eq); }
    else if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) v = args[++i];
    else v = "1";
    if (replay_safe_option(k)) continue;
    mix(k); mix(v);
  }
  return h;
}
bool replay_safe_option(const std::string& k) {
  static const std::set<std::string> ok = {
      "iters", "warmup", "replay", "trace-save", "trace-load", "sync-timers", "sync-collectives", "json", "csv", "label",
      "verbose", "transport", "workers-per-node", "ranks-per-node", "window-mb", "cuda-aware", "gdr",
      "allow-multi-rank-per-node", "peak-tflops", "graph-dot", "arena-mb", "device", "tf32", "timeline",
      "timeline-iters"};
  return ok.count(k) > 0;
}

// =================================================================== file format
namespace {
constexpr char kMagic[8] = {'X', 'P', 'T', 'R', 'A', 'C', 'E', '1'};
struct W {
  std::ofstream o;
  template <class T> void pod(const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
  void str(const std::string& s) { pod(uint64_t(s.size())); o.write(s.data(), s.size()); }
  template <class T> void vec(const std::vector<T>& v) { pod(uint64_t(v.size())); o.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T)); }
  void ops(const std::vector<TOp>& v) {
    pod(uint64_t(v.size()));
    for (auto& op : v) { pod(uint16_t(op.kind)); pod(op.cat); pod(op.recompute); pod(op.flops); pod(op.time); vec(op.a); vec(op.f); }
  }
};
struct Rd {
  std::ifstream i;
  std::string path;
  template <class T> T pod() { T v{}; i.read(reinterpret_cast<char*>(&v), sizeof(T)); if (!i) fail("truncated trace file " + path); return v; }
  std::string str() { auto n = pod<uint64_t>(); std::string s(n, '\0'); i.read(s.data(), n); return s; }
  template <class T> std::vector<T> vec() {
    auto n = pod<uint64_t>();
    std::vector<T> v(n);
    i.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    if (!i) fail("truncated trace file " + path);
    return v;
  }
  std::vector<TOp> ops() {
    std::vector<TOp> v(pod<uint64_t>());
    for (auto& op : v) {
      op.kind = TOpKind(pod<uint16_t>()); op.cat = pod<uint8_t>(); op.recompute = pod<uint8_t>();
      op.flops = pod<double>(); op.time = pod<double>(); op.a = vec<int64_t>(); op.f = vec<double>();
    }
    return v;
  }
};
}  // namespace

void Trace::save(const std::string& path) const {
  W w; w.o.open(path, std::ios::binary);
  if (!w.o) fail("cannot write trace file " + path);
  w.o.write(kMagic, 8);
  w.pod(uint32_t(1));
  w.pod(uint64_t(args.size())); for (auto& a : args) w.str(a);
  w.pod(config_hash); w.pod(int32_t(world)); w.pod(int32_t(worker)); w.pod(int32_t(backend));
  w.pod(arena_bytes); w.pod(opt_state);
  w.pod(uint64_t(comms.size())); for (auto& c : comms) { w.pod(int32_t(c.first)); w.pod(int32_t(c.second)); }
  w.pod(uint64_t(mem_peak.size())); for (auto& m : mem_peak) { w.str(m.first); w.pod(m.second); }
  w.pod(uint64_t(consts.size())); for (auto& c : consts) w.vec(c);
  w.ops(init); w.ops(step);
  if (!w.o) fail("error writing trace file " + path);
}
Trace Trace::load(const std::string& path) {
  Rd r; r.path = path; r.i.open(path, std::ios::binary);
  if (!r.i) fail("cannot open trace file " + path);
  char m[8]; r.i.read(m, 8);
  if (!r.i || std::memcmp(m, kMagic, 8) != 0) fail(path + " is not an xpsim trace");
  if (r.pod<uint32_t>() != 1) fail(path + ": unsupported trace version");
  Trace t;
  t.args.resize(r.pod<uint64_t>()); for (auto& a : t.args) a = r.str();
  t.config_hash = r.pod<uint64_t>(); t.world = r.pod<int32_t>(); t.worker = r.pod<int32_t>(); t.backend = r.pod<int32_t>();
  t.arena_bytes = r.pod<uint64_t>(); t.opt_state = r.pod<int64_t>();
  t.comms.resize(r.pod<uint64_t>()); for (auto& c : t.comms) { c.first = r.pod<int32_t>(); c.second = r.pod<int32_t>(); }
  t.mem_peak.resize(r.pod<uint64_t>()); for (auto& m : t.mem_peak) { m.first = r.str(); m.second = r.pod<uint64_t>(); }
  t.consts.resize(r.pod<uint64_t>()); for (auto& c : t.consts) c = r.vec<uint8_t>();
  t.init = r.ops(); t.step = r.ops();
  return t;
}

// =================================================================== replayer
Replayer::Replayer(const Trace& t, Mode m, const std::string& dot_prefix) : t_(t), mode_(m) {
  const double t0 = now_sec();
  if (mode_ == Mode::GRAPH && !dev::graphs_supported()) mode_ = Mode::EAGER;
  // constants -> device memory (outside the arena)
  for (auto& c : t_.consts) {
    if (dev::is_gpu()) {
      dev::Suspend s;
      void* p = dev::alloc(std::max<size_t>(c.size(), 16));
      dev::memcpy_h2d(p, c.data(), c.size());
      const_dev_.push_back(p);
    } else const_dev_.push_back(const_cast<uint8_t*>(c.data()));
  }
  // split the step into compute and communication segments
  for (size_t i = 0; i < t_.step.size();) {
    Seg s; s.begin = i; s.comm = is_comm_op(t_.step[i].kind);
    if (s.comm) ++i;
    else while (i < t_.step.size() && !is_comm_op(t_.step[i].kind)) ++i;
    s.end = i;
    if (!s.comm) {
      double tot = 0;
      for (size_t k = s.begin; k < s.end; ++k) {
        const TOp& op = t_.step[k];
        if (op.cat != 255 && op.time > 0) { s.cat_w[op.cat] += op.time; tot += op.time; }
        (op.recompute ? s.flops_h : s.flops_m) += op.flops;
      }
      s.flops_h += s.flops_m;
      if (tot > 0) for (auto& kv : s.cat_w) kv.second /= tot;
      else s.cat_w[int(Cat::ELEMWISE)] = 1.0;
    }
    segs_.push_back(std::move(s));
  }
  graphs_.assign(segs_.size(), nullptr);
  if (mode_ == Mode::GRAPH) {
    dev::sync();
    for (size_t k = 0; k < segs_.size(); ++k) {
      if (segs_[k].comm) continue;
      std::string dot = dot_prefix.empty() ? "" : dot_prefix + ".w" + std::to_string(my_worker()) + ".seg" + std::to_string(k) + ".dot";
      dev::graph_begin();
      for (size_t i = segs_[k].begin; i < segs_[k].end; ++i) exec(t_.step[i]);
      graphs_[k] = dev::graph_end(dot.c_str());
    }
  }
  build_s_ = now_sec() - t0;
}
Replayer::~Replayer() {
  for (void* g : graphs_) dev::graph_destroy(g);
  if (dev::is_gpu()) { dev::Suspend s; for (void* p : const_dev_) dev::free(p); }
}
int Replayer::segments() const { return int(segs_.size()); }
int Replayer::compute_segments() const { int n = 0; for (auto& s : segs_) n += !s.comm; return n; }
int Replayer::markers() const {
  int n = 0;
  for (auto& s : segs_) n += s.comm && t_.step[s.begin].kind == TOpKind::C_MARK;
  return n;
}

void* Replayer::P(int64_t e) const {
  if (e == -1) return nullptr;
  if (e >= 0) return dev::arena_base() + e;
  return const_dev_.at(size_t(-(e + 2)));
}

void Replayer::run_init() {
  for (const TOp& op : t_.init) exec(op);
  dev::sync();
}

void Replayer::run_step() {
  auto& m = metrics();
  for (size_t k = 0; k < segs_.size(); ++k) {
    const Seg& s = segs_[k];
    if (s.comm) { exec(t_.step[s.begin]); continue; }
    if (m.sync_timers && dev::is_gpu()) dev::sync();
    const double t0 = now_sec();
    if (graphs_[k]) dev::graph_launch(graphs_[k]);
    else for (size_t i = s.begin; i < s.end; ++i) exec(t_.step[i]);
    if (m.sync_timers && dev::is_gpu()) dev::sync();
    const double dt = now_sec() - t0;
    timeline_event(graphs_[k] ? "graph segment" : "eager segment", Cat(s.cat_w.begin()->first), 0, t0, dt);
    if (!m.recording) continue;
    for (auto& kv : s.cat_w) { m.cat[kv.first].time += dt * kv.second; m.cat[kv.first].calls += 1; }
    m.flops_model += s.flops_m; m.flops_hw += s.flops_h;
  }
}

void Replayer::exec(const TOp& op) {
  const auto& a = op.a;
  const auto& f = op.f;
  auto dt = [](int64_t v) { return DType(int(v)); };
  auto ori = [](int64_t v) { return v ? T : N; };
  switch (op.kind) {
    case TOpKind::GEMM:
      dev::gemm_strided_batched(ori(a[0]), ori(a[1]), a[2], a[3], a[4], float(f[0]), P(a[5]), dt(a[6]), a[7], a[8], P(a[9]),
                                dt(a[10]), a[11], a[12], float(f[1]), P(a[13]), dt(a[14]), a[15], a[16], a[17]);
      break;
    case TOpKind::MEMCPY: dev::memcpy_d2d(P(a[0]), P(a[1]), size_t(a[2])); break;
    case TOpKind::UPLOAD: dev::memcpy_d2d(P(a[0]), P(a[1]), size_t(a[2])); break;  // constant -> arena
    case TOpKind::MEMSET: dev::memset_zero(P(a[0]), size_t(a[1])); break;
    case TOpKind::COPY2D: dev::copy2d(P(a[0]), a[1], P(a[2]), a[3], a[4], a[5], a[6]); break;
    case TOpKind::FILL: dev::fill_random(P(a[0]), size_t(a[1]), dt(a[2]), uint64_t(a[3]), float(f[0])); break;
    case TOpKind::CAST: dev::cast(P(a[0]), dt(a[1]), P(a[2]), dt(a[3]), size_t(a[4])); break;
    case TOpKind::AXPY: dev::axpy(size_t(a[0]), float(f[0]), P(a[1]), P(a[2]), dt(a[3])); break;
    case TOpKind::SCALE: dev::scale(size_t(a[0]), float(f[0]), P(a[1]), dt(a[2])); break;
    case TOpKind::GELU_F: dev::gelu_fwd(size_t(a[0]), P(a[1]), P(a[2]), dt(a[3])); break;
    case TOpKind::GELU_B: dev::gelu_bwd(size_t(a[0]), P(a[1]), P(a[2]), P(a[3]), dt(a[4])); break;
    case TOpKind::SOFTMAX: dev::softmax_cols(P(a[0]), a[1], a[2], a[3], float(f[0]), a[4] != 0, a[5], dt(a[6])); break;
    case TOpKind::SOFTMAX_B: dev::softmax_bwd_cols(P(a[0]), P(a[1]), a[2], a[3], a[4], float(f[0]), dt(a[5])); break;
    case TOpKind::COPY_ROWS:
      dev::copy_rows_d(P(a[0]), a[1], a[2] != 0, static_cast<const int*>(P(a[3])), P(a[4]), a[5], a[6] != 0,
                       static_cast<const int*>(P(a[7])), a[8], a[9], static_cast<const float*>(P(a[10])), a[11] != 0, dt(a[12]));
      break;
    case TOpKind::ADAMW:
      dev::adamw_d(P(a[0]), dt(a[1]), P(a[2]), dt(a[3]), P(a[4]), P(a[5]), size_t(a[6]), float(f[0]), float(f[1]),
                   float(f[2]), float(f[3]), float(f[4]), static_cast<const dev::OptState*>(P(a[7])));
      break;
    case TOpKind::SUMSQ_ACC: dev::sumsq_acc(P(a[0]), size_t(a[1]), dt(a[2]), static_cast<double*>(P(a[3])), f[0]); break;
    case TOpKind::OPT_UPDATE: dev::opt_update(static_cast<dev::OptState*>(P(a[0])), float(f[0]), float(f[1])); break;
    case TOpKind::COL_MAX: dev::col_max_d(P(a[0]), a[1], a[2], a[3], dt(a[4]), static_cast<float*>(P(a[5]))); break;
    case TOpKind::COL_SUMEXP:
      dev::col_sumexp_d(P(a[0]), a[1], a[2], a[3], dt(a[4]), static_cast<const float*>(P(a[5])), static_cast<float*>(P(a[6])));
      break;
    case TOpKind::XENT:
      dev::xent_grad_d(P(a[0]), a[1], a[2], a[3], dt(a[4]), static_cast<const float*>(P(a[5])), static_cast<const float*>(P(a[6])),
                       static_cast<const int*>(P(a[7])), float(f[0]));
      break;
    case TOpKind::MAX_BLOCKS: dev::max_blocks(static_cast<float*>(P(a[0])), static_cast<const float*>(P(a[1])), int(a[2]), size_t(a[3])); break;

    // ------------------------------------------------ communication
    case TOpKind::C_POST: {
      Comm& c = Comm::from_index(int(a[0]));
      const int type = int(a[1]), p = int(a[10]);
      const Cat cat = Cat(int(a[8]));
      const DType d = dt(a[5]);
      const size_t n = size_t(a[4]);
      Request r;
      switch (type) {
        case 0: r = c.iallreduce(P(a[3]), n, d, cat); break;
        case 1: r = c.iallgather(P(a[2]), P(a[3]), n, d, cat); break;
        case 2: r = c.ireduce_scatter(P(a[2]), P(a[3]), n, d, cat); break;
        case 3: {
          std::vector<size_t> sc(p), sd(p), rc(p), rd(p);
          for (int j = 0; j < p; ++j) {
            sc[j] = size_t(a[11 + j]); sd[j] = size_t(a[11 + p + j]); rc[j] = size_t(a[11 + 2 * p + j]); rd[j] = size_t(a[11 + 3 * p + j]);
          }
          r = c.ialltoallv(P(a[2]), sc, sd, P(a[3]), rc, rd, d, cat);
          break;
        }
        case 4: r = c.isend(P(a[2]), n, int(a[6]), int(a[7]), cat); break;
        case 5: r = c.irecv(P(a[3]), n, int(a[6]), int(a[7]), cat); break;
        case 6: r = c.ibarrier(cat); break;
        default: fail("trace: unknown communication op");
      }
      reqs_[int(a[9])] = {int(a[0]), r};
      break;
    }
    case TOpKind::C_WAIT: {
      auto it = reqs_.find(int(a[1]));
      if (it == reqs_.end()) fail("trace: wait on an unknown request");
      Comm::from_index(it->second.first).wait(it->second.second);
      reqs_.erase(it);
      break;
    }
    case TOpKind::C_HOST: {
      Comm& c = Comm::from_index(int(a[0]));
      const int which = int(a[1]), n = int(a[2]);
      const Cat cat = Cat(int(a[3]));
      const auto& blob = t_.consts.at(size_t(-(a[4] + 2)));
      switch (which) {
        case 0: { std::vector<double> b(n); std::memcpy(b.data(), blob.data(), n * 8); c.allreduce_host(b.data(), n, cat); break; }
        case 1: { std::vector<double> b(n), o(size_t(n) * c.size()); std::memcpy(b.data(), blob.data(), n * 8);
                  c.allgather_host(b.data(), o.data(), n, cat); break; }
        case 2: { std::vector<float> b(n); std::memcpy(b.data(), blob.data(), n * 4); c.allreduce_max_host(b.data(), n, cat); break; }
        case 3: { std::vector<long long> b(size_t(n) * c.size()), o(b.size()); std::memcpy(b.data(), blob.data(), b.size() * 8);
                  c.alltoall_host(b.data(), o.data(), n, cat); break; }
        default: fail("trace: unknown host collective");
      }
      break;
    }
    case TOpKind::C_MARK: {  // re-emit pipeline spans on the replay timeline
      if (a[1]) { mark_t0_.push_back(now_sec()); break; }
      if (mark_t0_.empty()) break;
      const double t0 = mark_t0_.back(); mark_t0_.pop_back();
      auto& m = metrics();
      if (m.timeline.on) m.timeline.ev.push_back({span_name(int(a[0])), uint8_t(NCAT), 0, t0, now_sec() - t0, int(a[2]), int(a[3])});
      break;
    }
    default: fail("trace: unknown op kind " + std::to_string(int(op.kind)));
  }
}

}  // namespace xpsim
