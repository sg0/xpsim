// Backend dispatch, allocation tracking, the replay arena and the trace
// recording hooks of every device op.
#include <chrono>
#include <unordered_map>
#include "device_impl.hpp"
#include "xpsim/trace.hpp"

namespace xpsim::dev {

namespace {
struct Live { size_t bytes; std::string tag; bool arena; };
struct ArenaState {
  char* base = nullptr;
  size_t size = 0, hwm = 0;
  std::map<size_t, size_t> free_;  // offset -> length
};
struct State {
  Backend backend = Backend::CPU;
  int device = 0;
  const char* tag = "workspace";
  std::map<std::string, MemStats> stats;
  std::unordered_map<void*, Live> live;
  size_t current = 0, peak = 0;
  ArenaState arena;
  int suspend = 0;
};
thread_local State S;
constexpr size_t kAlign = 256;
}  // namespace

#define DISPATCH(call) (S.backend == Backend::GPU ? gpu::call : cpu::call)
#define TP(p) trace::ptr(p)
#define REC(...) do { if (trace::active()) trace::rec(__VA_ARGS__); } while (0)
static int64_t I(DType d) { return int64_t(d); }
static int64_t I(Orientation o) { return o == N ? 0 : 1; }

void init(Backend b, int device_id) {
  S.backend = b; S.device = device_id;
  if (b == Backend::GPU) {
    if (!cuda_compiled()) fail("GPU backend requested but xpsim was built without CUDA (make CUDA=1)");
    gpu::init(device_id);
  }
}
void finalize() { arena_end(); if (S.backend == Backend::GPU) gpu::finalize(); }
Backend backend() { return S.backend; }
bool is_gpu() { return S.backend == Backend::GPU; }
int device_id() { return S.device; }
bool cuda_compiled() {
#ifdef XPSIM_USE_CUDA
  return true;
#else
  return false;
#endif
}
int gpu_count() { return cuda_compiled() ? gpu::count() : 0; }
std::string device_name() { return is_gpu() ? gpu::name() : std::string("cpu"); }

// ---------------------------------------------------------------- arena
void arena_begin(size_t bytes) {
  if (S.arena.base) fail("arena already active");
  bytes = (bytes + kAlign - 1) / kAlign * kAlign;
  S.arena.base = static_cast<char*>(DISPATCH(arena_reserve(bytes)));
  if (!S.arena.base) fail("could not reserve a " + std::to_string(bytes >> 20) + " MiB arena (--arena-mb)");
  S.arena.size = bytes; S.arena.hwm = 0;
  S.arena.free_.clear(); S.arena.free_[0] = bytes;
}
void arena_end() {
  if (!S.arena.base) return;
  for (auto it = S.live.begin(); it != S.live.end();) {  // drop bookkeeping of leftovers
    if (it->second.arena) { S.stats[it->second.tag].current -= it->second.bytes; S.current -= it->second.bytes; it = S.live.erase(it); }
    else ++it;
  }
  DISPATCH(arena_release(S.arena.base, S.arena.size));
  S.arena = ArenaState{};
}
bool arena_active() { return S.arena.base != nullptr; }
char* arena_base() { return S.arena.base; }
size_t arena_size() { return S.arena.size; }
size_t arena_high_water() { return S.arena.hwm; }
long long arena_offset(const void* p) {
  auto* c = static_cast<const char*>(p);
  if (!S.arena.base || c < S.arena.base || c >= S.arena.base + S.arena.size) return -1;
  return (long long)(c - S.arena.base);
}
Suspend::Suspend() { ++S.suspend; }
Suspend::~Suspend() { --S.suspend; }
bool suspended() { return S.suspend > 0; }

static void* arena_alloc(size_t bytes) {
  bytes = (bytes + kAlign - 1) / kAlign * kAlign;
  for (auto it = S.arena.free_.begin(); it != S.arena.free_.end(); ++it)
    if (it->second >= bytes) {
      size_t off = it->first, len = it->second;
      S.arena.free_.erase(it);
      if (len > bytes) S.arena.free_[off + bytes] = len - bytes;
      S.arena.hwm = std::max(S.arena.hwm, off + bytes);
      return S.arena.base + off;
    }
  fail("arena exhausted (" + std::to_string(S.arena.size >> 20) + " MiB); raise --arena-mb");
}
static void arena_free(void* p, size_t bytes) {
  bytes = (bytes + kAlign - 1) / kAlign * kAlign;
  size_t off = size_t(static_cast<char*>(p) - S.arena.base);
  auto it = S.arena.free_.emplace(off, bytes).first;
  auto nx = std::next(it);
  if (nx != S.arena.free_.end() && it->first + it->second == nx->first) { it->second += nx->second; S.arena.free_.erase(nx); }
  if (it != S.arena.free_.begin()) {
    auto pv = std::prev(it);
    if (pv->first + pv->second == it->first) { pv->second += it->second; S.arena.free_.erase(it); }
  }
}

// ---------------------------------------------------------------- memory
void* alloc(size_t bytes) {
  if (bytes == 0) bytes = 16;
  const bool in_arena = S.arena.base && S.suspend == 0;
  void* p = in_arena ? arena_alloc(bytes) : DISPATCH(alloc(bytes));
  if (!p) fail("device allocation of " + std::to_string(bytes) + " bytes failed");
  S.live[p] = {bytes, S.tag, in_arena};
  auto& st = S.stats[S.tag];
  st.current += bytes; st.peak = std::max(st.peak, st.current);
  S.current += bytes; S.peak = std::max(S.peak, S.current);
  return p;
}
void free(void* p) {
  if (!p) return;
  auto it = S.live.find(p);
  if (it != S.live.end()) {
    S.stats[it->second.tag].current -= it->second.bytes;
    S.current -= it->second.bytes;
    const bool a = it->second.arena; const size_t b = it->second.bytes;
    S.live.erase(it);
    if (a) { arena_free(p, b); return; }
  }
  DISPATCH(free(p));
}
void* host_alloc(size_t bytes) { return DISPATCH(host_alloc(bytes)); }
void host_free(void* p) { if (p) DISPATCH(host_free(p)); }
const std::map<std::string, MemStats>& mem_stats() { return S.stats; }
size_t mem_current() { return S.current; }
size_t mem_peak() { return S.peak; }
void mem_reset_peak() { S.peak = S.current; for (auto& kv : S.stats) kv.second.peak = kv.second.current; }
void mem_note_peak(const std::string& tag, size_t bytes) {
  auto& st = S.stats[tag];
  st.peak = std::max(st.peak, bytes);
  if (tag == "__total__") { S.stats.erase(tag); S.peak = std::max(S.peak, bytes); }
}
MemScope::MemScope(const char* tag) : prev(S.tag) { S.tag = tag; }
MemScope::~MemScope() { S.tag = prev; }
size_t free_memory() { return DISPATCH(free_memory()); }

// ---------------------------------------------------------------- data movement
void memcpy_d2d(void* d, const void* s, size_t b) {
  if (!b) return;
  REC(TOpKind::MEMCPY, {TP(d), TP(s), int64_t(b)});
  DISPATCH(memcpy_any(d, s, b, CPY_D2D));
}
void memcpy_h2d(void* d, const void* s, size_t b) {
  if (!b) return;
  if (trace::active()) trace::rec(TOpKind::UPLOAD, {TP(d), trace::cst(s, b), int64_t(b)});
  DISPATCH(memcpy_any(d, s, b, CPY_H2D));
}
void memcpy_d2h(void* d, const void* s, size_t b) {
  if (!b) return;
  if (trace::active()) fail("device->host copy inside a recorded step (data-dependent host logic cannot be replayed)");
  DISPATCH(memcpy_any(d, s, b, CPY_D2H));
}
void memset_zero(void* p, size_t b) {
  if (!b) return;
  REC(TOpKind::MEMSET, {TP(p), int64_t(b)});
  DISPATCH(memset_zero(p, b));
}
void copy2d(void* d, size_t ldd, const void* s, size_t lds, size_t r, size_t c, size_t es) {
  if (!(r && c)) return;
  REC(TOpKind::COPY2D, {TP(d), int64_t(ldd), TP(s), int64_t(lds), int64_t(r), int64_t(c), int64_t(es)});
  DISPATCH(copy2d(d, ldd, s, lds, r, c, es));
}
void sync() { DISPATCH(sync()); }

// ---------------------------------------------------------------- GEMM
void gemm(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha, const void* A, DType dtA,
          Int lda, const void* B, DType dtB, Int ldb, float beta, void* C, DType dtC, Int ldc) {
  gemm_strided_batched(oA, oB, m, n, k, alpha, A, dtA, lda, 0, B, dtB, ldb, 0, beta, C, dtC, ldc, 0, 1);
}
void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha,
                          const void* A, DType dtA, Int lda, Int sA, const void* B, DType dtB, Int ldb,
                          Int sB, float beta, void* C, DType dtC, Int ldc, Int sC, Int batch) {
  if (m == 0 || n == 0 || batch == 0) return;
  REC(TOpKind::GEMM, {I(oA), I(oB), m, n, k, TP(A), I(dtA), lda, sA, TP(B), I(dtB), ldb, sB, TP(C), I(dtC), ldc, sC, batch},
      {alpha, beta});
  DISPATCH(gemm_strided_batched(oA, oB, m, n, k, alpha, A, dtA, lda, sA, B, dtB, ldb, sB, beta, C, dtC, ldc, sC, batch));
}

double gemm_peak_tflops(DType dt, Int n) {
  size_t es = dtype_size(dt);
  void* A = alloc(n * n * es); void* B = alloc(n * n * es); void* C = alloc(n * n * es);
  fill_random(A, n * n, dt, 1, 0.1f); fill_random(B, n * n, dt, 2, 0.1f);
  gemm(N, N, n, n, n, 1.f, A, dt, n, B, dt, n, 0.f, C, dt, n); sync();  // warm-up
  int reps = is_gpu() ? 10 : 2;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) gemm(N, N, n, n, n, 1.f, A, dt, n, B, dt, n, 0.f, C, dt, n);
  sync();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  free(A); free(B); free(C);
  return 2.0 * double(n) * n * n * reps / s / 1e12;
}

// ---------------------------------------------------------------- elementwise
void fill_random(void* p, size_t n, DType dt, uint64_t seed, float sc) {
  if (!n) return;
  REC(TOpKind::FILL, {TP(p), int64_t(n), I(dt), int64_t(seed)}, {sc});
  DISPATCH(fill_random(p, n, dt, seed, sc));
}
void cast(void* d, DType dtd, const void* s, DType dts, size_t n) {
  if (!n) return;
  REC(TOpKind::CAST, {TP(d), I(dtd), TP(s), I(dts), int64_t(n)});
  DISPATCH(cast(d, dtd, s, dts, n));
}
void axpy(size_t n, float a, const void* x, void* y, DType dt) {
  if (!n) return;
  REC(TOpKind::AXPY, {int64_t(n), TP(x), TP(y), I(dt)}, {a});
  DISPATCH(axpy(n, a, x, y, dt));
}
void scale(size_t n, float a, void* x, DType dt) {
  if (!n) return;
  REC(TOpKind::SCALE, {int64_t(n), TP(x), I(dt)}, {a});
  DISPATCH(scale(n, a, x, dt));
}
void gelu_fwd(size_t n, const void* i, void* o, DType dt) {
  if (!n) return;
  REC(TOpKind::GELU_F, {int64_t(n), TP(i), TP(o), I(dt)});
  DISPATCH(gelu_fwd(n, i, o, dt));
}
void gelu_bwd(size_t n, const void* p, const void* d, void* di, DType dt) {
  if (!n) return;
  REC(TOpKind::GELU_B, {int64_t(n), TP(p), TP(d), TP(di), I(dt)});
  DISPATCH(gelu_bwd(n, p, d, di, dt));
}
void softmax_cols(void* S_, Int r, Int c, Int ld, float sc, bool causal, Int qo, DType dt) {
  if (!(r && c)) return;
  REC(TOpKind::SOFTMAX, {TP(S_), r, c, ld, int64_t(causal), qo, I(dt)}, {sc});
  DISPATCH(softmax_cols(S_, r, c, ld, sc, causal, qo, dt));
}
void softmax_bwd_cols(const void* P, void* dP, Int r, Int c, Int ld, float sc, DType dt) {
  if (!(r && c)) return;
  REC(TOpKind::SOFTMAX_B, {TP(P), TP(dP), r, c, ld, I(dt)}, {sc});
  DISPATCH(softmax_bwd_cols(P, dP, r, c, ld, sc, dt));
}
void copy_rows(const void* src, Int lds, bool srm, const int* srow, void* dst, Int ldd, bool drm,
               const int* drow, Int n, Int width, const float* w, bool acc, DType dt) {
  if (!(n && width)) return;
  if (trace::active())  // host index arrays become trace constants
    trace::rec(TOpKind::COPY_ROWS, {TP(src), lds, int64_t(srm), srow ? trace::cst(srow, n * 4) : -1, TP(dst), ldd,
                                    int64_t(drm), drow ? trace::cst(drow, n * 4) : -1, n, width,
                                    w ? trace::cst(w, n * 4) : -1, int64_t(acc), I(dt)});
  DISPATCH(copy_rows(src, lds, srm, srow, dst, ldd, drm, drow, n, width, w, acc, dt));
}
void copy_rows_d(const void* src, Int lds, bool srm, const int* srow, void* dst, Int ldd, bool drm,
                 const int* drow, Int n, Int width, const float* w, bool acc, DType dt) {
  if (!(n && width)) return;
  REC(TOpKind::COPY_ROWS, {TP(src), lds, int64_t(srm), TP(srow), TP(dst), ldd, int64_t(drm), TP(drow), n, width, TP(w),
                           int64_t(acc), I(dt)});
  DISPATCH(copy_rows_d(src, lds, srm, srow, dst, ldd, drm, drow, n, width, w, acc, dt));
}
void adamw(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1, float b2,
           float eps, float wd, int step, float gs) {
  if (trace::active()) fail("adamw with host step/scale inside a recorded step; use adamw_d");
  if (n) DISPATCH(adamw(p, dtp, g, dtg, m, v, n, lr, b1, b2, eps, wd, step, gs));
}
void adamw_d(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1, float b2,
             float eps, float wd, const OptState* st) {
  if (!n) return;
  REC(TOpKind::ADAMW, {TP(p), I(dtp), TP(g), I(dtg), TP(m), TP(v), int64_t(n), TP(st)}, {lr, b1, b2, eps, wd});
  DISPATCH(adamw_d(p, dtp, g, dtg, m, v, n, lr, b1, b2, eps, wd, st));
}
void set_tf32(bool on) { DISPATCH(set_tf32(on)); }
const char* gemm_engine() { return DISPATCH(gemm_engine()); }
static void no_trace(const char* what) {
  if (trace::active()) fail(std::string(what) + " returns data to the host inside a recorded step; use the _d variant");
}
double sumsq(const void* p, size_t n, DType dt) { no_trace("sumsq"); return n ? DISPATCH(sumsq(p, n, dt)) : 0.0; }
size_t count_nonfinite(const void* p, size_t n, DType dt) {
  no_trace("count_nonfinite"); return n ? DISPATCH(count_nonfinite(p, n, dt)) : 0;
}
void col_max(const void* L, Int r, Int c, Int ld, DType dt, float* o) { no_trace("col_max"); DISPATCH(col_max(L, r, c, ld, dt, o)); }
void col_sumexp(const void* L, Int r, Int c, Int ld, DType dt, const float* mx, float* o) {
  no_trace("col_sumexp"); DISPATCH(col_sumexp(L, r, c, ld, dt, mx, o));
}
void xent_grad(void* L, Int r, Int c, Int ld, DType dt, const float* mx, const float* sm, const int* t, float sc) {
  no_trace("xent_grad"); DISPATCH(xent_grad(L, r, c, ld, dt, mx, sm, t, sc));
}
void col_max_d(const void* L, Int r, Int c, Int ld, DType dt, float* o) {
  REC(TOpKind::COL_MAX, {TP(L), r, c, ld, I(dt), TP(o)});
  DISPATCH(col_max_d(L, r, c, ld, dt, o));
}
void col_sumexp_d(const void* L, Int r, Int c, Int ld, DType dt, const float* mx, float* o) {
  REC(TOpKind::COL_SUMEXP, {TP(L), r, c, ld, I(dt), TP(mx), TP(o)});
  DISPATCH(col_sumexp_d(L, r, c, ld, dt, mx, o));
}
void xent_grad_d(void* L, Int r, Int c, Int ld, DType dt, const float* mx, const float* sm, const int* t, float sc) {
  REC(TOpKind::XENT, {TP(L), r, c, ld, I(dt), TP(mx), TP(sm), TP(t)}, {sc});
  DISPATCH(xent_grad_d(L, r, c, ld, dt, mx, sm, t, sc));
}
void max_blocks(float* d, const float* s, int p, size_t n) {
  if (!n) return;
  REC(TOpKind::MAX_BLOCKS, {TP(d), TP(s), p, int64_t(n)});
  DISPATCH(max_blocks(d, s, p, n));
}
void sumsq_acc(const void* x, size_t n, DType dt, double* acc, double w) {
  if (!n) return;
  REC(TOpKind::SUMSQ_ACC, {TP(x), int64_t(n), I(dt), TP(acc)}, {w});
  DISPATCH(sumsq_acc(x, n, dt, acc, w));
}
void opt_update(OptState* st, float ls, float clip) {
  REC(TOpKind::OPT_UPDATE, {TP(st)}, {ls, clip});
  DISPATCH(opt_update(st, ls, clip));
}

// ---------------------------------------------------------------- graphs
bool graphs_supported() { return S.backend == Backend::GPU; }
void graph_begin() { DISPATCH(graph_begin()); }
void* graph_end(const char* dot) { return DISPATCH(graph_end(dot)); }
void graph_launch(void* e) { DISPATCH(graph_launch(e)); }
void graph_destroy(void* e) { if (e) DISPATCH(graph_destroy(e)); }

void xfer_copy(int device, void* d, const void* s, size_t b) { if (b) DISPATCH(xfer_copy(device, d, s, b)); }
void xfer_reduce(int device, void* d, const void* const* s, int k, size_t n, DType dt, bool acc) {
  if (n) DISPATCH(xfer_reduce(device, d, s, k, n, dt, acc));
}
void xfer_sync(int device) { DISPATCH(xfer_sync(device)); }

}  // namespace xpsim::dev

#ifndef XPSIM_USE_CUDA
// Stubs so that CPU-only builds link; init() refuses the GPU backend first.
namespace xpsim::dev::gpu {
#define STUB { fail("CUDA support not compiled in"); }
void init(int) STUB
void finalize() {}
int count() { return 0; }
std::string name() { return "none"; }
void* alloc(size_t) STUB
void free(void*) STUB
void* host_alloc(size_t) STUB
void host_free(void*) STUB
void memcpy_any(void*, const void*, size_t, int) STUB
void memset_zero(void*, size_t) STUB
void copy2d(void*, size_t, const void*, size_t, size_t, size_t, size_t) STUB
void sync() STUB
void gemm_strided_batched(Orientation, Orientation, Int, Int, Int, float, const void*, DType, Int, Int,
                          const void*, DType, Int, Int, float, void*, DType, Int, Int, Int) STUB
void fill_random(void*, size_t, DType, uint64_t, float) STUB
void cast(void*, DType, const void*, DType, size_t) STUB
void axpy(size_t, float, const void*, void*, DType) STUB
void scale(size_t, float, void*, DType) STUB
void gelu_fwd(size_t, const void*, void*, DType) STUB
void gelu_bwd(size_t, const void*, const void*, void*, DType) STUB
void softmax_cols(void*, Int, Int, Int, float, bool, Int, DType) STUB
void softmax_bwd_cols(const void*, void*, Int, Int, Int, float, DType) STUB
void copy_rows(const void*, Int, bool, const int*, void*, Int, bool, const int*, Int, Int, const float*, bool, DType) STUB
void adamw(void*, DType, const void*, DType, void*, void*, size_t, float, float, float, float, float, int, float) STUB
void set_tf32(bool) {}
const char* gemm_engine() { return "none"; }
double sumsq(const void*, size_t, DType) STUB
size_t count_nonfinite(const void*, size_t, DType) STUB
void col_max(const void*, Int, Int, Int, DType, float*) STUB
void col_sumexp(const void*, Int, Int, Int, DType, const float*, float*) STUB
void xent_grad(void*, Int, Int, Int, DType, const float*, const float*, const int*, float) STUB
void xfer_copy(int, void*, const void*, size_t) STUB
void xfer_reduce(int, void*, const void* const*, int, size_t, DType, bool) STUB
void xfer_sync(int) STUB
void copy_rows_d(const void*, Int, bool, const int*, void*, Int, bool, const int*, Int, Int, const float*, bool, DType) STUB
void col_max_d(const void*, Int, Int, Int, DType, float*) STUB
void col_sumexp_d(const void*, Int, Int, Int, DType, const float*, float*) STUB
void xent_grad_d(void*, Int, Int, Int, DType, const float*, const float*, const int*, float) STUB
void max_blocks(float*, const float*, int, size_t) STUB
void sumsq_acc(const void*, size_t, DType, double*, double) STUB
void opt_update(OptState*, float, float) STUB
void adamw_d(void*, DType, const void*, DType, void*, void*, size_t, float, float, float, float, float, const OptState*) STUB
void graph_begin() STUB
void* graph_end(const char*) STUB
void graph_launch(void*) STUB
void graph_destroy(void*) STUB
size_t free_memory() STUB
void* arena_reserve(size_t) STUB
void arena_release(void*, size_t) STUB
#undef STUB
}  // namespace xpsim::dev::gpu
#endif
