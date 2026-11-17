// Backend dispatch + allocation tracking.
#include <chrono>
#include <unordered_map>
#include "device_impl.hpp"

namespace xpsim::dev {

namespace {
struct State {
  Backend backend = Backend::CPU;
  int device = 0;
  const char* tag = "workspace";
  std::map<std::string, MemStats> stats;
  std::unordered_map<void*, std::pair<size_t, std::string>> live;
  size_t current = 0, peak = 0;
};
thread_local State S;
}  // namespace

#define DISPATCH(call) (S.backend == Backend::GPU ? gpu::call : cpu::call)

void init(Backend b, int device_id) {
  S.backend = b; S.device = device_id;
  if (b == Backend::GPU) {
    if (!cuda_compiled()) fail("GPU backend requested but xpsim was built without CUDA (make CUDA=1)");
    gpu::init(device_id);
  }
}
void finalize() { if (S.backend == Backend::GPU) gpu::finalize(); }
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

void* alloc(size_t bytes) {
  if (bytes == 0) bytes = 16;
  void* p = DISPATCH(alloc(bytes));
  if (!p) fail("device allocation of " + std::to_string(bytes) + " bytes failed");
  S.live[p] = {bytes, S.tag};
  auto& st = S.stats[S.tag];
  st.current += bytes; st.peak = std::max(st.peak, st.current);
  S.current += bytes; S.peak = std::max(S.peak, S.current);
  return p;
}
void free(void* p) {
  if (!p) return;
  auto it = S.live.find(p);
  if (it != S.live.end()) {
    S.stats[it->second.second].current -= it->second.first;
    S.current -= it->second.first;
    S.live.erase(it);
  }
  DISPATCH(free(p));
}
void* host_alloc(size_t bytes) { return DISPATCH(host_alloc(bytes)); }
void host_free(void* p) { if (p) DISPATCH(host_free(p)); }
const std::map<std::string, MemStats>& mem_stats() { return S.stats; }
size_t mem_current() { return S.current; }
size_t mem_peak() { return S.peak; }
void mem_reset_peak() { S.peak = S.current; for (auto& kv : S.stats) kv.second.peak = kv.second.current; }
MemScope::MemScope(const char* tag) : prev(S.tag) { S.tag = tag; }
MemScope::~MemScope() { S.tag = prev; }

void memcpy_d2d(void* d, const void* s, size_t b) { if (b) DISPATCH(memcpy_any(d, s, b, CPY_D2D)); }
void memcpy_h2d(void* d, const void* s, size_t b) { if (b) DISPATCH(memcpy_any(d, s, b, CPY_H2D)); }
void memcpy_d2h(void* d, const void* s, size_t b) { if (b) DISPATCH(memcpy_any(d, s, b, CPY_D2H)); }
void memset_zero(void* p, size_t b) { if (b) DISPATCH(memset_zero(p, b)); }
void copy2d(void* d, size_t ldd, const void* s, size_t lds, size_t r, size_t c, size_t es) {
  if (r && c) DISPATCH(copy2d(d, ldd, s, lds, r, c, es));
}
void sync() { DISPATCH(sync()); }

void gemm(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha, const void* A, DType dtA,
          Int lda, const void* B, DType dtB, Int ldb, float beta, void* C, DType dtC, Int ldc) {
  if (m == 0 || n == 0) return;
  DISPATCH(gemm_strided_batched(oA, oB, m, n, k, alpha, A, dtA, lda, 0, B, dtB, ldb, 0, beta, C, dtC, ldc, 0, 1));
}
void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha,
                          const void* A, DType dtA, Int lda, Int sA, const void* B, DType dtB, Int ldb,
                          Int sB, float beta, void* C, DType dtC, Int ldc, Int sC, Int batch) {
  if (m == 0 || n == 0 || batch == 0) return;
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

void fill_random(void* p, size_t n, DType dt, uint64_t seed, float sc) { if (n) DISPATCH(fill_random(p, n, dt, seed, sc)); }
void cast(void* d, DType dtd, const void* s, DType dts, size_t n) { if (n) DISPATCH(cast(d, dtd, s, dts, n)); }
void axpy(size_t n, float a, const void* x, void* y, DType dt) { if (n) DISPATCH(axpy(n, a, x, y, dt)); }
void scale(size_t n, float a, void* x, DType dt) { if (n) DISPATCH(scale(n, a, x, dt)); }
void gelu_fwd(size_t n, const void* i, void* o, DType dt) { if (n) DISPATCH(gelu_fwd(n, i, o, dt)); }
void gelu_bwd(size_t n, const void* p, const void* d, void* di, DType dt) { if (n) DISPATCH(gelu_bwd(n, p, d, di, dt)); }
void softmax_cols(void* S_, Int r, Int c, Int ld, float sc, bool causal, Int qo, DType dt) {
  if (r && c) DISPATCH(softmax_cols(S_, r, c, ld, sc, causal, qo, dt));
}
void softmax_bwd_cols(const void* P, void* dP, Int r, Int c, Int ld, float sc, DType dt) {
  if (r && c) DISPATCH(softmax_bwd_cols(P, dP, r, c, ld, sc, dt));
}
void copy_rows(const void* src, Int lds, bool srm, const int* srow, void* dst, Int ldd, bool drm,
               const int* drow, Int n, Int width, const float* w, bool acc, DType dt) {
  if (n && width) DISPATCH(copy_rows(src, lds, srm, srow, dst, ldd, drm, drow, n, width, w, acc, dt));
}
void adamw(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1, float b2,
           float eps, float wd, int step, float gs) {
  if (n) DISPATCH(adamw(p, dtp, g, dtg, m, v, n, lr, b1, b2, eps, wd, step, gs));
}
void set_tf32(bool on) { DISPATCH(set_tf32(on)); }
const char* gemm_engine() { return DISPATCH(gemm_engine()); }
double sumsq(const void* p, size_t n, DType dt) { return n ? DISPATCH(sumsq(p, n, dt)) : 0.0; }
size_t count_nonfinite(const void* p, size_t n, DType dt) { return n ? DISPATCH(count_nonfinite(p, n, dt)) : 0; }
void col_max(const void* L, Int r, Int c, Int ld, DType dt, float* o) { DISPATCH(col_max(L, r, c, ld, dt, o)); }
void col_sumexp(const void* L, Int r, Int c, Int ld, DType dt, const float* mx, float* o) {
  DISPATCH(col_sumexp(L, r, c, ld, dt, mx, o));
}
void xent_grad(void* L, Int r, Int c, Int ld, DType dt, const float* mx, const float* sm, const int* t, float sc) {
  DISPATCH(xent_grad(L, r, c, ld, dt, mx, sm, t, sc));
}
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
#undef STUB
}  // namespace xpsim::dev::gpu
#endif
