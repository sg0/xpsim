// CUDA backend: cuBLAS GemmEx (fp32 accumulate, bf16/fp16 tensor cores) and
// simple custom kernels. One stream + cuBLAS handle per worker thread; the
// transport engine uses a separate stream per device (xfer_* functions).
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>
#include "device_impl.hpp"

namespace xpsim::dev::gpu {
extern bool g_tf32;

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) fail(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) fail(std::string(#x) + " failed, status " + std::to_string(int(s_))); } while (0)

namespace {
struct Worker {
  int device = -1;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  void* iscratch = nullptr; size_t iscratch_bytes = 0;  // uploaded index/weight arrays
  float* fscratch = nullptr; size_t fscratch_n = 0;
};
thread_local Worker W;

std::mutex g_eng_mu;
std::vector<cudaStream_t> g_eng_streams;  // per device, used by the progress engine

cudaStream_t engine_stream(int dev) {
  std::lock_guard<std::mutex> lk(g_eng_mu);
  if ((int)g_eng_streams.size() <= dev) g_eng_streams.resize(dev + 1, nullptr);
  if (!g_eng_streams[dev]) {
    CUDA_OK(cudaSetDevice(dev));
    CUDA_OK(cudaStreamCreateWithFlags(&g_eng_streams[dev], cudaStreamNonBlocking));
  }
  return g_eng_streams[dev];
}

void* scratch(size_t bytes) {
  if (bytes > W.iscratch_bytes) {
    if (W.iscratch) { CUDA_OK(cudaStreamSynchronize(W.stream)); cudaFree(W.iscratch); }
    W.iscratch_bytes = std::max<size_t>(bytes, 1 << 20);
    CUDA_OK(cudaMalloc(&W.iscratch, W.iscratch_bytes));
  }
  return W.iscratch;
}
float* fscratch(size_t n) {
  if (n > W.fscratch_n) {
    if (W.fscratch) { CUDA_OK(cudaStreamSynchronize(W.stream)); cudaFree(W.fscratch); }
    W.fscratch_n = std::max<size_t>(n, 1 << 16);
    CUDA_OK(cudaMalloc(&W.fscratch, W.fscratch_n * sizeof(float)));
  }
  return W.fscratch;
}

__device__ __forceinline__ float tof(float x) { return x; }
__device__ __forceinline__ float tof(double x) { return float(x); }
__device__ __forceinline__ float tof(__half x) { return __half2float(x); }
__device__ __forceinline__ float tof(__nv_bfloat16 x) { return __bfloat162float(x); }
template <class T> __device__ __forceinline__ T fromf(float);
template <> __device__ __forceinline__ float fromf<float>(float x) { return x; }
template <> __device__ __forceinline__ double fromf<double>(float x) { return x; }
template <> __device__ __forceinline__ __half fromf<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <class F> void with_type(DType dt, F&& f) {
  switch (dt) {
    case DType::F32: f(float{}); break;
    case DType::BF16: f(__nv_bfloat16{}); break;
    case DType::F16: f(__half{}); break;
    case DType::F64: f(double{}); break;
  }
}
cudaDataType_t cuda_type(DType dt) {
  switch (dt) { case DType::F32: return CUDA_R_32F; case DType::BF16: return CUDA_R_16BF; case DType::F16: return CUDA_R_16F; case DType::F64: return CUDA_R_64F; }
  return CUDA_R_32F;
}
inline unsigned blocks(size_t n, unsigned t = 256) { return unsigned(std::min<size_t>((n + t - 1) / t, 65535 * 8)); }

// ---------------- kernels ----------------
__device__ __forceinline__ float hash_uniform(unsigned long long seed, unsigned long long i) {
  unsigned long long z = seed * 0x9E3779B97F4A7C15ull + i + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
  return float(z >> 40) * (2.0f / 16777216.0f) - 1.0f;
}
template <class TT> __global__ void k_fill(TT* x, size_t n, unsigned long long seed, float sc) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = fromf<TT>(sc * hash_uniform(seed, i));
}
template <class TS, class TD> __global__ void k_cast(TD* y, const TS* x, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = fromf<TD>(tof(x[i]));
}
template <class TT> __global__ void k_axpy(size_t n, float a, const TT* x, TT* y) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = fromf<TT>(tof(y[i]) + a * tof(x[i]));
}
template <class TT> __global__ void k_scale(size_t n, float a, TT* x) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = fromf<TT>(a * tof(x[i]));
}
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ float gelu_d(float x) {
  float u = 0.7978845608f * (x + 0.044715f * x * x * x), th = tanhf(u);
  return 0.5f * (1.f + th) + 0.5f * x * (1.f - th * th) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
}
template <class TT> __global__ void k_gelu_f(size_t n, const TT* x, TT* y) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = fromf<TT>(gelu(tof(x[i])));
}
template <class TT> __global__ void k_gelu_b(size_t n, const TT* x, const TT* g, TT* d) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    d[i] = fromf<TT>(tof(g[i]) * gelu_d(tof(x[i])));
}
__device__ float block_reduce(float v, bool is_max) {
  __shared__ float sh[32];
  for (int o = 16; o > 0; o >>= 1) { float w = __shfl_down_sync(0xffffffffu, v, o); v = is_max ? fmaxf(v, w) : v + w; }
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = threadIdx.x < nw ? sh[threadIdx.x] : (is_max ? -INFINITY : 0.f);
  if (wid == 0)
    for (int o = 16; o > 0; o >>= 1) { float w = __shfl_down_sync(0xffffffffu, v, o); v = is_max ? fmaxf(v, w) : v + w; }
  __shared__ float res;
  if (threadIdx.x == 0) res = v;
  __syncthreads();
  return res;
}
template <class TT> __global__ void k_softmax_cols(TT* S, long long rows, long long ld, float sc, bool causal, long long qoff) {
  long long c = blockIdx.x; TT* s = S + c * ld;
  long long vis = causal ? min(rows, qoff + c + 1) : rows; if (vis < 0) vis = 0;
  float mx = -INFINITY;
  for (long long r = threadIdx.x; r < vis; r += blockDim.x) mx = fmaxf(mx, sc * tof(s[r]));
  mx = block_reduce(mx, true);
  float sum = 0.f;
  for (long long r = threadIdx.x; r < vis; r += blockDim.x) sum += __expf(sc * tof(s[r]) - mx);
  sum = block_reduce(sum, false);
  float inv = sum > 0.f ? 1.f / sum : 0.f;
  for (long long r = threadIdx.x; r < rows; r += blockDim.x)
    s[r] = fromf<TT>(r < vis ? __expf(sc * tof(s[r]) - mx) * inv : 0.f);
}
template <class TT> __global__ void k_softmax_bwd(const TT* P, TT* dP, long long rows, long long ld, float sc) {
  long long c = blockIdx.x; const TT* p = P + c * ld; TT* d = dP + c * ld;
  float dot = 0.f;
  for (long long r = threadIdx.x; r < rows; r += blockDim.x) dot += tof(p[r]) * tof(d[r]);
  dot = block_reduce(dot, false);
  for (long long r = threadIdx.x; r < rows; r += blockDim.x) d[r] = fromf<TT>(sc * tof(p[r]) * (tof(d[r]) - dot));
}
template <class TT> __global__ void k_copy_rows(const TT* src, long long lds, bool srm, const int* srow, TT* dst,
                                                long long ldd, bool drm, const int* drow, long long n, long long width,
                                                const float* w) {
  size_t total = size_t(n) * width;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    long long r = i / width, c = i % width;
    long long sr = srow ? srow[r] : r, dr = drow ? drow[r] : r;
    float v = sr < 0 ? 0.f : (w ? w[r] : 1.f) * tof(src[srm ? sr * lds + c : sr + c * lds]);
    dst[drm ? dr * ldd + c : dr + c * ldd] = fromf<TT>(v);
  }
}
// Accumulating variant: one thread per column walks all rows -> no write races
// even when drow contains duplicates (e.g. embedding gradient).
template <class TT> __global__ void k_copy_rows_acc(const TT* src, long long lds, bool srm, const int* srow, TT* dst,
                                                    long long ldd, bool drm, const int* drow, long long n,
                                                    long long width, const float* w) {
  for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; c < width; c += (long long)gridDim.x * blockDim.x)
    for (long long r = 0; r < n; ++r) {
      long long sr = srow ? srow[r] : r, dr = drow ? drow[r] : r;
      if (sr < 0) continue;
      TT& o = dst[drm ? dr * ldd + c : dr + c * ldd];
      o = fromf<TT>(tof(o) + (w ? w[r] : 1.f) * tof(src[srm ? sr * lds + c : sr + c * lds]));
    }
}
__device__ __forceinline__ double tod(double x) { return x; }
template <class TT> __device__ __forceinline__ double tod(TT x) { return tof(x); }
template <class TP, class TT> __global__ void k_adamw(TP* p, const TT* g, TP* m, TP* v, size_t n, TP lr, TP b1,
                                                      TP b2, TP eps, TP wd, TP c1, TP c2, TP gs) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    TP gi = gs * TP(tod(g[i]));
    TP mi = b1 * m[i] + (TP(1) - b1) * gi, vi = b2 * v[i] + (TP(1) - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] -= lr * ((mi / c1) / (sqrt(vi / c2) + eps) + wd * p[i]);
  }
}
template <class TT> __global__ void k_sumsq(const TT* x, size_t n, double* out) {
  float s = 0.f;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    float f = tof(x[i]); s += f * f;
  }
  s = block_reduce(s, false);
  if (threadIdx.x == 0) atomicAdd(out, double(s));
}
template <class TT> __global__ void k_nonfinite(const TT* x, size_t n, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    c += !isfinite(tof(x[i]));
  if (c) atomicAdd(out, c);
}
template <class TT> __global__ void k_colstat(const TT* L, long long rows, long long ld, const float* mx, float* out) {
  long long c = blockIdx.x; const TT* l = L + c * ld;
  float v = mx ? 0.f : -INFINITY;
  for (long long r = threadIdx.x; r < rows; r += blockDim.x)
    v = mx ? v + __expf(tof(l[r]) - mx[c]) : fmaxf(v, tof(l[r]));
  v = block_reduce(v, mx == nullptr);
  if (threadIdx.x == 0) out[c] = v;
}
template <class TT> __global__ void k_xent(TT* L, long long rows, long long ld, const float* mx, const float* sm,
                                           const int* tgt, float sc) {
  long long c = blockIdx.x; TT* l = L + c * ld;
  for (long long r = threadIdx.x; r < rows; r += blockDim.x) {
    float pr = __expf(tof(l[r]) - mx[c]) / sm[c];
    l[r] = fromf<TT>(sc * (pr - (tgt[c] == r ? 1.f : 0.f)));
  }
}
struct Ptrs16 { const void* p[16]; };
template <class TT> __global__ void k_reduce(TT* d, Ptrs16 s, int k, size_t n, bool acc) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    float v = acc ? tof(d[i]) : 0.f;
    for (int j = 0; j < k; ++j) v += tof(static_cast<const TT*>(s.p[j])[i]);
    d[i] = fromf<TT>(v);
  }
}
}  // namespace

bool g_tf32 = false;
void set_tf32(bool on) { g_tf32 = on; }
const char* gemm_engine() { return g_tf32 ? "cuBLAS GemmEx (tensor cores: DMMA/HMMA, TF32 for fp32)" : "cuBLAS GemmEx (tensor cores: DMMA/HMMA)"; }

void init(int device) {
  W.device = device;
  CUDA_OK(cudaSetDevice(device));
  int major = 0; CUDA_OK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  if (major < 8) fail("xpsim GPU backend requires sm_80 or newer");
  CUDA_OK(cudaStreamCreateWithFlags(&W.stream, cudaStreamNonBlocking));
  CUBLAS_OK(cublasCreate(&W.blas));
  CUBLAS_OK(cublasSetStream(W.blas, W.stream));
}
void finalize() {
  if (W.blas) cublasDestroy(W.blas);
  if (W.iscratch) cudaFree(W.iscratch);
  if (W.fscratch) cudaFree(W.fscratch);
  if (W.stream) cudaStreamDestroy(W.stream);
  W = Worker{};
}
int count() { int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess) return 0; return n; }
std::string name() { cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, W.device)); return p.name; }

void* alloc(size_t b) { void* p = nullptr; CUDA_OK(cudaMalloc(&p, b)); return p; }
void free(void* p) { cudaStreamSynchronize(W.stream); cudaFree(p); }
void* host_alloc(size_t b) { void* p = nullptr; CUDA_OK(cudaMallocHost(&p, b)); return p; }
void host_free(void* p) { cudaFreeHost(p); }
void memcpy_any(void* d, const void* s, size_t b, int kind) {
  CUDA_OK(cudaMemcpyAsync(d, s, b, cudaMemcpyDefault, W.stream));
  if (kind != CPY_D2D) CUDA_OK(cudaStreamSynchronize(W.stream));
}
void memset_zero(void* p, size_t b) { CUDA_OK(cudaMemsetAsync(p, 0, b, W.stream)); }
void copy2d(void* d, size_t ldd, const void* s, size_t lds, size_t rows, size_t cols, size_t es) {
  CUDA_OK(cudaMemcpy2DAsync(d, ldd * es, s, lds * es, rows * es, cols, cudaMemcpyDefault, W.stream));
}
void sync() { CUDA_OK(cudaStreamSynchronize(W.stream)); }

void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha, const void* A, DType dtA,
                          Int lda, Int sA, const void* B, DType dtB, Int ldb, Int sB, float beta, void* C, DType dtC,
                          Int ldc, Int sC, Int batch) {
  if (dtA != dtB) fail("gpu gemm: A and B must share a dtype");
  if (dtA == DType::F32 && dtC != DType::F32) fail("gpu gemm: fp32 inputs need fp32 output");
  auto op = [](Orientation o) { return o == N ? CUBLAS_OP_N : CUBLAS_OP_T; };
  if (k == 0) {  // C = beta*C
    with_type(dtC, [&](auto t) { using TT = decltype(t);
      for (Int b = 0; b < batch; ++b)
        for (Int j = 0; j < n; ++j) k_scale<TT><<<blocks(m), 256, 0, W.stream>>>(m, beta, static_cast<TT*>(C) + b * sC + j * ldc);
    });
    return;
  }
  // sm_80+: cuBLAS selects tensor-core kernels for all of these compute types:
  //   FP64      -> CUBLAS_COMPUTE_64F           (DMMA FP64 tensor cores)
  //   FP32      -> CUBLAS_COMPUTE_32F_FAST_TF32 (TF32 tensor cores) if --tf32, else SIMT FP32
  //   BF16/FP16 -> CUBLAS_COMPUTE_32F           (HMMA, FP32 accumulation)
  if (dtA == DType::F64) {
    if (dtC != DType::F64) fail("gpu gemm: fp64 inputs need fp64 output");
    double a = alpha, b = beta;
    CUBLAS_OK(cublasGemmStridedBatchedEx(W.blas, op(oA), op(oB), int(m), int(n), int(k), &a, A, CUDA_R_64F, int(lda), sA,
                                         B, CUDA_R_64F, int(ldb), sB, &b, C, CUDA_R_64F, int(ldc), sC, int(batch),
                                         CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
    return;
  }
  cublasComputeType_t ct = (dtA == DType::F32 && g_tf32) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  CUBLAS_OK(cublasGemmStridedBatchedEx(W.blas, op(oA), op(oB), int(m), int(n), int(k), &alpha, A, cuda_type(dtA), int(lda),
                                       sA, B, cuda_type(dtB), int(ldb), sB, &beta, C, cuda_type(dtC), int(ldc), sC,
                                       int(batch), ct, CUBLAS_GEMM_DEFAULT));
}

void fill_random(void* p, size_t n, DType dt, uint64_t seed, float sc) {
  with_type(dt, [&](auto t) { using TT = decltype(t); k_fill<TT><<<blocks(n), 256, 0, W.stream>>>(static_cast<TT*>(p), n, seed, sc); });
}
void cast(void* d, DType dtd, const void* s, DType dts, size_t n) {
  with_type(dts, [&](auto ts) { with_type(dtd, [&](auto td) {
    using TS = decltype(ts); using TD = decltype(td);
    k_cast<TS, TD><<<blocks(n), 256, 0, W.stream>>>(static_cast<TD*>(d), static_cast<const TS*>(s), n);
  }); });
}
void axpy(size_t n, float a, const void* x, void* y, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t); k_axpy<TT><<<blocks(n), 256, 0, W.stream>>>(n, a, static_cast<const TT*>(x), static_cast<TT*>(y)); });
}
void scale(size_t n, float a, void* x, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t); k_scale<TT><<<blocks(n), 256, 0, W.stream>>>(n, a, static_cast<TT*>(x)); });
}
void gelu_fwd(size_t n, const void* i, void* o, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t); k_gelu_f<TT><<<blocks(n), 256, 0, W.stream>>>(n, static_cast<const TT*>(i), static_cast<TT*>(o)); });
}
void gelu_bwd(size_t n, const void* p, const void* g, void* d, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t);
    k_gelu_b<TT><<<blocks(n), 256, 0, W.stream>>>(n, static_cast<const TT*>(p), static_cast<const TT*>(g), static_cast<TT*>(d)); });
}
void softmax_cols(void* S, Int rows, Int cols, Int ld, float sc, bool causal, Int qoff, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t);
    k_softmax_cols<TT><<<unsigned(cols), 256, 0, W.stream>>>(static_cast<TT*>(S), rows, ld, sc, causal, qoff); });
}
void softmax_bwd_cols(const void* P, void* dP, Int rows, Int cols, Int ld, float sc, DType dt) {
  with_type(dt, [&](auto t) { using TT = decltype(t);
    k_softmax_bwd<TT><<<unsigned(cols), 256, 0, W.stream>>>(static_cast<const TT*>(P), static_cast<TT*>(dP), rows, ld, sc); });
}
void copy_rows(const void* src, Int lds, bool srm, const int* srow, void* dst, Int ldd, bool drm, const int* drow,
               Int n, Int width, const float* w, bool acc, DType dt) {
  size_t ib = (srow ? n : 0) * sizeof(int), jb = (drow ? n : 0) * sizeof(int), wb = (w ? n : 0) * sizeof(float);
  char* base = static_cast<char*>(scratch(ib + jb + wb + 64));
  int* ds = srow ? reinterpret_cast<int*>(base) : nullptr;
  int* dd = drow ? reinterpret_cast<int*>(base + ib) : nullptr;
  float* dw = w ? reinterpret_cast<float*>(base + ib + jb) : nullptr;
  if (ds) CUDA_OK(cudaMemcpyAsync(ds, srow, ib, cudaMemcpyHostToDevice, W.stream));
  if (dd) CUDA_OK(cudaMemcpyAsync(dd, drow, jb, cudaMemcpyHostToDevice, W.stream));
  if (dw) CUDA_OK(cudaMemcpyAsync(dw, w, wb, cudaMemcpyHostToDevice, W.stream));
  with_type(dt, [&](auto t) { using TT = decltype(t);
    if (acc) k_copy_rows_acc<TT><<<blocks(width), 256, 0, W.stream>>>(static_cast<const TT*>(src), lds, srm, ds, static_cast<TT*>(dst), ldd, drm, dd, n, width, dw);
    else k_copy_rows<TT><<<blocks(size_t(n) * width), 256, 0, W.stream>>>(static_cast<const TT*>(src), lds, srm, ds, static_cast<TT*>(dst), ldd, drm, dd, n, width, dw);
  });
  // The host arrays may die after return and scratch is reused: keep it simple.
  CUDA_OK(cudaStreamSynchronize(W.stream));
}
void adamw(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1, float b2,
           float eps, float wd, int step, float gs) {
  double c1 = 1.0 - std::pow(double(b1), step), c2 = 1.0 - std::pow(double(b2), step);
  auto launch = [&](auto tp) {
    using TP = decltype(tp);
    with_type(dtg, [&](auto t) { using TT = decltype(t);
      k_adamw<TP, TT><<<blocks(n), 256, 0, W.stream>>>(static_cast<TP*>(p), static_cast<const TT*>(g), static_cast<TP*>(m),
          static_cast<TP*>(v), n, TP(lr), TP(b1), TP(b2), TP(eps), TP(wd), TP(c1), TP(c2), TP(gs)); });
  };
  if (dtp == DType::F64) launch(double{}); else if (dtp == DType::F32) launch(float{}); else fail("master must be fp32/fp64");
}
double sumsq(const void* x, size_t n, DType dt) {
  double* d = static_cast<double*>(scratch(sizeof(double)));
  CUDA_OK(cudaMemsetAsync(d, 0, sizeof(double), W.stream));
  with_type(dt, [&](auto t) { using TT = decltype(t); k_sumsq<TT><<<blocks(n), 256, 0, W.stream>>>(static_cast<const TT*>(x), n, d); });
  double h = 0; CUDA_OK(cudaMemcpyAsync(&h, d, sizeof(double), cudaMemcpyDeviceToHost, W.stream));
  CUDA_OK(cudaStreamSynchronize(W.stream));
  return h;
}
size_t count_nonfinite(const void* x, size_t n, DType dt) {
  auto* d = static_cast<unsigned long long*>(scratch(8));
  CUDA_OK(cudaMemsetAsync(d, 0, 8, W.stream));
  with_type(dt, [&](auto t) { using TT = decltype(t); k_nonfinite<TT><<<blocks(n), 256, 0, W.stream>>>(static_cast<const TT*>(x), n, d); });
  unsigned long long h = 0; CUDA_OK(cudaMemcpyAsync(&h, d, 8, cudaMemcpyDeviceToHost, W.stream));
  CUDA_OK(cudaStreamSynchronize(W.stream));
  return size_t(h);
}
void col_max(const void* L, Int rows, Int cols, Int ld, DType dt, float* out) {
  float* d = fscratch(cols);
  with_type(dt, [&](auto t) { using TT = decltype(t); k_colstat<TT><<<unsigned(cols), 256, 0, W.stream>>>(static_cast<const TT*>(L), rows, ld, nullptr, d); });
  CUDA_OK(cudaMemcpyAsync(out, d, cols * sizeof(float), cudaMemcpyDeviceToHost, W.stream));
  CUDA_OK(cudaStreamSynchronize(W.stream));
}
void col_sumexp(const void* L, Int rows, Int cols, Int ld, DType dt, const float* mx, float* out) {
  float* d = fscratch(2 * cols);
  CUDA_OK(cudaMemcpyAsync(d + cols, mx, cols * sizeof(float), cudaMemcpyHostToDevice, W.stream));
  with_type(dt, [&](auto t) { using TT = decltype(t); k_colstat<TT><<<unsigned(cols), 256, 0, W.stream>>>(static_cast<const TT*>(L), rows, ld, d + cols, d); });
  CUDA_OK(cudaMemcpyAsync(out, d, cols * sizeof(float), cudaMemcpyDeviceToHost, W.stream));
  CUDA_OK(cudaStreamSynchronize(W.stream));
}
void xent_grad(void* L, Int rows, Int cols, Int ld, DType dt, const float* mx, const float* sm, const int* tgt, float sc) {
  float* d = fscratch(2 * cols);
  int* it = static_cast<int*>(scratch(cols * sizeof(int)));
  CUDA_OK(cudaMemcpyAsync(d, mx, cols * sizeof(float), cudaMemcpyHostToDevice, W.stream));
  CUDA_OK(cudaMemcpyAsync(d + cols, sm, cols * sizeof(float), cudaMemcpyHostToDevice, W.stream));
  CUDA_OK(cudaMemcpyAsync(it, tgt, cols * sizeof(int), cudaMemcpyHostToDevice, W.stream));
  with_type(dt, [&](auto t) { using TT = decltype(t); k_xent<TT><<<unsigned(cols), 256, 0, W.stream>>>(static_cast<TT*>(L), rows, ld, d, d + cols, it, sc); });
  CUDA_OK(cudaStreamSynchronize(W.stream));
}

// ---------------- transport engine helpers ----------------
void xfer_copy(int device, void* d, const void* s, size_t b) {
  cudaStream_t st = engine_stream(device);
  CUDA_OK(cudaSetDevice(device));
  CUDA_OK(cudaMemcpyAsync(d, s, b, cudaMemcpyDefault, st));  // VMM peer pointers are UVA
}
void xfer_reduce(int device, void* d, const void* const* srcs, int k, size_t n, DType dt, bool acc) {
  cudaStream_t st = engine_stream(device);
  CUDA_OK(cudaSetDevice(device));
  for (int j0 = 0; j0 < k; j0 += 16) {
    Ptrs16 p{}; int kk = std::min(16, k - j0);
    for (int j = 0; j < kk; ++j) p.p[j] = srcs[j0 + j];
    bool a = acc || j0 > 0;
    with_type(dt, [&](auto t) { using TT = decltype(t); k_reduce<TT><<<blocks(n), 256, 0, st>>>(static_cast<TT*>(d), p, kk, n, a); });
  }
}
void xfer_sync(int device) { CUDA_OK(cudaSetDevice(device)); CUDA_OK(cudaStreamSynchronize(engine_stream(device))); }

}  // namespace xpsim::dev::gpu
