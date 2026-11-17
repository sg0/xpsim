// Device abstraction: one API, two backends (CPU, CUDA GPU).
// All buffers handed to these functions live in "device memory" of the active
// backend (host memory for the CPU backend). The backend state (stream, cuBLAS
// handle, memory tracker) is thread-local so that several simulated workers can
// run as threads of one process (hierarchical transport / MPI shim).
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "xpsim/types.hpp"

namespace xpsim::dev {

enum class Backend { CPU, GPU };

void init(Backend b, int device_id);
void finalize();
Backend backend();
bool is_gpu();
int device_id();
bool cuda_compiled();
int gpu_count();             // visible CUDA devices (0 if not compiled with CUDA)
std::string device_name();

// ---------------- memory (tracked by tag) ----------------
void* alloc(size_t bytes);
void free(void* p);
void* host_alloc(size_t bytes);   // pinned when GPU backend
void host_free(void* p);

struct MemStats { size_t current = 0, peak = 0; };
const std::map<std::string, MemStats>& mem_stats();
size_t mem_current();
size_t mem_peak();
void mem_reset_peak();
// RAII tag for allocations made inside the scope ("params", "grads", ...).
struct MemScope { explicit MemScope(const char* tag); ~MemScope(); const char* prev; };

// ---------------- data movement ----------------
void memcpy_d2d(void* dst, const void* src, size_t bytes);
void memcpy_h2d(void* dst, const void* src, size_t bytes);
void memcpy_d2h(void* dst, const void* src, size_t bytes);
void memset_zero(void* p, size_t bytes);
// 2-D copy of a column-major (rows x cols) block with element size esize.
void copy2d(void* dst, size_t ldd, const void* src, size_t lds, size_t rows, size_t cols, size_t esize);
void sync();  // wait for all work queued by this worker

// ---------------- dense linear algebra (column-major) ----------------
// C = alpha * op(A) * op(B) + beta * C ; A,B share dtA/dtB, C may differ
// (e.g. bf16 inputs with fp32 accumulation into fp32 main_grad).
// GPU: cuBLAS GemmEx, tensor cores on sm_80+ (FP64 DMMA via COMPUTE_64F,
// TF32 for fp32 when enabled, BF16/FP16 HMMA with FP32 accumulation).
// CPU: vendor CBLAS sgemm/dgemm when built with BLAS=..., else built-in kernel.
// Elementwise kernels evaluate in fp32 arithmetic for every storage type.
void gemm(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha,
          const void* A, DType dtA, Int lda, const void* B, DType dtB, Int ldb,
          float beta, void* C, DType dtC, Int ldc);
void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha,
                          const void* A, DType dtA, Int lda, Int strideA,
                          const void* B, DType dtB, Int ldb, Int strideB, float beta,
                          void* C, DType dtC, Int ldc, Int strideC, Int batch);
double gemm_peak_tflops(DType dt, Int n);  // micro-benchmark: attainable GEMM rate

// ---------------- elementwise / reductions ----------------
void fill_random(void* p, size_t n, DType dt, uint64_t seed, float scale);
void cast(void* dst, DType dtd, const void* src, DType dts, size_t n);
void axpy(size_t n, float alpha, const void* x, void* y, DType dt);     // y += a x
void scale(size_t n, float alpha, void* x, DType dt);
void gelu_fwd(size_t n, const void* in, void* out, DType dt);
void gelu_bwd(size_t n, const void* pre, const void* dout, void* din, DType dt);
// Column softmax on S^T (rows = keys, cols = queries), with 1/sqrt(d) scaling and
// an optional causal mask (key j visible to query i iff j <= q_offset + i).
void softmax_cols(void* S, Int rows, Int cols, Int ld, float scale, bool causal, Int q_offset, DType dt);
// dS = scale * P o (dP - colsum(P o dP)), in place on dP.
void softmax_bwd_cols(const void* P, void* dP, Int rows, Int cols, Int ld, float scale, DType dt);
// Generic row copy: dst[drow[r], c] (+)= w[r] * src[srow[r], c] for r < n, c < width.
// Layout flags: row_major=true means element (r,c) at r*ld + c, else r + c*ld.
// Null index arrays mean identity, null weights mean 1. Index arrays are host
// pointers (uploaded internally). Negative source rows produce zeros.
void copy_rows(const void* src, Int lds, bool src_rm, const int* srow,
               void* dst, Int ldd, bool dst_rm, const int* drow,
               Int n, Int width, const float* w, bool accumulate, DType dt);
// Optimizer: AdamW on a master shard (fp32, or fp64 for fp64 training) with
// moments m, v in the master dtype and a gradient of dtype dtg.
void adamw(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1,
           float b2, float eps, float wd, int step, float grad_scale);
// GEMM engine knobs: TF32 tensor cores for fp32 GEMMs (GPU, sm_80+).
void set_tf32(bool on);
const char* gemm_engine();  // human readable: "cuBLAS (tensor cores)", "CBLAS", ...
double sumsq(const void* p, size_t n, DType dt);
size_t count_nonfinite(const void* p, size_t n, DType dt);
// Vocab-parallel cross entropy on L^T (rows = local vocab, cols = tokens).
void col_max(const void* L, Int rows, Int cols, Int ld, DType dt, float* out_host);
void col_sumexp(const void* L, Int rows, Int cols, Int ld, DType dt, const float* max_host, float* out_host);
void xent_grad(void* L, Int rows, Int cols, Int ld, DType dt, const float* max_host,
               const float* sum_host, const int* target_host, float scale);

// ---------------- transport helpers (callable from a progress thread) ------
// Explicit-device variants used by the intra-node engine: they run on an
// internal stream of `device` regardless of the calling thread.
void xfer_copy(int device, void* dst, const void* src, size_t bytes);
void xfer_reduce(int device, void* dst, const void* const* srcs, int k, size_t n, DType dt, bool accumulate);
void xfer_sync(int device);

}  // namespace xpsim::dev
