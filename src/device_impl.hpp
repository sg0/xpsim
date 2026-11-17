// Internal: per-backend kernel entry points with identical signatures.
#pragma once
#include "xpsim/device.hpp"

#define XPSIM_BACKEND_DECLS                                                                          \
  void* alloc(size_t bytes);                                                                         \
  void free(void* p);                                                                                \
  void* host_alloc(size_t bytes);                                                                    \
  void host_free(void* p);                                                                           \
  void memcpy_any(void* dst, const void* src, size_t bytes, int kind);                               \
  void memset_zero(void* p, size_t bytes);                                                           \
  void copy2d(void* dst, size_t ldd, const void* src, size_t lds, size_t rows, size_t cols, size_t es); \
  void sync();                                                                                       \
  void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha,        \
                            const void* A, DType dtA, Int lda, Int sA, const void* B, DType dtB,     \
                            Int ldb, Int sB, float beta, void* C, DType dtC, Int ldc, Int sC,        \
                            Int batch);                                                              \
  void fill_random(void* p, size_t n, DType dt, uint64_t seed, float scale);                         \
  void cast(void* dst, DType dtd, const void* src, DType dts, size_t n);                             \
  void axpy(size_t n, float alpha, const void* x, void* y, DType dt);                                \
  void scale(size_t n, float alpha, void* x, DType dt);                                              \
  void gelu_fwd(size_t n, const void* in, void* out, DType dt);                                      \
  void gelu_bwd(size_t n, const void* pre, const void* dout, void* din, DType dt);                   \
  void softmax_cols(void* S, Int rows, Int cols, Int ld, float scale, bool causal, Int qoff, DType dt); \
  void softmax_bwd_cols(const void* P, void* dP, Int rows, Int cols, Int ld, float scale, DType dt); \
  void copy_rows(const void* src, Int lds, bool src_rm, const int* srow, void* dst, Int ldd,         \
                 bool dst_rm, const int* drow, Int n, Int width, const float* w, bool acc, DType dt); \
  void adamw(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr,    \
             float b1, float b2, float eps, float wd, int step, float gscale);                       \
  double sumsq(const void* p, size_t n, DType dt);                                                   \
  size_t count_nonfinite(const void* p, size_t n, DType dt);                                         \
  void col_max(const void* L, Int rows, Int cols, Int ld, DType dt, float* out);                     \
  void col_sumexp(const void* L, Int rows, Int cols, Int ld, DType dt, const float* mx, float* out); \
  void xent_grad(void* L, Int rows, Int cols, Int ld, DType dt, const float* mx, const float* sm,    \
                 const int* tgt, float scale);                                                       \
  void xfer_copy(int device, void* dst, const void* src, size_t bytes);                              \
  void xfer_reduce(int device, void* dst, const void* const* srcs, int k, size_t n, DType dt, bool acc); \
  void xfer_sync(int device);                                                                        \
  void set_tf32(bool on);                                                                            \
  const char* gemm_engine();

namespace xpsim::dev::cpu { XPSIM_BACKEND_DECLS }
namespace xpsim::dev::gpu {
XPSIM_BACKEND_DECLS
void init(int device);
void finalize();
int count();
std::string name();
}
// memcpy kinds
namespace xpsim::dev { enum { CPY_D2D = 0, CPY_H2D = 1, CPY_D2H = 2 }; }
