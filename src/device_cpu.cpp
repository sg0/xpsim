// CPU backend: blocked, OpenMP-parallel kernels. Half-precision storage is
// converted to fp32 for arithmetic (fp32 accumulation, like tensor cores).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#include "device_impl.hpp"
#include "xpsim/half_host.hpp"

namespace xpsim::dev::cpu {

template <class F> static void with_type(DType dt, F&& f) {
  switch (dt) {
    case DType::F32: f(float{}); break;
    case DType::BF16: f(bf16_t{}); break;
    case DType::F16: f(fp16_t{}); break;
    case DType::F64: f(double{}); break;
  }
}

void* alloc(size_t bytes) { return std::aligned_alloc(64, (bytes + 63) / 64 * 64); }
void free(void* p) { std::free(p); }
void* host_alloc(size_t bytes) { return std::aligned_alloc(64, (bytes + 63) / 64 * 64); }
void host_free(void* p) { std::free(p); }
void memcpy_any(void* d, const void* s, size_t b, int) { std::memcpy(d, s, b); }
void memset_zero(void* p, size_t b) { std::memset(p, 0, b); }
void copy2d(void* d, size_t ldd, const void* s, size_t lds, size_t rows, size_t cols, size_t es) {
  auto* dc = static_cast<char*>(d); auto* sc = static_cast<const char*>(s);
  if (ldd == rows && lds == rows) { std::memcpy(dc, sc, rows * cols * es); return; }
  for (size_t j = 0; j < cols; ++j) std::memcpy(dc + j * ldd * es, sc + j * lds * es, rows * es);
}
void sync() {}

// ---------------------------------------------------------------- GEMM
#ifdef XPSIM_CBLAS
// Prototypes declared here so any CBLAS works (MKL, OpenBLAS, BLIS, ...).
extern "C" {
void cblas_sgemm(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
void cblas_dgemm(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
}
#endif
namespace {
thread_local std::vector<float> scrA, scrB, scrC;
thread_local std::vector<double> dscrA, dscrB;

// Built-in fallback: C(m x n) += alpha * A(m x k) * B(k x n), column-major.
template <class R>
void gemm_nn(Int m, Int n, Int k, R alpha, const R* A, Int lda, const R* B, Int ldb, R* C, Int ldc) {
  constexpr Int MB = 256, KB = 128, NB = 64;
#pragma omp parallel for schedule(static) collapse(2) if (m * n * k > 64 * 64 * 64)
  for (Int jb = 0; jb < n; jb += NB)
    for (Int ib = 0; ib < m; ib += MB) {
      const Int je = std::min(n, jb + NB), ie = std::min(m, ib + MB), mb = ie - ib;
      for (Int pb = 0; pb < k; pb += KB) {
        const Int pe = std::min(k, pb + KB);
        Int j = jb;
        for (; j + 4 <= je; j += 4) {
          R* c0 = C + ib + j * ldc; R* c1 = c0 + ldc; R* c2 = c1 + ldc; R* c3 = c2 + ldc;
          for (Int p = pb; p < pe; ++p) {
            const R* a = A + ib + p * lda;
            const R b0 = alpha * B[p + j * ldb], b1 = alpha * B[p + (j + 1) * ldb];
            const R b2 = alpha * B[p + (j + 2) * ldb], b3 = alpha * B[p + (j + 3) * ldb];
#pragma omp simd
            for (Int i = 0; i < mb; ++i) {
              const R ai = a[i];
              c0[i] += ai * b0; c1[i] += ai * b1; c2[i] += ai * b2; c3[i] += ai * b3;
            }
          }
        }
        for (; j < je; ++j) {
          R* c0 = C + ib + j * ldc;
          for (Int p = pb; p < pe; ++p) {
            const R* a = A + ib + p * lda; const R b0 = alpha * B[p + j * ldb];
#pragma omp simd
            for (Int i = 0; i < mb; ++i) c0[i] += a[i] * b0;
          }
        }
      }
    }
}

// Materialize op(X) (rows x cols) in real type R, column-major, ld = rows.
template <class R, class TT>
void to_real(std::vector<R>& out, const TT* X, Int ldx, bool trans, Int rows, Int cols) {
  out.resize(size_t(rows) * cols);
  R* o = out.data();
  if (!trans) {
#pragma omp parallel for if (rows * cols > 1 << 16)
    for (Int j = 0; j < cols; ++j)
      for (Int i = 0; i < rows; ++i) o[i + j * rows] = R(to_float_exact(X[i + j * ldx]));
  } else {
    constexpr Int BS = 32;
#pragma omp parallel for collapse(2) if (rows * cols > 1 << 16)
    for (Int jb = 0; jb < cols; jb += BS)
      for (Int ib = 0; ib < rows; ib += BS)
        for (Int j = jb; j < std::min(cols, jb + BS); ++j)
          for (Int i = ib; i < std::min(rows, ib + BS); ++i) o[i + j * rows] = R(to_float_exact(X[j + i * ldx]));
  }
}

// C = alpha op(A) op(B) + beta C in real type R (float or double).
template <class R>
void real_gemm(bool ta, bool tb, Int m, Int n, Int k, R alpha, const R* A, Int lda, const R* B, Int ldb, R beta, R* C,
               Int ldc, std::vector<R>& sa, std::vector<R>& sb) {
#ifdef XPSIM_CBLAS
  constexpr int ColMajor = 102, NoTrans = 111, Trans = 112;
  if constexpr (std::is_same_v<R, float>)
    cblas_sgemm(ColMajor, ta ? Trans : NoTrans, tb ? Trans : NoTrans, int(m), int(n), int(k), alpha, A, int(lda), B,
                int(ldb), beta, C, int(ldc));
  else
    cblas_dgemm(ColMajor, ta ? Trans : NoTrans, tb ? Trans : NoTrans, int(m), int(n), int(k), alpha, A, int(lda), B,
                int(ldb), beta, C, int(ldc));
#else
  if (ta) { to_real(sa, A, lda, true, m, k); A = sa.data(); lda = m; }
  if (tb) { to_real(sb, B, ldb, true, k, n); B = sb.data(); ldb = k; }
  for (Int j = 0; j < n; ++j) {
    R* c = C + j * ldc;
    if (beta == R(0)) std::fill(c, c + m, R(0));
    else if (beta != R(1)) for (Int i = 0; i < m; ++i) c[i] *= beta;
  }
  if (k > 0) gemm_nn<R>(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#endif
}
}  // namespace

void gemm_strided_batched(Orientation oA, Orientation oB, Int m, Int n, Int k, float alpha, const void* A,
                          DType dtA, Int lda, Int sA, const void* B, DType dtB, Int ldb, Int sB, float beta,
                          void* C, DType dtC, Int ldc, Int sC, Int batch) {
  const bool ta = oA == T, tb = oB == T;
  for (Int b = 0; b < batch; ++b) {
    const char* Ab = static_cast<const char*>(A) + size_t(b * sA) * dtype_size(dtA);
    const char* Bb = static_cast<const char*>(B) + size_t(b * sB) * dtype_size(dtB);
    char* Cb = static_cast<char*>(C) + size_t(b * sC) * dtype_size(dtC);
    if (dtA == DType::F64 || dtB == DType::F64 || dtC == DType::F64) {  // native DGEMM
      XPSIM_CHECK(dtA == DType::F64 && dtB == DType::F64 && dtC == DType::F64, "fp64 GEMM needs all-fp64 operands");
      real_gemm<double>(ta, tb, m, n, k, alpha, reinterpret_cast<const double*>(Ab), lda,
                        reinterpret_cast<const double*>(Bb), ldb, beta, reinterpret_cast<double*>(Cb), ldc, dscrA, dscrB);
      continue;
    }
    // fp32 path; 16-bit operands are widened to fp32 (fp32 accumulation).
    const float* Af = reinterpret_cast<const float*>(Ab); Int ldaf = lda; bool tA = ta;
    if (dtA != DType::F32) {
      with_type(dtA, [&](auto t) { to_real(scrA, reinterpret_cast<const decltype(t)*>(Ab), lda, ta, m, k); });
      Af = scrA.data(); ldaf = m; tA = false;
    }
    const float* Bf = reinterpret_cast<const float*>(Bb); Int ldbf = ldb; bool tB = tb;
    if (dtB != DType::F32) {
      with_type(dtB, [&](auto t) { to_real(scrB, reinterpret_cast<const decltype(t)*>(Bb), ldb, tb, k, n); });
      Bf = scrB.data(); ldbf = k; tB = false;
    }
    if (dtC == DType::F32) {
      real_gemm<float>(tA, tB, m, n, k, alpha, Af, ldaf, Bf, ldbf, beta, reinterpret_cast<float*>(Cb), ldc, scrA, scrB);
    } else {
      with_type(dtC, [&](auto t) {
        using TC = decltype(t);
        TC* Ct = reinterpret_cast<TC*>(Cb);
        scrC.assign(size_t(m) * n, 0.f);
        if (beta != 0.f)
          for (Int j = 0; j < n; ++j)
            for (Int i = 0; i < m; ++i) scrC[i + j * m] = beta * to_float(Ct[i + j * ldc]);
        std::vector<float> ta2, tb2;
        real_gemm<float>(tA, tB, m, n, k, alpha, Af, ldaf, Bf, ldbf, 1.f, scrC.data(), m, ta2, tb2);
        for (Int j = 0; j < n; ++j)
          for (Int i = 0; i < m; ++i) Ct[i + j * ldc] = from_float<TC>(scrC[i + j * m]);
      });
    }
  }
}
void set_tf32(bool) {}
const char* gemm_engine() {
#ifdef XPSIM_CBLAS
  return "vendor CBLAS (sgemm/dgemm)";
#else
  return "built-in blocked GEMM (no CBLAS)";
#endif
}

// ---------------------------------------------------------------- elementwise
static inline float hash_uniform(uint64_t seed, uint64_t i) {
  uint64_t z = seed * 0x9E3779B97F4A7C15ull + i + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
  return float(z >> 40) * (2.0f / 16777216.0f) - 1.0f;  // [-1, 1)
}
void fill_random(void* p, size_t n, DType dt, uint64_t seed, float sc) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); TT* x = static_cast<TT*>(p);
#pragma omp parallel for if (n > 1 << 16)
    for (size_t i = 0; i < n; ++i) x[i] = from_float<TT>(sc * hash_uniform(seed, i));
  });
}
void cast(void* d, DType dtd, const void* s, DType dts, size_t n) {
  with_type(dts, [&](auto ts) { with_type(dtd, [&](auto td) {
    using TS = decltype(ts); using TD = decltype(td);
    const TS* x = static_cast<const TS*>(s); TD* y = static_cast<TD*>(d);
#pragma omp parallel for if (n > 1 << 16)
    for (size_t i = 0; i < n; ++i) y[i] = from_float<TD>(to_float(x[i]));
  }); });
}
void axpy(size_t n, float a, const void* xv, void* yv, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* x = static_cast<const TT*>(xv); TT* y = static_cast<TT*>(yv);
#pragma omp parallel for if (n > 1 << 16)
    for (size_t i = 0; i < n; ++i) y[i] = from_float<TT>(to_float(y[i]) + a * to_float(x[i]));
  });
}
void scale(size_t n, float a, void* xv, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); TT* x = static_cast<TT*>(xv);
#pragma omp parallel for if (n > 1 << 16)
    for (size_t i = 0; i < n; ++i) x[i] = from_float<TT>(a * to_float(x[i]));
  });
}
static inline float gelu(float x) { return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
static inline float gelu_d(float x) {
  float u = 0.7978845608f * (x + 0.044715f * x * x * x), th = std::tanh(u);
  return 0.5f * (1.f + th) + 0.5f * x * (1.f - th * th) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
}
void gelu_fwd(size_t n, const void* in, void* out, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* x = static_cast<const TT*>(in); TT* y = static_cast<TT*>(out);
#pragma omp parallel for if (n > 1 << 15)
    for (size_t i = 0; i < n; ++i) y[i] = from_float<TT>(gelu(to_float(x[i])));
  });
}
void gelu_bwd(size_t n, const void* pre, const void* dout, void* din, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* x = static_cast<const TT*>(pre); const TT* g = static_cast<const TT*>(dout);
    TT* d = static_cast<TT*>(din);
#pragma omp parallel for if (n > 1 << 15)
    for (size_t i = 0; i < n; ++i) d[i] = from_float<TT>(to_float(g[i]) * gelu_d(to_float(x[i])));
  });
}
void softmax_cols(void* Sv, Int rows, Int cols, Int ld, float sc, bool causal, Int qoff, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); TT* S = static_cast<TT*>(Sv);
#pragma omp parallel for if (rows * cols > 1 << 15)
    for (Int c = 0; c < cols; ++c) {
      TT* s = S + c * ld;
      Int vis = causal ? std::min<Int>(rows, qoff + c + 1) : rows;
      float mx = -INFINITY;
      for (Int r = 0; r < vis; ++r) mx = std::max(mx, sc * to_float(s[r]));
      float sum = 0.f;
      for (Int r = 0; r < vis; ++r) sum += std::exp(sc * to_float(s[r]) - mx);
      float inv = sum > 0.f ? 1.f / sum : 0.f;
      for (Int r = 0; r < vis; ++r) s[r] = from_float<TT>(std::exp(sc * to_float(s[r]) - mx) * inv);
      for (Int r = std::max<Int>(vis, 0); r < rows; ++r) s[r] = from_float<TT>(0.f);
    }
  });
}
void softmax_bwd_cols(const void* Pv, void* dPv, Int rows, Int cols, Int ld, float sc, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* P = static_cast<const TT*>(Pv); TT* dP = static_cast<TT*>(dPv);
#pragma omp parallel for if (rows * cols > 1 << 15)
    for (Int c = 0; c < cols; ++c) {
      const TT* p = P + c * ld; TT* d = dP + c * ld; float dot = 0.f;
      for (Int r = 0; r < rows; ++r) dot += to_float(p[r]) * to_float(d[r]);
      for (Int r = 0; r < rows; ++r) d[r] = from_float<TT>(sc * to_float(p[r]) * (to_float(d[r]) - dot));
    }
  });
}
void copy_rows(const void* srcv, Int lds, bool srm, const int* srow, void* dstv, Int ldd, bool drm,
               const int* drow, Int n, Int width, const float* w, bool acc, DType dt) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* src = static_cast<const TT*>(srcv); TT* dst = static_cast<TT*>(dstv);
    auto body = [&](Int r) {
      const Int sr = srow ? srow[r] : r, dr = drow ? drow[r] : r;
      const float wr = w ? w[r] : 1.f;
      for (Int c = 0; c < width; ++c) {
        float v = sr < 0 ? 0.f : wr * to_float(src[srm ? sr * lds + c : sr + c * lds]);
        TT& o = dst[drm ? dr * ldd + c : dr + c * ldd];
        o = from_float<TT>(acc ? to_float(o) + v : v);
      }
    };
    if (acc) { for (Int r = 0; r < n; ++r) body(r); }  // duplicates in drow allowed
    else {
#pragma omp parallel for if (n * width > 1 << 15)
      for (Int r = 0; r < n; ++r) body(r);
    }
  });
}
void adamw(void* pv, DType dtp, const void* gv, DType dtg, void* mv, void* vv, size_t n, float lr, float b1, float b2,
           float eps, float wd, int step, float gs) {
  const double c1 = 1.0 - std::pow(double(b1), step), c2 = 1.0 - std::pow(double(b2), step);
  with_type(dtp, [&](auto tp) { with_type(dtg, [&](auto t) {
    using TP = decltype(tp); using TT = decltype(t);
    if constexpr (std::is_same_v<TP, float> || std::is_same_v<TP, double>) {
      TP* p = static_cast<TP*>(pv); TP* m = static_cast<TP*>(mv); TP* v = static_cast<TP*>(vv);
      const TT* g = static_cast<const TT*>(gv);
#pragma omp parallel for if (n > 1 << 15)
      for (size_t i = 0; i < n; ++i) {
        TP gi = TP(gs) * TP(to_float_exact(g[i]));
        m[i] = TP(b1) * m[i] + TP(1 - b1) * gi; v[i] = TP(b2) * v[i] + TP(1 - b2) * gi * gi;
        p[i] -= TP(lr) * ((m[i] / TP(c1)) / (std::sqrt(v[i] / TP(c2)) + TP(eps)) + TP(wd) * p[i]);
      }
    } else fail("master weights must be fp32 or fp64");
  }); });
}
double sumsq(const void* pv, size_t n, DType dt) {
  double s = 0;
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* x = static_cast<const TT*>(pv);
#pragma omp parallel for reduction(+ : s) if (n > 1 << 15)
    for (size_t i = 0; i < n; ++i) { double f = to_float(x[i]); s += f * f; }
  });
  return s;
}
size_t count_nonfinite(const void* pv, size_t n, DType dt) {
  size_t c = 0;
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* x = static_cast<const TT*>(pv);
#pragma omp parallel for reduction(+ : c) if (n > 1 << 15)
    for (size_t i = 0; i < n; ++i) c += !std::isfinite(to_float(x[i]));
  });
  return c;
}
void col_max(const void* Lv, Int rows, Int cols, Int ld, DType dt, float* out) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* L = static_cast<const TT*>(Lv);
#pragma omp parallel for if (rows * cols > 1 << 15)
    for (Int c = 0; c < cols; ++c) {
      float mx = -INFINITY; for (Int r = 0; r < rows; ++r) mx = std::max(mx, to_float(L[r + c * ld])); out[c] = mx;
    }
  });
}
void col_sumexp(const void* Lv, Int rows, Int cols, Int ld, DType dt, const float* mx, float* out) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); const TT* L = static_cast<const TT*>(Lv);
#pragma omp parallel for if (rows * cols > 1 << 15)
    for (Int c = 0; c < cols; ++c) {
      float s = 0.f; for (Int r = 0; r < rows; ++r) s += std::exp(to_float(L[r + c * ld]) - mx[c]); out[c] = s;
    }
  });
}
void xent_grad(void* Lv, Int rows, Int cols, Int ld, DType dt, const float* mx, const float* sm, const int* tgt, float sc) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); TT* L = static_cast<TT*>(Lv);
#pragma omp parallel for if (rows * cols > 1 << 15)
    for (Int c = 0; c < cols; ++c)
      for (Int r = 0; r < rows; ++r) {
        float pr = std::exp(to_float(L[r + c * ld]) - mx[c]) / sm[c];
        L[r + c * ld] = from_float<TT>(sc * (pr - (tgt[c] == r ? 1.f : 0.f)));
      }
  });
}
void xfer_copy(int, void* d, const void* s, size_t b) { std::memcpy(d, s, b); }
void xfer_reduce(int, void* dv, const void* const* srcs, int k, size_t n, DType dt, bool acc) {
  with_type(dt, [&](auto t) {
    using TT = decltype(t); TT* d = static_cast<TT*>(dv);
#pragma omp parallel for if (n > 1 << 16)
    for (size_t i = 0; i < n; ++i) {
      float s = acc ? to_float(d[i]) : 0.f;
      for (int j = 0; j < k; ++j) s += to_float(static_cast<const TT*>(srcs[j])[i]);
      d[i] = from_float<TT>(s);
    }
  });
}
void xfer_sync(int) {}

}  // namespace xpsim::dev::cpu

// ---------------------------------------------------------------- device-resident variants
#include <sys/mman.h>
namespace xpsim::dev::cpu {
void copy_rows_d(const void* src, Int lds, bool srm, const int* srow, void* dst, Int ldd, bool drm, const int* drow,
                 Int n, Int width, const float* w, bool acc, DType dt) {
  copy_rows(src, lds, srm, srow, dst, ldd, drm, drow, n, width, w, acc, dt);  // host == device
}
void col_max_d(const void* L, Int r, Int c, Int ld, DType dt, float* o) { col_max(L, r, c, ld, dt, o); }
void col_sumexp_d(const void* L, Int r, Int c, Int ld, DType dt, const float* mx, float* o) { col_sumexp(L, r, c, ld, dt, mx, o); }
void xent_grad_d(void* L, Int r, Int c, Int ld, DType dt, const float* mx, const float* sm, const int* t, float sc) {
  xent_grad(L, r, c, ld, dt, mx, sm, t, sc);
}
void max_blocks(float* d, const float* s, int p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float m = s[i];
    for (int j = 1; j < p; ++j) m = std::max(m, s[size_t(j) * n + i]);
    d[i] = m;
  }
}
void sumsq_acc(const void* x, size_t n, DType dt, double* acc, double w) { *acc += w * sumsq(x, n, dt); }
void opt_update(OptState* st, float ls, float clip) {
  const double norm = std::sqrt(st->norm2) / ls;
  st->norm = float(norm);
  st->skip = std::isfinite(norm) ? 0 : 1;
  double gs = 1.0 / ls;
  if (clip > 0 && std::isfinite(norm) && norm > clip) gs *= clip / (norm + 1e-6);
  st->gscale = float(gs);
  if (st->skip) ++st->skipped_total; else ++st->step;
}
void adamw_d(void* p, DType dtp, const void* g, DType dtg, void* m, void* v, size_t n, float lr, float b1, float b2,
             float eps, float wd, const OptState* st) {
  if (st->skip) return;
  adamw(p, dtp, g, dtg, m, v, n, lr, b1, b2, eps, wd, st->step, st->gscale);
}
void graph_begin() { fail("CUDA graphs need the GPU backend"); }
void* graph_end(const char*) { fail("CUDA graphs need the GPU backend"); }
void graph_launch(void*) { fail("CUDA graphs need the GPU backend"); }
void graph_destroy(void*) {}
size_t free_memory() { return size_t(64) << 30; }  // virtual reservation; pages are committed on touch
void* arena_reserve(size_t bytes) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  return p == MAP_FAILED ? nullptr : p;
}
void arena_release(void* p, size_t bytes) { munmap(p, bytes); }
}  // namespace xpsim::dev::cpu
