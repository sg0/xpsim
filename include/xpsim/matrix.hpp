// Matrix<T>: local, column-major matrix in device memory, modelled on
// El::Matrix (Height/Width/LDim/Buffer/Attach/View semantics). Owning
// matrices allocate through dev::alloc (tracked by the current MemScope tag);
// views alias another matrix's storage.
#pragma once
#include <utility>
#include "xpsim/device.hpp"
#include "xpsim/metrics.hpp"

namespace xpsim {

template <typename T>
class Matrix {
 public:
  Matrix() = default;
  Matrix(Int h, Int w) { Resize(h, w); }
  ~Matrix() { Release(); }
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&& o) noexcept { swap(o); }
  Matrix& operator=(Matrix&& o) noexcept { if (this != &o) { Release(); swap(o); } return *this; }

  // ---- El::Matrix-like interface ----
  Int Height() const { return h_; }
  Int Width() const { return w_; }
  Int LDim() const { return ld_; }
  size_t Size() const { return size_t(h_) * w_; }
  size_t Bytes() const { return Size() * sizeof(T); }
  bool Viewing() const { return !owner_; }
  bool Contiguous() const { return ld_ == h_ || w_ <= 1; }
  T* Buffer() { return data_; }
  const T* LockedBuffer() const { return data_; }
  T* Buffer(Int i, Int j) { return data_ + i + j * ld_; }
  const T* LockedBuffer(Int i, Int j) const { return data_ + i + j * ld_; }
  static constexpr DType dtype() { return DTypeOf<T>::value; }

  // Reallocates only if capacity is insufficient (like El::Matrix::Resize).
  void Resize(Int h, Int w) {
    XPSIM_CHECK(owner_ || data_ == nullptr, "cannot resize a view");
    size_t need = size_t(h) * w;
    if (need > cap_) { Release(); data_ = static_cast<T*>(dev::alloc(std::max<size_t>(need, 1) * sizeof(T))); cap_ = need; owner_ = true; }
    h_ = h; w_ = w; ld_ = std::max<Int>(h, 1);
  }
  void Empty() { Release(); }
  void Attach(Int h, Int w, T* buf, Int ld) { Release(); h_ = h; w_ = w; ld_ = std::max<Int>(ld, 1); data_ = buf; owner_ = false; }
  void LockedAttach(Int h, Int w, const T* buf, Int ld) { Attach(h, w, const_cast<T*>(buf), ld); }
  Matrix View(Int i, Int j, Int h, Int w) { Matrix v; v.Attach(h, w, Buffer(i, j), ld_); return v; }
  Matrix LockedView(Int i, Int j, Int h, Int w) const { Matrix v; v.LockedAttach(h, w, LockedBuffer(i, j), ld_); return v; }
  Matrix View() { return View(0, 0, h_, w_); }

  void Zero() {
    if (Contiguous()) dev::memset_zero(data_, Bytes());
    else for (Int j = 0; j < w_; ++j) dev::memset_zero(Buffer(0, j), size_t(h_) * sizeof(T));
  }
  void MakeUniform(uint64_t seed, float scale) {  // El::MakeUniform analogue (device RNG)
    XPSIM_CHECK(Contiguous(), "MakeUniform on a non-contiguous view");
    dev::fill_random(data_, Size(), dtype(), seed, scale);
  }

 private:
  void Release() {
    if (owner_ && data_) dev::free(data_);
    data_ = nullptr; h_ = w_ = 0; ld_ = 1; cap_ = 0; owner_ = false;
  }
  void swap(Matrix& o) {
    std::swap(data_, o.data_); std::swap(h_, o.h_); std::swap(w_, o.w_); std::swap(ld_, o.ld_);
    std::swap(cap_, o.cap_); std::swap(owner_, o.owner_);
  }
  T* data_ = nullptr;
  Int h_ = 0, w_ = 0, ld_ = 1;
  size_t cap_ = 0;
  bool owner_ = false;
};

// ---------------- local BLAS-like operations (timed + FLOP counted) -------------
// C = alpha op(A) op(B) + beta C  (El::Gemm signature)
template <typename TA, typename TC>
void Gemm(Orientation oA, Orientation oB, float alpha, const Matrix<TA>& A, const Matrix<TA>& B, float beta,
          Matrix<TC>& C, Cat cat) {
  const Int m = C.Height(), n = C.Width();
  const Int k = oA == N ? A.Width() : A.Height();
  XPSIM_CHECK((oA == N ? A.Height() : A.Width()) == m, "Gemm: A/C height mismatch");
  XPSIM_CHECK((oB == N ? B.Width() : B.Height()) == n, "Gemm: B/C width mismatch");
  XPSIM_CHECK((oB == N ? B.Height() : B.Width()) == k, "Gemm: inner dimension mismatch");
  ScopedTimer t(cat);
  dev::gemm(oA, oB, m, n, k, alpha, A.LockedBuffer(), Matrix<TA>::dtype(), A.LDim(), B.LockedBuffer(),
            Matrix<TA>::dtype(), B.LDim(), beta, C.Buffer(), Matrix<TC>::dtype(), C.LDim());
  count_flops(2.0 * double(m) * n * k);
}

template <typename T>
void Copy(const Matrix<T>& A, Matrix<T>& B) {  // El::Copy (local)
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  dev::copy2d(B.Buffer(), B.LDim(), A.LockedBuffer(), A.LDim(), A.Height(), A.Width(), sizeof(T));
}
template <typename T>
void Axpy(float alpha, const Matrix<T>& X, Matrix<T>& Y) {  // Y += alpha X
  XPSIM_CHECK(X.Height() == Y.Height() && X.Width() == Y.Width(), "Axpy: shape mismatch");
  ScopedTimer t(Cat::ELEMWISE);
  if (X.Contiguous() && Y.Contiguous()) dev::axpy(X.Size(), alpha, X.LockedBuffer(), Y.Buffer(), Matrix<T>::dtype());
  else for (Int j = 0; j < X.Width(); ++j) dev::axpy(X.Height(), alpha, X.LockedBuffer(0, j), Y.Buffer(0, j), Matrix<T>::dtype());
}

}  // namespace xpsim
