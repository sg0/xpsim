// DistMatrix<T, U, V>: distributed matrix following El::DistMatrix's design
// (column distribution U, row distribution V over a 2-D process grid), using
// *block* distributions (contiguous blocks of Height/stride rows), which is what
// tensor/sequence/FSDP sharding needs. Redistributions use the same names as
// Elemental and map to collectives:
//
//   [STAR,STAR] <- [VC,STAR] / [STAR,VC] / [MC,MR]   : AllGather
//   [VC,STAR] / [STAR,VC] / [MC,MR] <- [STAR,STAR]   : local filter (no comm)
//   Contract([STAR,STAR] partial sums -> [VC,STAR])  : ReduceScatter (El::Contract)
//   AllReduce([STAR,STAR], comm)                     : AllReduce   (El::AllReduce)
//
// Grid: El::Grid analogue built on a xpsim::Comm. MC = the grid's column
// communicator (size r), MR = row communicator (size c), VC = all r*c ranks in
// column-major order. A 1-D grid (r = size, c = 1) is the common case.
#pragma once
#include <memory>
#include "xpsim/comm.hpp"
#include "xpsim/matrix.hpp"

namespace xpsim {

enum class Dist { MC, MR, VC, STAR };
constexpr Dist MC = Dist::MC, MR = Dist::MR, VC = Dist::VC, STAR = Dist::STAR;

class DistGrid {
 public:
  DistGrid() = default;
  // Collective over WORLD (sub-communicators are created).
  DistGrid(const Comm& vc, int height) : vc_(vc), r_(height), c_(vc.size() / height) {
    XPSIM_CHECK(vc.size() % height == 0, "grid height must divide the communicator size");
    const int q = vc.rank(), row = q % r_, col = q / r_;
    std::vector<int> mc, mr;
    for (int i = 0; i < r_; ++i) mc.push_back(vc.members()[i + col * r_]);
    for (int j = 0; j < c_; ++j) mr.push_back(vc.members()[row + j * r_]);
    if (r_ > 1 && c_ > 1) { mc_ = Comm::create(vc.role(), mc); mr_ = Comm::create(vc.role(), mr); }
    else if (c_ == 1) mc_ = vc;  // 1-D column grid: MC == VC, MR is trivial
    else mr_ = vc;               // 1-D row grid
  }
  int Height() const { return r_; }
  int Width() const { return c_; }
  int Size() const { return r_ * c_; }
  int Row() const { return vc_.rank() % r_; }
  int Col() const { return vc_.rank() / r_; }
  int VCRank() const { return vc_.rank(); }
  Comm& VCComm() { return vc_; }
  Comm& MCComm() { return mc_; }
  Comm& MRComm() { return mr_; }
  const Comm& VCComm() const { return vc_; }

  int Stride(Dist d) const { return d == STAR ? 1 : d == VC ? Size() : d == MC ? r_ : c_; }
  int Shift(Dist d) const { return d == STAR ? 0 : d == VC ? VCRank() : d == MC ? Row() : Col(); }

 private:
  Comm vc_, mc_, mr_;
  int r_ = 1, c_ = 1;
};

template <typename T, Dist U, Dist V>
class DistMatrix {
 public:
  DistMatrix() = default;
  explicit DistMatrix(DistGrid& g, Int h = 0, Int w = 0) : g_(&g) { Resize(h, w); }
  DistMatrix(DistMatrix&&) noexcept = default;
  DistMatrix& operator=(DistMatrix&&) noexcept = default;

  void SetGrid(DistGrid& g) { g_ = &g; }
  DistGrid& Grid() const { return *g_; }
  Int Height() const { return h_; }
  Int Width() const { return w_; }
  int ColStride() const { return g_->Stride(U); }
  int RowStride() const { return g_->Stride(V); }
  int ColShift() const { return g_->Shift(U); }
  int RowShift() const { return g_->Shift(V); }
  Int LocalHeight() const { return h_ / ColStride(); }
  Int LocalWidth() const { return w_ / RowStride(); }
  Int GlobalRow(Int iLoc) const { return ColShift() * LocalHeight() + iLoc; }
  Int GlobalCol(Int jLoc) const { return RowShift() * LocalWidth() + jLoc; }
  Matrix<T>& Matrix_() { return local_; }  // El: A.Matrix()
  const Matrix<T>& LockedMatrix() const { return local_; }
  static constexpr Dist ColDist() { return U; }
  static constexpr Dist RowDist() { return V; }

  void Resize(Int h, Int w) {
    h_ = h; w_ = w;
    XPSIM_CHECK(h % ColStride() == 0, "DistMatrix: height " + std::to_string(h) + " not divisible by column stride " + std::to_string(ColStride()));
    XPSIM_CHECK(w % RowStride() == 0, "DistMatrix: width " + std::to_string(w) + " not divisible by row stride " + std::to_string(RowStride()));
    local_.Resize(LocalHeight(), LocalWidth());
  }
  // View an existing local buffer (e.g. a slice of an FSDP flat parameter).
  void Attach(Int h, Int w, T* buf, Int ld) {
    h_ = h; w_ = w;
    local_.Attach(LocalHeight(), LocalWidth(), buf, ld);
  }
  void MakeUniform(uint64_t seed, float scale) {
    // Seed by global block so every distribution of the same matrix agrees.
    local_.MakeUniform(seed * 1000003ull + uint64_t(ColShift()) * 1009ull + uint64_t(RowShift()), scale);
  }

 private:
  DistGrid* g_ = nullptr;
  Int h_ = 0, w_ = 0;
  Matrix<T> local_;
};

// ------------------------------------------------------------ redistributions
namespace detail {
// Pack a column-major (h x w, ld) matrix into p row-blocks, each contiguous.
template <typename T>
void pack_row_blocks(const Matrix<T>& A, int p, T* out) {
  const Int hb = A.Height() / p;
  for (int r = 0; r < p; ++r)
    dev::copy2d(out + size_t(r) * hb * A.Width(), hb, A.LockedBuffer(r * hb, 0), A.LDim(), hb, A.Width(), sizeof(T));
}
template <typename T>
void unpack_row_blocks(const T* in, int p, Matrix<T>& A) {
  const Int hb = A.Height() / p;
  for (int r = 0; r < p; ++r)
    dev::copy2d(A.Buffer(r * hb, 0), A.LDim(), in + size_t(r) * hb * A.Width(), hb, hb, A.Width(), sizeof(T));
}
template <typename T> struct Scratch {
  T* p = nullptr;
  explicit Scratch(size_t n) { dev::MemScope ms("comm_buffers"); p = static_cast<T*>(dev::alloc(n * sizeof(T))); }
  ~Scratch() { dev::free(p); }
};
}  // namespace detail

// [STAR,STAR] <- [VC,STAR] : AllGather of row blocks
template <typename T>
void Copy(const DistMatrix<T, VC, STAR>& A, DistMatrix<T, STAR, STAR>& B, Cat cat) {
  auto& g = A.Grid(); const int p = g.Size();
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  const Matrix<T>& Al = A.LockedMatrix(); Matrix<T>& Bl = B.Matrix_();
  if (p == 1) { Copy(Al, Bl); return; }
  const size_t blk = size_t(Al.Height()) * Al.Width();
  if (A.Width() == 1 && Al.Contiguous() && Bl.Contiguous()) {  // vectors: gather straight into place
    g.VCComm().allgather(Al.LockedBuffer(), Bl.Buffer(), blk, Matrix<T>::dtype(), cat);
    return;
  }
  detail::Scratch<T> s(blk * (p + 1));
  dev::copy2d(s.p, Al.Height(), Al.LockedBuffer(), Al.LDim(), Al.Height(), Al.Width(), sizeof(T));
  g.VCComm().allgather(s.p, s.p + blk, blk, Matrix<T>::dtype(), cat);
  detail::unpack_row_blocks(s.p + blk, p, Bl);
}
// [STAR,STAR] <- [STAR,VC] : AllGather of column blocks (contiguous, no packing)
template <typename T>
void Copy(const DistMatrix<T, STAR, VC>& A, DistMatrix<T, STAR, STAR>& B, Cat cat) {
  auto& g = A.Grid();
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  const Matrix<T>& Al = A.LockedMatrix(); Matrix<T>& Bl = B.Matrix_();
  if (g.Size() == 1) { Copy(Al, Bl); return; }
  XPSIM_CHECK(Al.Contiguous() && Bl.Contiguous(), "[STAR,VC] gather needs contiguous storage");
  g.VCComm().allgather(Al.LockedBuffer(), Bl.Buffer(), Al.Size(), Matrix<T>::dtype(), cat);
}
// [STAR,STAR] <- [MC,MR] : AllGather of 2-D blocks over VC
template <typename T>
void Copy(const DistMatrix<T, MC, MR>& A, DistMatrix<T, STAR, STAR>& B, Cat cat) {
  auto& g = A.Grid(); const int r = g.Height(), c = g.Width(), p = r * c;
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  const Matrix<T>& Al = A.LockedMatrix(); Matrix<T>& Bl = B.Matrix_();
  const Int hb = Al.Height(), wb = Al.Width(); const size_t blk = size_t(hb) * wb;
  detail::Scratch<T> s(blk * (p + 1));
  dev::copy2d(s.p, hb, Al.LockedBuffer(), Al.LDim(), hb, wb, sizeof(T));
  g.VCComm().allgather(s.p, s.p + blk, blk, Matrix<T>::dtype(), cat);
  for (int q = 0; q < p; ++q)
    dev::copy2d(Bl.Buffer((q % r) * hb, (q / r) * wb), Bl.LDim(), s.p + blk * (q + 1), hb, hb, wb, sizeof(T));
}
// Distributed <- [STAR,STAR] : local filter
template <typename T, Dist U, Dist V>
void Copy(const DistMatrix<T, STAR, STAR>& A, DistMatrix<T, U, V>& B) {
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  Matrix<T>& Bl = B.Matrix_();
  const Matrix<T>& Al = A.LockedMatrix();
  dev::copy2d(Bl.Buffer(), Bl.LDim(), Al.LockedBuffer(B.ColShift() * Bl.Height(), B.RowShift() * Bl.Width()), Al.LDim(),
              Bl.Height(), Bl.Width(), sizeof(T));
}
// El::Contract: sum the [STAR,STAR] partial results and scatter row blocks.
template <typename T>
void Contract(const DistMatrix<T, STAR, STAR>& A, DistMatrix<T, VC, STAR>& B, Cat cat) {
  auto& g = B.Grid(); const int p = g.Size();
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  const Matrix<T>& Al = A.LockedMatrix(); Matrix<T>& Bl = B.Matrix_();
  if (p == 1) { Copy(Al, Bl); return; }
  const size_t blk = size_t(Bl.Height()) * Bl.Width();
  detail::Scratch<T> s(blk * (p + 1));
  detail::pack_row_blocks(Al, p, s.p + blk);
  g.VCComm().reduce_scatter(s.p + blk, s.p, blk, Matrix<T>::dtype(), cat);
  dev::copy2d(Bl.Buffer(), Bl.LDim(), s.p, Bl.Height(), Bl.Height(), Bl.Width(), sizeof(T));
}
template <typename T>
void Contract(const DistMatrix<T, STAR, STAR>& A, DistMatrix<T, STAR, VC>& B, Cat cat) {
  auto& g = B.Grid();
  if (B.Height() != A.Height() || B.Width() != A.Width()) B.Resize(A.Height(), A.Width());
  const Matrix<T>& Al = A.LockedMatrix(); Matrix<T>& Bl = B.Matrix_();
  if (g.Size() == 1) { Copy(Al, Bl); return; }
  XPSIM_CHECK(Al.Contiguous() && Bl.Contiguous(), "[STAR,VC] contract needs contiguous storage");
  g.VCComm().reduce_scatter(Al.LockedBuffer(), Bl.Buffer(), Bl.Size(), Matrix<T>::dtype(), cat);
}
// El::AllReduce on the local data of a replicated matrix.
template <typename T>
void AllReduce(DistMatrix<T, STAR, STAR>& A, Comm& comm, Cat cat) {
  Matrix<T>& Al = A.Matrix_();
  XPSIM_CHECK(Al.Contiguous(), "AllReduce needs contiguous storage");
  comm.allreduce(Al.Buffer(), Al.Size(), Matrix<T>::dtype(), cat);
}
template <typename T>
void AllReduce(Matrix<T>& A, Comm& comm, Cat cat) {
  XPSIM_CHECK(A.Contiguous(), "AllReduce needs contiguous storage");
  comm.allreduce(A.Buffer(), A.Size(), Matrix<T>::dtype(), cat);
}

}  // namespace xpsim
