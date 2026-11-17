// FlatParam<T, GT>: one layer's parameters flattened into a single buffer
// (FSDP "flat parameter"), with ZeRO stage 0-3 semantics:
//
//   stage 0 : full params, full grads, full optimizer state; grads all-reduced
//             over the data-parallel group (DDP).
//   stage 1 : full params + grads, optimizer state sharded over the shard group;
//             grads reduce-scattered once per step, updated shards all-gathered.
//   stage 2 : like 1, but grads are reduce-scattered after every microbatch's
//             backward and only the gradient shard is kept.
//   stage 3 : params sharded too (FSDP): all-gathered before each forward and
//             backward use, freed right after (reshard-after-forward).
//
// Sharding uses DistMatrix<T,VC,STAR> (N x 1) over the shard DistGrid; the
// gathered parameter is DistMatrix<T,STAR,STAR>. With HSDP the gradient shard
// is additionally all-reduced over the outer (replicate x CP) group.
#pragma once
#include <string>
#include <type_traits>
#include <vector>
#include "xpsim/dist_matrix.hpp"

namespace xpsim {

struct ParamCats { Cat ag, rs, ar_outer, ar_full, ag_post; };
inline ParamCats dense_cats() { return {Cat::FSDP_AG, Cat::FSDP_RS, Cat::HSDP_AR, Cat::DP_AR, Cat::ZERO_AG}; }
inline ParamCats expert_cats() { return {Cat::EXP_AG, Cat::EXP_RS, Cat::EXP_AR, Cat::EXP_AR, Cat::EXP_AG}; }

template <typename T, typename GT>
class FlatParam {
 public:
  using MT = std::conditional_t<std::is_same_v<T, double>, double, float>;  // master precision
  struct Tensor { std::string name; Int h, w; size_t off; };

  FlatParam(std::string name, int zero, DistGrid* shard, Comm* outer, Comm* full, ParamCats cats, bool overlap)
      : name_(std::move(name)), zero_(zero), shard_(shard), outer_(outer), full_comm_(full), cats_(cats), overlap_(overlap) {}

  void add(const std::string& n, Int h, Int w) { tensors_.push_back({n, h, w, numel_}); numel_ += size_t(h) * w; }

  // Allocate storage and initialize with random master weights.
  void finalize(uint64_t seed, float scale) {
    const int S = shard_size();
    N_ = (numel_ + S * 64 - 1) / (S * 64) * (S * 64);  // pad: divisible by S, 64-element aligned shards
    n_ = N_ / S;
    const size_t mn = zero_ == 0 ? N_ : n_;
    {
      dev::MemScope ms("optimizer");
      master_.Resize(Int(mn), 1); m_.Resize(Int(mn), 1); v_.Resize(Int(mn), 1);
      m_.Zero(); v_.Zero();
      // Each shard is seeded by its global offset so all layouts agree.
      dev::fill_random(master_.Buffer(), mn, master_.dtype(), seed * 7919ull + (zero_ == 0 ? 0 : shard_rank()), scale);
    }
    {
      dev::MemScope ms("params");
      if (zero_ >= 1) { pshard_.SetGrid(*shard_grid()); pshard_.Resize(Int(N_), 1); }
      if (zero_ <= 2) { full_.Resize(Int(N_), 1); }
    }
    {
      dev::MemScope ms("grads");
      if (zero_ <= 1) { gfull_.Resize(Int(N_), 1); gfull_.Zero(); }
      if (zero_ >= 1) { gshard_.Resize(Int(n_), 1); gshard_.Zero(); }
    }
    publish_params();
    if (zero_ == 1 || zero_ == 2) gather_post_step(false);
  }

  size_t numel() const { return numel_; }
  size_t padded() const { return N_; }
  size_t shard_numel() const { return n_; }
  int zero() const { return zero_; }
  const std::string& name() const { return name_; }

  // ---------------- parameter availability ----------------
  void gather(bool async) {  // ZeRO-3: all-gather the full parameter
    if (zero_ < 3 || gathered_) return;
    { dev::MemScope ms("params"); full_.Resize(Int(N_), 1); }
    if (shard_size() == 1) { dev::memcpy_d2d(full_.Buffer(), pshard_.LockedMatrix().LockedBuffer(), N_ * sizeof(T)); gathered_ = true; return; }
    auto& c = shard_grid()->VCComm();
    if (async) pending_ag_ = c.iallgather(pshard_.LockedMatrix().LockedBuffer(), full_.Buffer(), n_, Matrix<T>::dtype(), cats_.ag);
    else c.allgather(pshard_.LockedMatrix().LockedBuffer(), full_.Buffer(), n_, Matrix<T>::dtype(), cats_.ag);
    gathered_ = true;
  }
  void wait_gather() { if (pending_ag_.active()) shard_grid()->VCComm().wait(pending_ag_); }
  void release() {  // ZeRO-3: free the gathered parameter
    if (zero_ < 3) return;
    wait_gather();
    full_.Empty(); gathered_ = false;
  }
  Matrix<T> W(int i) {
    XPSIM_CHECK(zero_ < 3 || gathered_, name_ + ": parameter used without gather()");
    wait_gather();
    auto& t = tensors_[i];
    Matrix<T> v; v.Attach(t.h, t.w, full_.Buffer() + t.off, t.h); return v;
  }

  // ---------------- gradients ----------------
  void begin_backward() {  // ZeRO-2/3: transient full gradient for this microbatch
    if (zero_ >= 2) { dev::MemScope ms("grads"); gfull_.Resize(Int(N_), 1); gfull_.Zero(); }
  }
  Matrix<GT> dW(int i) {
    auto& t = tensors_[i];
    Matrix<GT> v; v.Attach(t.h, t.w, gfull_.Buffer() + t.off, t.h); return v;
  }
  void zero_grads() {
    if (zero_ <= 1) gfull_.Zero();
    if (zero_ >= 1) gshard_.Zero();
  }
  // Called after this layer's backward of a microbatch.
  void reduce_grads(bool last_microbatch) {
    if (zero_ == 0) {
      if (!last_microbatch) return;
      if (full_comm_ && full_comm_->size() > 1) {
        if (overlap_) pending_.push_back({full_comm_->iallreduce(gfull_.Buffer(), N_, Matrix<GT>::dtype(), cats_.ar_full), full_comm_, Matrix<GT>(), false});
        else full_comm_->allreduce(gfull_.Buffer(), N_, Matrix<GT>::dtype(), cats_.ar_full);
      }
      return;
    }
    if (zero_ == 1 && !last_microbatch) return;
    // reduce-scatter the full gradient into a shard
    Matrix<GT> tmp;
    { dev::MemScope ms("comm_buffers"); tmp.Resize(Int(n_), 1); }
    if (shard_size() == 1) dev::memcpy_d2d(tmp.Buffer(), gfull_.LockedBuffer(), n_ * sizeof(GT));
    else if (overlap_) {
      pending_.push_back({shard_grid()->VCComm().ireduce_scatter(gfull_.LockedBuffer(), tmp.Buffer(), n_, Matrix<GT>::dtype(), cats_.rs),
                          &shard_grid()->VCComm(), std::move(tmp), last_microbatch});
      if (zero_ >= 2) keep_gfull_.push_back(std::move(gfull_));  // must survive until the RS completes
      return;
    } else shard_grid()->VCComm().reduce_scatter(gfull_.LockedBuffer(), tmp.Buffer(), n_, Matrix<GT>::dtype(), cats_.rs);
    accumulate_shard(tmp, last_microbatch);
    if (zero_ >= 2) gfull_.Empty();
  }
  // Complete all outstanding gradient communication (before the optimizer).
  void finish_reductions() {
    for (auto& p : pending_) {
      p.comm->wait(p.req);
      if (p.buf.Height() > 0) accumulate_shard(p.buf, p.last);
    }
    pending_.clear();
    keep_gfull_.clear();
    if (zero_ >= 2) gfull_.Empty();
  }
  // Sum of squares of the locally owned gradient, de-duplicated by replication.
  double grad_sumsq() const {
    if (zero_ == 0) return dev::sumsq(gfull_.LockedBuffer(), N_, Matrix<GT>::dtype()) / std::max(1, full_comm_ ? full_comm_->size() : 1);
    return dev::sumsq(gshard_.LockedBuffer(), n_, Matrix<GT>::dtype()) / std::max(1, outer_ ? outer_->size() : 1);
  }

  // ---------------- optimizer ----------------
  void step(int t, float lr, float grad_scale) {
    {
      ScopedTimer tm(Cat::OPTIMIZER);
      const void* g = zero_ == 0 ? static_cast<const void*>(gfull_.LockedBuffer()) : static_cast<const void*>(gshard_.LockedBuffer());
      dev::adamw(master_.Buffer(), Matrix<MT>::dtype(), g, Matrix<GT>::dtype(), m_.Buffer(), v_.Buffer(), master_.Size(),
                 lr, 0.9f, 0.95f, 1e-8f, 0.01f, t, grad_scale);
      publish_params();
    }
    if (zero_ == 1 || zero_ == 2) gather_post_step(overlap_);
  }
  void wait_post_step() { if (pending_post_.active()) shard_grid()->VCComm().wait(pending_post_); }

  // memory footprint (bytes) of persistent state, for reporting
  size_t bytes_params() const { return (zero_ <= 2 ? N_ : n_) * sizeof(T); }

 private:
  struct Pending { Request req; Comm* comm; Matrix<GT> buf; bool last; };
  int shard_size() const { return shard_ ? shard_->Size() : 1; }
  int shard_rank() const { return shard_ ? shard_->VCRank() : 0; }
  DistGrid* shard_grid() const { return shard_; }

  void accumulate_shard(Matrix<GT>& tmp, bool last) {
    { ScopedTimer t(Cat::ELEMWISE); dev::axpy(n_, 1.f, tmp.LockedBuffer(), gshard_.Buffer(), Matrix<GT>::dtype()); }
    if (last && outer_ && outer_->size() > 1) outer_->allreduce(gshard_.Buffer(), n_, Matrix<GT>::dtype(), cats_.ar_outer);
  }
  void publish_params() {  // cast master -> compute-precision params
    if (zero_ == 0) dev::cast(full_.Buffer(), Matrix<T>::dtype(), master_.LockedBuffer(), Matrix<MT>::dtype(), N_);
    else dev::cast(pshard_.Matrix_().Buffer(), Matrix<T>::dtype(), master_.LockedBuffer(), Matrix<MT>::dtype(), n_);
  }
  void gather_post_step(bool async) {  // ZeRO-1/2: all-gather updated shards
    if (shard_size() == 1) { dev::memcpy_d2d(full_.Buffer(), pshard_.LockedMatrix().LockedBuffer(), N_ * sizeof(T)); return; }
    auto& c = shard_grid()->VCComm();
    if (async) pending_post_ = c.iallgather(pshard_.LockedMatrix().LockedBuffer(), full_.Buffer(), n_, Matrix<T>::dtype(), cats_.ag_post);
    else c.allgather(pshard_.LockedMatrix().LockedBuffer(), full_.Buffer(), n_, Matrix<T>::dtype(), cats_.ag_post);
  }

  std::string name_;
  int zero_;
  DistGrid* shard_;
  Comm* outer_;
  Comm* full_comm_;
  ParamCats cats_;
  bool overlap_;
  std::vector<Tensor> tensors_;
  size_t numel_ = 0, N_ = 0, n_ = 0;
  Matrix<MT> master_, m_, v_;
  DistMatrix<T, VC, STAR> pshard_;
  Matrix<T> full_;
  Matrix<GT> gfull_, gshard_;
  std::vector<Matrix<GT>> keep_gfull_;
  std::vector<Pending> pending_;
  Request pending_ag_, pending_post_;
  bool gathered_ = false;
};

}  // namespace xpsim
