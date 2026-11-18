// Transformer training step on the distributed grid.
//
// Layer (pre-LN decoder block; norms/biases omitted, they are bandwidth-trivial):
//   attention : QKV = X Wqkv (column-parallel, heads split over TP, GQA)
//               [CP] all-gather K,V over the context-parallel group
//               per (sample, head): S^T = K Q^T, causal softmax, O = P V
//               Y = O Wo (row-parallel) -> TP all-reduce, or reduce-scatter (SP)
//   MLP       : Z = H W1 (column-par.), GELU, U = G W2 (row-par.) -> TP reduce
//   MoE       : router GEMM + Router plan, EP all-to-all dispatch, per-expert
//               FFN, all-to-all combine (tokens sliced over TP ranks)
// Activations are column-major (tokens x features). With sequence parallelism
// a layer's input/output is DistMatrix<T,VC,STAR> (token rows split over TP)
// and the GEMM operands are DistMatrix<T,STAR,STAR>, obtained by Copy
// (all-gather) and returned by Contract (reduce-scatter), as in Elemental.
#include "xpsim/trainer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <fstream>
#include <random>
#include "xpsim/flat_param.hpp"
#include "xpsim/grid.hpp"
#include "xpsim/moe_router.hpp"
#include "xpsim/trace.hpp"
#include <functional>

namespace xpsim {

void report(const Config& cfg, const ProcessGrid& g, double peak_tflops, const ModelCounts& mc);  // report.cpp

ModelCounts model_counts(const Config& c) {
  ModelCounts m;
  const double h = c.hidden, dh = c.head_dim();
  const double attn = h * (c.heads + 2.0 * c.kv_heads) * dh + c.heads * dh * h;
  const double mlp = 2.0 * h * c.ffn, expert = 2.0 * h * c.expert_ffn;
  for (int l = 0; l < c.layers; ++l) {
    m.params_dense += attn;
    if (c.is_moe_layer(l)) { m.params_dense += h * c.experts; m.params_expert += expert * c.experts; m.params_active += attn + h * c.experts + expert * c.topk; }
    else { m.params_dense += mlp; m.params_active += attn + mlp; }
  }
  m.params_dense += 2.0 * c.vocab * h;  // embedding + LM head (untied)
  m.params_active += 2.0 * c.vocab * h - c.vocab * h;  // embedding lookup is not a matmul
  m.params_total = m.params_dense + m.params_expert;
  // training FLOPs per token: 6*(matmul params) + attention scores/values
  // (fwd 4*s*h per layer, x3 for fwd+bwd), full (unmasked) score matrix.
  m.model_flops_per_token = 6.0 * m.params_active + 12.0 * c.layers * double(c.seq) * c.heads * dh;
  return m;
}

namespace {

template <typename T, typename GT>
class Trainer {
  using FP = FlatParam<T, GT>;

 public:
  Trainer(const Config& cfg, ProcessGrid& g) : cfg_(cfg), g_(g), router_(nullptr) {
    h_ = cfg.hidden; dh_ = cfg.head_dim(); tp_ = cfg.tp; cp_ = cfg.cp;
    nq_ = cfg.heads / tp_; nkv_ = cfg.kv_heads / tp_; grp_ = nq_ / nkv_;
    wqkv_ = (nq_ + 2 * nkv_) * dh_; kvw_ = nkv_ * dh_;
    ffn_ = cfg.ffn / tp_; V_ = cfg.vocab / tp_;
    s_l_ = cfg.seq / cp_; b_ = cfg.micro_batch; T_l_ = b_ * s_l_; Ts_ = T_l_ / tp_;
    rows_ = cfg.sp ? Ts_ : T_l_;
    s_k_ = cfg.seq;  // keys per sample after the CP gather
    E_ = cfg.experts; el_ = E_ > 0 ? E_ / cfg.ep : 0; effn_ = cfg.expert_ffn;
    m_ = cfg.microbatches;
    if (E_ > 0) router_ = std::make_unique<Router>(cfg, g);

    // layer -> stage assignment (first stages take the remainder)
    const int base = cfg.layers / cfg.pp, extra = cfg.layers % cfg.pp;
    l0_ = 0;
    for (int s = 0; s < g.me.pp; ++s) l0_ += base + (s < extra);
    nl_ = base + (g.me.pp < extra);
    const int zero = cfg.zero_stage();
    Comm* outer = &g_.hsdp;
    for (int j = 0; j < nl_; ++j) {
      const int L = l0_ + j;
      auto P = std::make_unique<FP>("layer" + std::to_string(L), zero, g_.fsdpGrid.get(), outer, &g_.dp, dense_cats(), cfg.overlap);
      P->add("Wqkv", h_, wqkv_); P->add("Wo", nq_ * dh_, h_);
      if (cfg.is_moe_layer(L)) P->add("Wg", h_, E_);
      else { P->add("W1", h_, ffn_); P->add("W2", ffn_, h_); }
      P->finalize(cfg.seed + 101 * L + 7 * g.me.t, 0.02f);
      dense_.push_back(std::move(P));
      moe_.push_back(cfg.is_moe_layer(L));
      if (cfg.is_moe_layer(L)) {
        auto X = std::make_unique<FP>("experts" + std::to_string(L), zero, g_.edpGrid.get(), nullptr, &g_.edp, expert_cats(), cfg.overlap);
        for (int e = 0; e < el_; ++e) { X->add("W1e", h_, effn_); X->add("W2e", effn_, h_); }
        X->finalize(cfg.seed + 9001 * L + 13 * g.ep_rank, 0.02f);
        expert_.push_back(std::move(X));
      } else expert_.push_back(nullptr);
    }
    if (g.first_stage()) {
      embed_ = std::make_unique<FP>("embedding", zero, g_.fsdpGrid.get(), outer, &g_.dp, dense_cats(), cfg.overlap);
      embed_->add("E", V_, h_); embed_->finalize(cfg.seed + 5 + g.me.t, 0.02f);
    }
    if (g.last_stage()) {
      head_ = std::make_unique<FP>("lm_head", zero, g_.fsdpGrid.get(), outer, &g_.dp, dense_cats(), cfg.overlap);
      head_->add("Wout", h_, V_); head_->finalize(cfg.seed + 3 + g.me.t, 0.02f);
    }
    mb_.resize(m_);
    { dev::MemScope ms("optimizer"); opt_ = static_cast<dev::OptState*>(dev::alloc(sizeof(dev::OptState))); }
    dev::memset_zero(opt_, sizeof(dev::OptState));
  }

 private:
  struct LayerCache {
    Matrix<T> X, Xg, QKV, KVf, P, A, H, Hg, Z, G;
    // MoE
    RoutePlan plan;
    std::vector<long long> rcnt;                  // [ep][el] rows received
    std::vector<size_t> sc, sd, rc, rd;           // a2a element counts/displs (dispatch direction)
    std::vector<std::vector<int>> epos;           // per local expert: rows in the receive buffer
    std::vector<Matrix<T>> Xe, Ze, Ge;
    size_t nrecv = 0;
  };
  struct MbState {
    std::vector<LayerCache> lc;
    Matrix<T> act_in, grad_in, out, dx_send, head_hg, dlt;
    std::vector<int> ids;
    Request act_req, grad_req, send_req;
  };

  // ---------------- TP redistribution helpers (Elemental-style) ----------------
  // [VC,STAR] (sequence-parallel rows) -> [STAR,STAR]
  void gather_tp(const Matrix<T>& X, Matrix<T>& Xg) {
    { dev::MemScope ms("activations"); Xg.Resize(T_l_, X.Width()); }
    DistMatrix<T, VC, STAR> Xd; Xd.SetGrid(*g_.tpGrid); Xd.Attach(T_l_, X.Width(), const_cast<T*>(X.LockedBuffer()), X.LDim());
    DistMatrix<T, STAR, STAR> Xf; Xf.SetGrid(*g_.tpGrid); Xf.Attach(T_l_, X.Width(), Xg.Buffer(), Xg.LDim());
    Copy(Xd, Xf, Cat::TP_AG);
  }
  // partial sums over TP -> all-reduce ([STAR,STAR]) or Contract to [VC,STAR] (SP)
  Matrix<T> reduce_tp(Matrix<T>&& partial) {
    if (!cfg_.sp) { AllReduce(partial, g_.tp, Cat::TP_AR); return std::move(partial); }
    Matrix<T> out; { dev::MemScope ms("activations"); out.Resize(Ts_, partial.Width()); }
    DistMatrix<T, STAR, STAR> Pd; Pd.SetGrid(*g_.tpGrid); Pd.Attach(T_l_, partial.Width(), partial.Buffer(), partial.LDim());
    DistMatrix<T, VC, STAR> Od; Od.SetGrid(*g_.tpGrid); Od.Attach(T_l_, partial.Width(), out.Buffer(), out.LDim());
    Contract(Pd, Od, Cat::TP_RS);
    return out;
  }
  // [VC,STAR] over the TP group from [STAR,STAR] partials of a token slice
  // (MoE outputs when SP is off): all-gather the per-rank token slices.
  void allgather_slices(const Matrix<T>& S, Matrix<T>& F) { gather_tp(S, F); }

  static Matrix<T> act(Int h, Int w) { dev::MemScope ms("activations"); return Matrix<T>(h, w); }

  void bgemm(Orientation oA, Orientation oB, Int m, Int n, Int k, const T* A, Int lda, Int sA, const T* B, Int ldb,
             Int sB, float beta, T* C, Int ldc, Int sC, Int batch) {
    dev::gemm_strided_batched(oA, oB, m, n, k, 1.f, A, Matrix<T>::dtype(), lda, sA, B, Matrix<T>::dtype(), ldb, sB, beta, C,
                              Matrix<T>::dtype(), ldc, sC, batch);
    count_flops(2.0 * double(m) * n * k * batch);
  }

  // ---------------- attention core ----------------
  void attn_fwd(LayerCache& c) {
    const T* K; Int ldk; Int krow;  // K/V base, ld, rows per sample
    if (cp_ > 1) {
      Matrix<T> KV = c.QKV.View(0, nq_ * dh_, T_l_, 2 * kvw_);
      DistMatrix<T, VC, STAR> KVd; KVd.SetGrid(*g_.cpGrid); KVd.Attach(cp_ * T_l_, 2 * kvw_, KV.Buffer(), KV.LDim());
      DistMatrix<T, STAR, STAR> KVg(*g_.cpGrid, cp_ * T_l_, 2 * kvw_);
      Copy(KVd, KVg, Cat::CP_AG);
      c.KVf = act(b_ * s_k_, 2 * kvw_);
      for (int r = 0; r < cp_; ++r)  // rank-major -> sample-major
        for (int bi = 0; bi < b_; ++bi)
          dev::copy2d(c.KVf.Buffer(bi * s_k_ + r * s_l_, 0), c.KVf.LDim(), KVg.Matrix_().Buffer(r * T_l_ + bi * s_l_, 0),
                      KVg.Matrix_().LDim(), s_l_, 2 * kvw_, sizeof(T));
      K = c.KVf.LockedBuffer(); ldk = c.KVf.LDim(); krow = s_k_;
    } else { K = c.QKV.LockedBuffer(0, nq_ * dh_); ldk = T_l_; krow = s_l_; }
    const Int sk = cp_ > 1 ? s_k_ : s_l_;
    const T* V = K + size_t(kvw_) * ldk;
    ScopedTimer t(Cat::ATTN_FWD);
    c.P = act(sk, Int(b_) * nq_ * s_l_);
    c.A = act(T_l_, nq_ * dh_);
    const float scale = 1.f / std::sqrt(float(dh_));
    const Int qoff = Int(g_.me.c) * s_l_;
    for (int bi = 0; bi < b_; ++bi)
      for (int gk = 0; gk < nkv_; ++gk) {
        const int h0 = gk * grp_;
        const T* Kg = K + bi * krow + size_t(gk) * dh_ * ldk;
        const T* Vg = V + bi * krow + size_t(gk) * dh_ * ldk;
        const T* Q = c.QKV.LockedBuffer(bi * s_l_, h0 * dh_);
        T* P = c.P.Buffer(0, (Int(bi) * nq_ + h0) * s_l_);
        bgemm(N, TR, sk, s_l_, dh_, Kg, ldk, 0, Q, T_l_, Int(dh_) * T_l_, 0.f, P, sk, sk * s_l_, grp_);
        for (int hq = 0; hq < grp_; ++hq)
          dev::softmax_cols(P + size_t(hq) * sk * s_l_, sk, s_l_, sk, scale, cfg_.causal, qoff, Matrix<T>::dtype());
        bgemm(TR, N, s_l_, dh_, sk, P, sk, sk * s_l_, Vg, ldk, 0, 0.f, c.A.Buffer(bi * s_l_, h0 * dh_), T_l_,
              Int(dh_) * T_l_, grp_);
      }
  }
  // dA -> dQKV (dQ, dK, dV column blocks)
  void attn_bwd(LayerCache& c, const Matrix<T>& dA, Matrix<T>& dQKV) {
    const Int sk = cp_ > 1 ? s_k_ : s_l_;
    const T* K; Int ldk, krow;
    T* dK; Int lddk;
    Matrix<T> dKVf;
    if (cp_ > 1) {
      K = c.KVf.LockedBuffer(); ldk = c.KVf.LDim(); krow = s_k_;
      dKVf = act(b_ * s_k_, 2 * kvw_); dKVf.Zero(); dK = dKVf.Buffer(); lddk = dKVf.LDim();
    } else {
      K = c.QKV.LockedBuffer(0, nq_ * dh_); ldk = T_l_; krow = s_l_;
      Matrix<T> v = dQKV.View(0, nq_ * dh_, T_l_, 2 * kvw_); v.Zero(); dK = v.Buffer(); lddk = T_l_;
    }
    const T* V = K + size_t(kvw_) * ldk;
    T* dV = dK + size_t(kvw_) * lddk;
    {
      ScopedTimer t(Cat::ATTN_BWD);
      const float scale = 1.f / std::sqrt(float(dh_));
      Matrix<T> dP = act(sk, Int(grp_) * s_l_);
      for (int bi = 0; bi < b_; ++bi)
        for (int gk = 0; gk < nkv_; ++gk) {
          const int h0 = gk * grp_;
          const T* Kg = K + bi * krow + size_t(gk) * dh_ * ldk;
          const T* Vg = V + bi * krow + size_t(gk) * dh_ * ldk;
          T* dKg = dK + bi * krow + size_t(gk) * dh_ * lddk;
          T* dVg = dV + bi * krow + size_t(gk) * dh_ * lddk;
          const T* P = c.P.LockedBuffer(0, (Int(bi) * nq_ + h0) * s_l_);
          const T* dO = dA.LockedBuffer(bi * s_l_, h0 * dh_);
          const T* Q = c.QKV.LockedBuffer(bi * s_l_, h0 * dh_);
          for (int hq = 0; hq < grp_; ++hq)  // dV += P^T dO (accumulate over the GQA group)
            bgemm(N, N, sk, dh_, s_l_, P + size_t(hq) * sk * s_l_, sk, 0, dO + size_t(hq) * dh_ * T_l_, T_l_, 0, 1.f, dVg,
                  lddk, 0, 1);
          bgemm(N, TR, sk, s_l_, dh_, Vg, ldk, 0, dO, T_l_, Int(dh_) * T_l_, 0.f, dP.Buffer(), sk, sk * s_l_, grp_);
          dev::softmax_bwd_cols(P, dP.Buffer(), sk, Int(grp_) * s_l_, sk, scale, Matrix<T>::dtype());
          bgemm(TR, N, s_l_, dh_, sk, dP.Buffer(), sk, sk * s_l_, Kg, ldk, 0, 0.f, dQKV.Buffer(bi * s_l_, h0 * dh_), T_l_,
                Int(dh_) * T_l_, grp_);
          for (int hq = 0; hq < grp_; ++hq)  // dK += dS^T-contraction with Q
            bgemm(N, N, sk, dh_, s_l_, dP.Buffer() + size_t(hq) * sk * s_l_, sk, 0, Q + size_t(hq) * dh_ * T_l_, T_l_, 0,
                  1.f, dKg, lddk, 0, 1);
        }
    }
    if (cp_ > 1) {  // sample-major -> rank-major, then reduce-scatter over CP
      DistMatrix<T, STAR, STAR> G(*g_.cpGrid, cp_ * T_l_, 2 * kvw_);
      for (int r = 0; r < cp_; ++r)
        for (int bi = 0; bi < b_; ++bi)
          dev::copy2d(G.Matrix_().Buffer(r * T_l_ + bi * s_l_, 0), G.Matrix_().LDim(), dKVf.LockedBuffer(bi * s_k_ + r * s_l_, 0),
                      dKVf.LDim(), s_l_, 2 * kvw_, sizeof(T));
      Matrix<T> v = dQKV.View(0, nq_ * dh_, T_l_, 2 * kvw_);
      DistMatrix<T, VC, STAR> L; L.SetGrid(*g_.cpGrid); L.Attach(cp_ * T_l_, 2 * kvw_, v.Buffer(), v.LDim());
      Contract(G, L, Cat::CP_RS);
    }
  }

  // ---------------- MoE ----------------
  Matrix<T> moe_fwd(int j, int L, int mb, int iter, LayerCache& c) {
    FP& D = *dense_[j]; FP& X = *expert_[j];
    const int ep = cfg_.ep;
    Matrix<T> Hs = cfg_.sp ? c.H.View() : c.H.View(Int(g_.me.t) * Ts_, 0, Ts_, h_);
    {  // router GEMM (gating logits); the plan itself comes from the Router policy
      Matrix<T> logits = act(Ts_, E_);
      Gemm(N, N, 1.f, Hs, D.W(2), 0.f, logits, Cat::GEMM_FWD);
    }
    c.plan = router_->route(L, mb, iter, int(Ts_));
    const RoutePlan& P = c.plan;
    c.rcnt.assign(size_t(ep) * el_, 0);
    g_.ep.alltoall_host(P.counts.data(), c.rcnt.data(), el_, Cat::EP_A2A);
    c.sc.assign(ep, 0); c.sd.assign(ep, 0); c.rc.assign(ep, 0); c.rd.assign(ep, 0);
    size_t nrecv = 0;
    for (int r = 0; r < ep; ++r) {
      c.sc[r] = P.rows_to[r] * h_; c.sd[r] = P.rows_off[r] * h_;
      size_t rr = 0; for (int e = 0; e < el_; ++e) rr += size_t(c.rcnt[size_t(r) * el_ + e]);
      c.rd[r] = nrecv * h_; c.rc[r] = rr * h_; nrecv += rr;
    }
    c.nrecv = nrecv;
    const Int nsend = Int(P.token.size());
    Matrix<T> sendbuf = act(std::max<Int>(1, nsend * h_), 1), recvbuf = act(std::max<Int>(1, Int(nrecv) * h_), 1);
    { ScopedTimer t(Cat::MOE_ROUTE);
      dev::copy_rows(Hs.LockedBuffer(), Hs.LDim(), false, P.token.data(), sendbuf.Buffer(), h_, true, nullptr, nsend, h_,
                     nullptr, false, Matrix<T>::dtype()); }
    g_.ep.alltoallv(sendbuf.LockedBuffer(), c.sc, c.sd, recvbuf.Buffer(), c.rc, c.rd, Matrix<T>::dtype(), Cat::EP_A2A);
    // regroup received rows by local expert
    c.epos.assign(el_, {});
    for (int r = 0; r < ep; ++r) {
      size_t base = c.rd[r] / h_;
      for (int e = 0; e < el_; ++e) {
        long long n = c.rcnt[size_t(r) * el_ + e];
        for (long long i = 0; i < n; ++i) c.epos[e].push_back(int(base + i));
        base += size_t(n);
      }
    }
    Matrix<T> backbuf = act(std::max<Int>(1, Int(nrecv) * h_), 1);
    c.Xe.resize(el_); c.Ze.resize(el_); c.Ge.resize(el_);
    double maxload = 0;
    for (int e = 0; e < el_; ++e) {
      const Int ne = Int(c.epos[e].size());
      maxload = std::max(maxload, double(ne));
      c.Xe[e] = act(ne, h_); c.Ze[e] = act(ne, effn_); c.Ge[e] = act(ne, effn_);
      if (ne == 0) continue;
      { ScopedTimer t(Cat::MOE_ROUTE);
        dev::copy_rows(recvbuf.LockedBuffer(), h_, true, c.epos[e].data(), c.Xe[e].Buffer(), ne, false, nullptr, ne, h_,
                       nullptr, false, Matrix<T>::dtype()); }
      Gemm(N, N, 1.f, c.Xe[e], X.W(2 * e), 0.f, c.Ze[e], Cat::GEMM_FWD);
      { ScopedTimer t(Cat::ELEMWISE); dev::gelu_fwd(c.Ze[e].Size(), c.Ze[e].LockedBuffer(), c.Ge[e].Buffer(), Matrix<T>::dtype()); }
      Matrix<T> Ye = act(ne, h_);
      Gemm(N, N, 1.f, c.Ge[e], X.W(2 * e + 1), 0.f, Ye, Cat::GEMM_FWD);
      { ScopedTimer t(Cat::MOE_ROUTE);
        dev::copy_rows(Ye.LockedBuffer(), ne, false, nullptr, backbuf.Buffer(), h_, true, c.epos[e].data(), ne, h_, nullptr,
                       false, Matrix<T>::dtype()); }
    }
    auto& m = metrics();
    if (m.recording && !m.in_recompute) {
      m.moe.max_expert_load += maxload; m.moe.mean_expert_load += el_ ? double(nrecv) / el_ : 0; m.moe.layers += 1;
    }
    Matrix<T> ret = act(std::max<Int>(1, nsend * h_), 1);
    g_.ep.alltoallv(backbuf.LockedBuffer(), c.rc, c.rd, ret.Buffer(), c.sc, c.sd, Matrix<T>::dtype(), Cat::EP_A2A);
    Matrix<T> Us = act(Ts_, h_); Us.Zero();
    { ScopedTimer t(Cat::MOE_ROUTE);
      dev::copy_rows(ret.LockedBuffer(), h_, true, nullptr, Us.Buffer(), Us.LDim(), false, P.token.data(), nsend, h_,
                     P.weight.data(), true, Matrix<T>::dtype()); }
    if (cfg_.sp || tp_ == 1) return Us;
    Matrix<T> U; allgather_slices(Us, U);
    return U;
  }

  Matrix<T> moe_bwd(int j, LayerCache& c, const Matrix<T>& dOut) {
    FP& D = *dense_[j]; FP& X = *expert_[j];
    const RoutePlan& P = c.plan;
    Matrix<T> dOs = cfg_.sp ? const_cast<Matrix<T>&>(dOut).View() : const_cast<Matrix<T>&>(dOut).View(Int(g_.me.t) * Ts_, 0, Ts_, h_);
    Matrix<T> Hs = cfg_.sp ? c.H.View() : c.H.View(Int(g_.me.t) * Ts_, 0, Ts_, h_);
    const Int nsend = Int(P.token.size()), nrecv = Int(c.nrecv);
    Matrix<T> sendg = act(std::max<Int>(1, nsend * h_), 1), recvg = act(std::max<Int>(1, nrecv * h_), 1);
    { ScopedTimer t(Cat::MOE_ROUTE);
      dev::copy_rows(dOs.LockedBuffer(), dOs.LDim(), false, P.token.data(), sendg.Buffer(), h_, true, nullptr, nsend, h_,
                     P.weight.data(), false, Matrix<T>::dtype()); }
    g_.ep.alltoallv(sendg.LockedBuffer(), c.sc, c.sd, recvg.Buffer(), c.rc, c.rd, Matrix<T>::dtype(), Cat::EP_A2A);
    Matrix<T> backg = act(std::max<Int>(1, nrecv * h_), 1);
    for (int e = 0; e < el_; ++e) {
      const Int ne = Int(c.epos[e].size());
      if (ne == 0) continue;
      Matrix<T> dYe = act(ne, h_), dG = act(ne, effn_), dXe = act(ne, h_);
      { ScopedTimer t(Cat::MOE_ROUTE);
        dev::copy_rows(recvg.LockedBuffer(), h_, true, c.epos[e].data(), dYe.Buffer(), ne, false, nullptr, ne, h_, nullptr,
                       false, Matrix<T>::dtype()); }
      Matrix<GT> dW2 = X.dW(2 * e + 1), dW1 = X.dW(2 * e);
      Gemm(TR, N, 1.f, c.Ge[e], dYe, 1.f, dW2, Cat::GEMM_BWD);
      Gemm(N, TR, 1.f, dYe, X.W(2 * e + 1), 0.f, dG, Cat::GEMM_BWD);
      { ScopedTimer t(Cat::ELEMWISE); dev::gelu_bwd(dG.Size(), c.Ze[e].LockedBuffer(), dG.LockedBuffer(), dG.Buffer(), Matrix<T>::dtype()); }
      Gemm(TR, N, 1.f, c.Xe[e], dG, 1.f, dW1, Cat::GEMM_BWD);
      Gemm(N, TR, 1.f, dG, X.W(2 * e), 0.f, dXe, Cat::GEMM_BWD);
      { ScopedTimer t(Cat::MOE_ROUTE);
        dev::copy_rows(dXe.LockedBuffer(), ne, false, nullptr, backg.Buffer(), h_, true, c.epos[e].data(), ne, h_, nullptr,
                       false, Matrix<T>::dtype()); }
    }
    Matrix<T> retg = act(std::max<Int>(1, nsend * h_), 1);
    g_.ep.alltoallv(backg.LockedBuffer(), c.rc, c.rd, retg.Buffer(), c.sc, c.sd, Matrix<T>::dtype(), Cat::EP_A2A);
    Matrix<T> dHs = act(Ts_, h_); dHs.Zero();
    { ScopedTimer t(Cat::MOE_ROUTE);
      dev::copy_rows(retg.LockedBuffer(), h_, true, nullptr, dHs.Buffer(), dHs.LDim(), false, P.token.data(), nsend, h_,
                     nullptr, true, Matrix<T>::dtype()); }
    {  // router backward (gating gradient is random: cost and communication only)
      Matrix<T> dlog = act(Ts_, E_); dlog.MakeUniform(cfg_.seed + 31, 1e-3f);
      Matrix<GT> dWg; { dev::MemScope ms("grads"); dWg.Resize(h_, E_); }
      Gemm(TR, N, 1.f, Hs, dlog, 0.f, dWg, Cat::GEMM_BWD);
      if (tp_ > 1) AllReduce(dWg, g_.tp, Cat::TP_AR);  // router weight is replicated over TP
      Matrix<GT> g = D.dW(2); Axpy(1.f, dWg, g);
      Gemm(N, TR, 1.f, dlog, D.W(2), 1.f, dHs, Cat::GEMM_BWD);
    }
    if (cfg_.sp || tp_ == 1) return dHs;
    Matrix<T> dH; allgather_slices(dHs, dH);
    return dH;
  }

  // ---------------- one transformer layer ----------------
  Matrix<T> fwd_layer(int j, int mb, int iter, LayerCache& c) {
    FP& D = *dense_[j];
    const int L = l0_ + j;
    if (cfg_.sp) gather_tp(c.X, c.Xg);
    const Matrix<T>& Xg = cfg_.sp ? c.Xg : c.X;
    c.QKV = act(T_l_, wqkv_);
    Gemm(N, N, 1.f, Xg, D.W(0), 0.f, c.QKV, Cat::GEMM_FWD);
    attn_fwd(c);
    Matrix<T> Y = act(T_l_, h_);
    Gemm(N, N, 1.f, c.A, D.W(1), 0.f, Y, Cat::GEMM_FWD);
    Matrix<T> Yr = reduce_tp(std::move(Y));
    c.H = act(rows_, h_); Copy(c.X, c.H); Axpy(1.f, Yr, c.H);
    Matrix<T> out = act(rows_, h_); Copy(c.H, out);
    if (!moe_[j]) {
      if (cfg_.sp) gather_tp(c.H, c.Hg);
      const Matrix<T>& Hg = cfg_.sp ? c.Hg : c.H;
      c.Z = act(T_l_, ffn_); c.G = act(T_l_, ffn_);
      Gemm(N, N, 1.f, Hg, D.W(2), 0.f, c.Z, Cat::GEMM_FWD);
      { ScopedTimer t(Cat::ELEMWISE); dev::gelu_fwd(c.Z.Size(), c.Z.LockedBuffer(), c.G.Buffer(), Matrix<T>::dtype()); }
      Matrix<T> U = act(T_l_, h_);
      Gemm(N, N, 1.f, c.G, D.W(3), 0.f, U, Cat::GEMM_FWD);
      Matrix<T> Ur = reduce_tp(std::move(U));
      Axpy(1.f, Ur, out);
    } else {
      Matrix<T> U = moe_fwd(j, L, mb, iter, c);
      Axpy(1.f, U, out);
    }
    return out;
  }

  Matrix<T> bwd_layer(int j, LayerCache& c, const Matrix<T>& dOut) {
    FP& D = *dense_[j];
    Matrix<T> dH = act(rows_, h_); Copy(dOut, dH);  // residual
    if (!moe_[j]) {
      Matrix<T> dUgm; if (cfg_.sp) gather_tp(dOut, dUgm);
      const Matrix<T>& dUg = cfg_.sp ? dUgm : dOut;
      const Matrix<T>& Hg = cfg_.sp ? c.Hg : c.H;
      Matrix<GT> dW2 = D.dW(3), dW1 = D.dW(2);
      Gemm(TR, N, 1.f, c.G, dUg, 1.f, dW2, Cat::GEMM_BWD);
      Matrix<T> dG = act(T_l_, ffn_);
      Gemm(N, TR, 1.f, dUg, D.W(3), 0.f, dG, Cat::GEMM_BWD);
      { ScopedTimer t(Cat::ELEMWISE); dev::gelu_bwd(dG.Size(), c.Z.LockedBuffer(), dG.LockedBuffer(), dG.Buffer(), Matrix<T>::dtype()); }
      Gemm(TR, N, 1.f, Hg, dG, 1.f, dW1, Cat::GEMM_BWD);
      Matrix<T> dHg = act(T_l_, h_);
      Gemm(N, TR, 1.f, dG, D.W(2), 0.f, dHg, Cat::GEMM_BWD);
      Matrix<T> dHr = reduce_tp(std::move(dHg));
      Axpy(1.f, dHr, dH);
    } else {
      Matrix<T> dM = moe_bwd(j, c, dOut);
      Axpy(1.f, dM, dH);
    }
    // attention block
    Matrix<T> dYgm; if (cfg_.sp) gather_tp(dH, dYgm);
    const Matrix<T>& dYg = cfg_.sp ? dYgm : dH;
    Matrix<GT> dWo = D.dW(1), dWqkv = D.dW(0);
    Gemm(TR, N, 1.f, c.A, dYg, 1.f, dWo, Cat::GEMM_BWD);
    Matrix<T> dA = act(T_l_, nq_ * dh_);
    Gemm(N, TR, 1.f, dYg, D.W(1), 0.f, dA, Cat::GEMM_BWD);
    Matrix<T> dQKV = act(T_l_, wqkv_);
    attn_bwd(c, dA, dQKV);
    const Matrix<T>& Xg = cfg_.sp ? c.Xg : c.X;
    Gemm(TR, N, 1.f, Xg, dQKV, 1.f, dWqkv, Cat::GEMM_BWD);
    Matrix<T> dXg = act(T_l_, h_);
    Gemm(N, TR, 1.f, dQKV, D.W(0), 0.f, dXg, Cat::GEMM_BWD);
    Matrix<T> dXr = reduce_tp(std::move(dXg));
    Axpy(1.f, dXr, dH);
    return dH;
  }

  // ---------------- embedding / head ----------------
  void make_tokens(int iter, int mb, std::vector<int>& ids, uint64_t salt) const {
    std::mt19937_64 r(cfg_.seed * 1315423911ull + uint64_t(iter) * 7919 + mb * 104729 + uint64_t(g_.d) * 15485863 +
                      uint64_t(g_.me.c) * 32452843 + salt);
    std::uniform_int_distribution<int> U(0, cfg_.vocab - 1);
    ids.resize(T_l_); for (auto& x : ids) x = U(r);
  }
  Matrix<T> embed_fwd(int iter, int mb, MbState& st) {
    make_tokens(iter, mb, st.ids, 0);
    embed_->gather(false);
    std::vector<int> src(T_l_);
    for (Int i = 0; i < T_l_; ++i) { int id = st.ids[i] - g_.me.t * V_; src[i] = (id >= 0 && id < V_) ? id : -1; }
    Matrix<T> Xf = act(T_l_, h_);
    Matrix<T> E = embed_->W(0);
    { ScopedTimer t(Cat::ELEMWISE);
      dev::copy_rows(E.LockedBuffer(), E.LDim(), false, src.data(), Xf.Buffer(), Xf.LDim(), false, nullptr, T_l_, h_, nullptr,
                     false, Matrix<T>::dtype()); }
    embed_->release();
    return reduce_tp(std::move(Xf));  // vocab-parallel: sum the shards' rows
  }
  void embed_bwd(MbState& st, const Matrix<T>& dX) {
    Matrix<T> dXf; if (cfg_.sp) gather_tp(dX, dXf);
    const Matrix<T>& G = cfg_.sp ? dXf : dX;
    std::vector<int> srow, drow;
    for (Int i = 0; i < T_l_; ++i) { int id = st.ids[i] - g_.me.t * V_; if (id >= 0 && id < V_) { srow.push_back(int(i)); drow.push_back(id); } }
    embed_->begin_backward();
    Matrix<GT> dE = embed_->dW(0);
    // cast the gradient rows to GT and scatter-add (duplicates accumulate)
    Matrix<GT> Gc; { dev::MemScope ms("activations"); Gc.Resize(T_l_, h_); }
    { ScopedTimer t(Cat::ELEMWISE);
      dev::cast(Gc.Buffer(), Matrix<GT>::dtype(), G.LockedBuffer(), Matrix<T>::dtype(), G.Size());
      dev::copy_rows(Gc.LockedBuffer(), Gc.LDim(), false, srow.data(), dE.Buffer(), dE.LDim(), false, drow.data(),
                     Int(srow.size()), h_, nullptr, true, Matrix<GT>::dtype()); }
  }
  void head_fwd(int iter, int mb, MbState& st, const Matrix<T>& Xo) {
    head_->gather(false);
    if (cfg_.sp) gather_tp(Xo, st.head_hg); else { st.head_hg = act(T_l_, h_); Copy(Xo, st.head_hg); }
    st.dlt = act(V_, T_l_);
    Gemm(TR, TR, 1.f, head_->W(0), st.head_hg, 0.f, st.dlt, Cat::GEMM_FWD);  // logits^T (V_l x T_l)
    std::vector<int> tgt; make_tokens(iter, mb, tgt, 99);
    std::vector<int> ltgt(T_l_);
    for (Int i = 0; i < T_l_; ++i) { int id = tgt[i] - g_.me.t * V_; ltgt[i] = (id >= 0 && id < V_) ? id : -1; }
    // vocab-parallel softmax statistics stay on the device (graph-capturable)
    Matrix<float> mx, sm;
    int* tg;
    { dev::MemScope ms("activations"); mx.Resize(T_l_, 1); sm.Resize(T_l_, 1); tg = static_cast<int*>(dev::alloc(T_l_ * sizeof(int))); }
    dev::memcpy_h2d(tg, ltgt.data(), T_l_ * sizeof(int));
    { ScopedTimer t(Cat::ELEMWISE); dev::col_max_d(st.dlt.LockedBuffer(), V_, T_l_, st.dlt.LDim(), Matrix<T>::dtype(), mx.Buffer()); }
    g_.tp.allreduce_max(mx.Buffer(), size_t(T_l_), Cat::LOSS_AR);
    { ScopedTimer t(Cat::ELEMWISE);
      dev::col_sumexp_d(st.dlt.LockedBuffer(), V_, T_l_, st.dlt.LDim(), Matrix<T>::dtype(), mx.LockedBuffer(), sm.Buffer()); }
    g_.tp.allreduce(sm.Buffer(), size_t(T_l_), DType::F32, Cat::LOSS_AR);
    const float scale = float(cfg_.loss_scale / double(cfg_.tokens_per_iter()));
    { ScopedTimer t(Cat::ELEMWISE);
      dev::xent_grad_d(st.dlt.Buffer(), V_, T_l_, st.dlt.LDim(), Matrix<T>::dtype(), mx.LockedBuffer(), sm.LockedBuffer(), tg, scale); }
    dev::free(tg);
    head_->release();
  }
  Matrix<T> head_bwd(MbState& st) {
    head_->gather(false);
    head_->begin_backward();
    Matrix<GT> dW = head_->dW(0);
    Gemm(TR, TR, 1.f, st.head_hg, st.dlt, 1.f, dW, Cat::GEMM_BWD);
    Matrix<T> dHg = act(T_l_, h_);
    Gemm(TR, TR, 1.f, st.dlt, head_->W(0), 0.f, dHg, Cat::GEMM_BWD);
    st.dlt.Empty(); st.head_hg.Empty();
    head_->release();
    return reduce_tp(std::move(dHg));
  }

  // ---------------- pipeline steps ----------------
  void gather_layer(int j, bool async) {
    dense_[j]->gather(async);
    if (expert_[j]) expert_[j]->gather(async);
  }
  void release_layer(int j) { dense_[j]->release(); if (expert_[j]) expert_[j]->release(); }

  void forward(int iter, int mb) {
    MbState& st = mb_[mb];
    st.lc.clear(); st.lc.resize(nl_);
    if (g_.first_stage()) st.lc[0].X = embed_fwd(iter, mb, st);
    else {
      g_.pp.wait(st.act_req);  // includes pipeline-bubble wait
      st.lc[0].X = std::move(st.act_in);
    }
    if (nl_ > 0) gather_layer(0, false);
    for (int j = 0; j < nl_; ++j) {
      if (cfg_.overlap && j + 1 < nl_) gather_layer(j + 1, true);  // FSDP prefetch
      Matrix<T> out = fwd_layer(j, mb, iter, st.lc[j]);
      release_layer(j);
      if (cfg_.recompute) { Matrix<T> keep = std::move(st.lc[j].X); st.lc[j] = LayerCache{}; st.lc[j].X = std::move(keep); }
      if (j + 1 < nl_) st.lc[j + 1].X = std::move(out);
      else st.out = std::move(out);
      if (!cfg_.overlap && j + 1 < nl_) gather_layer(j + 1, false);
    }
    if (g_.last_stage()) { head_fwd(iter, mb, st, st.out); st.out.Empty(); }
    else st.send_req = g_.pp.isend(st.out.LockedBuffer(), st.out.Bytes(), g_.me.pp + 1, 2 * mb, Cat::PP_P2P);
  }

  void backward(int iter, int mb) {
    MbState& st = mb_[mb];
    const bool last = mb == m_ - 1;
    Matrix<T> dOut;
    if (g_.last_stage()) { dOut = head_bwd(st); head_->reduce_grads(last); }
    else { g_.pp.wait(st.grad_req); dOut = std::move(st.grad_in); }
    if (nl_ > 0) gather_layer(nl_ - 1, false);
    for (int j = nl_ - 1; j >= 0; --j) {
      if (cfg_.overlap && j > 0) gather_layer(j - 1, true);
      dense_[j]->begin_backward(); if (expert_[j]) expert_[j]->begin_backward();
      LayerCache& c = st.lc[j];
      if (cfg_.recompute) {
        metrics().in_recompute = true;
        Matrix<T> discard = fwd_layer(j, mb, iter, c);
        metrics().in_recompute = false;
      }
      Matrix<T> dX = bwd_layer(j, c, dOut);
      st.lc[j] = LayerCache{};
      release_layer(j);
      dense_[j]->reduce_grads(last); if (expert_[j]) expert_[j]->reduce_grads(last);
      dOut = std::move(dX);
      if (!cfg_.overlap && j > 0) gather_layer(j - 1, false);
    }
    if (g_.first_stage()) { embed_bwd(st, dOut); embed_->reduce_grads(last); }
    else {
      st.dx_send = std::move(dOut);
      st.send_req = g_.pp.isend(st.dx_send.LockedBuffer(), st.dx_send.Bytes(), g_.me.pp - 1, 2 * mb + 1, Cat::PP_P2P);
    }
  }

  std::vector<FP*> all_params() {
    std::vector<FP*> v;
    if (embed_) v.push_back(embed_.get());
    for (int j = 0; j < nl_; ++j) { v.push_back(dense_[j].get()); if (expert_[j]) v.push_back(expert_[j].get()); }
    if (head_) v.push_back(head_.get());
    return v;
  }

  void train_step(int iter) {
    auto params = all_params();
    for (auto* p : params) p->zero_grads();
    const size_t bytes = size_t(rows_) * h_ * sizeof(T);
    for (int mb = 0; mb < m_; ++mb) {  // pre-post all pipeline receives
      MbState& st = mb_[mb];
      if (!g_.first_stage()) { st.act_in = act(rows_, h_); st.act_req = g_.pp.irecv(st.act_in.Buffer(), bytes, g_.me.pp - 1, 2 * mb, Cat::PP_P2P); }
      if (!g_.last_stage()) { st.grad_in = act(rows_, h_); st.grad_req = g_.pp.irecv(st.grad_in.Buffer(), bytes, g_.me.pp + 1, 2 * mb + 1, Cat::PP_P2P); }
    }
    // schedule
    std::vector<std::pair<bool, int>> ops;
    if (cfg_.schedule == PPSchedule::GPIPE || cfg_.pp == 1) {
      for (int mb = 0; mb < m_; ++mb) ops.push_back({true, mb});
      for (int mb = 0; mb < m_; ++mb) ops.push_back({false, mb});
    } else {
      const int warm = std::min(cfg_.pp - g_.me.pp - 1, m_);
      for (int i = 0; i < warm; ++i) ops.push_back({true, i});
      for (int i = 0; i < m_ - warm; ++i) { ops.push_back({true, warm + i}); ops.push_back({false, i}); }
      for (int i = m_ - warm; i < m_; ++i) ops.push_back({false, i});
    }
    std::vector<Request> sends;
    for (auto& [fwd, mb] : ops) {
      if (fwd) forward(iter, mb); else backward(iter, mb);
      if (mb_[mb].send_req.active()) sends.push_back(std::move(mb_[mb].send_req));
    }
    for (auto* p : params) p->finish_reductions();
    // global gradient norm, clipping and fp16 overflow skip: all device-resident
    double* norm2 = reinterpret_cast<double*>(opt_);  // OptState::norm2 is the first member
    dev::memset_zero(norm2, sizeof(double));
    for (auto* p : params) p->grad_sumsq_acc(norm2);
    Comm::world().allreduce(norm2, 1, DType::F64, Cat::GRADNORM_AR);
    dev::opt_update(opt_, float(cfg_.loss_scale), float(cfg_.clip_grad));
    for (auto* p : params) p->step(float(cfg_.lr), opt_);
    for (auto* p : params) p->wait_post_step();
    g_.pp.waitall(sends);
    for (auto& st : mb_) { st.out.Empty(); st.dx_send.Empty(); }
  }

  const Config& cfg_;
  ProcessGrid& g_;
  std::unique_ptr<Router> router_;
  int h_, dh_, tp_, cp_, nq_, nkv_, grp_, wqkv_, kvw_, ffn_, V_, b_, E_, el_, effn_, m_;
  Int s_l_, T_l_, Ts_, rows_, s_k_;
  int l0_ = 0, nl_ = 0;
  dev::OptState* opt_ = nullptr;
  std::vector<std::unique_ptr<FP>> dense_, expert_;
  std::vector<bool> moe_;
  std::unique_ptr<FP> embed_, head_;
  std::vector<MbState> mb_;

 public:
  dev::OptState* opt_state() const { return opt_; }
  void step(int iter) { train_step(iter); }
  ~Trainer() { dev::free(opt_); }
};

// ------------------------------------------------------------------ driver
dev::OptState read_opt(const dev::OptState* d) {
  dev::OptState h{};
  if (d) dev::memcpy_d2h(&h, d, sizeof(h));
  return h;
}

// Shared iteration loop (live, record + in-process replay, replay from file).
void run_loop(const Config& cfg, const std::function<void(int)>& step, const std::function<dev::OptState*()>& opt) {
  auto& m = metrics();
  auto& world = Comm::world();
  const int total = cfg.warmup + cfg.iters;
  int skip0 = 0;
  for (int it = 0; it < total; ++it) {
    world.barrier();
    m.recording = it >= cfg.warmup;
    if (it == cfg.warmup) {
      if (m.replay.mode == "live") dev::mem_reset_peak();  // replay allocates nothing: keep the recorded step's peak
      skip0 = read_opt(opt()).skipped_total;
    }
    if (dev::is_gpu()) dev::sync();
    const double t0 = now_sec();
    step(it);
    if (dev::is_gpu()) dev::sync();
    const double dt = now_sec() - t0;
    if (m.recording) m.iter_times.push_back(dt);
    if (cfg.verbose) {
      const dev::OptState o = read_opt(opt());
      if (my_worker() == 0)
        std::printf("  iter %3d %s %.4f s  grad-norm %.6e%s  [%s]\n", it, m.recording ? "    " : "(wu)", dt, o.norm,
                    o.skip ? " (skipped)" : "", m.replay.mode == "live" || it <= int(m.replay.trace_iter) ? "live" : m.replay.mode.c_str());
    }
  }
  m.skipped_steps = read_opt(opt()).skipped_total - skip0;
  m.recording = false;
  world.barrier();
}

Replayer::Mode replay_mode(const Config& cfg) {
  return cfg.replay == "graph" ? Replayer::Mode::GRAPH : Replayer::Mode::EAGER;
}
void set_replay_info(const Replayer& r, const Trace& t, const std::string& source, int trace_iter) {
  auto& ri = metrics().replay;
  ri.mode = r.mode() == Replayer::Mode::GRAPH ? "graph" : "eager";
  ri.ops = double(t.step.size()); ri.segments = r.segments(); ri.compute_segments = r.compute_segments();
  ri.graphs = r.graphs(); ri.build_s = r.build_seconds(); ri.source = source; ri.trace_iter = trace_iter;
}
size_t arena_bytes(const Config& cfg, size_t at_least) {
  size_t b = cfg.arena_mb > 0 ? size_t(cfg.arena_mb * 1048576.0)
                              : (dev::is_gpu() ? size_t(double(dev::free_memory()) * 0.85) : (size_t(16) << 30));
  return std::max(b, at_least);
}
std::string trace_file(const std::string& prefix, int worker) { return prefix + ".w" + std::to_string(worker) + ".xpt"; }

void finalize_trace(Trace& t, const Config& cfg, const dev::OptState* opt) {
  t.args = cfg.args;
  t.config_hash = hash_args(cfg.args);
  t.world = world_size(); t.worker = my_worker(); t.backend = dev::is_gpu() ? 1 : 0;
  t.arena_bytes = dev::arena_high_water();
  t.opt_state = dev::arena_offset(opt);
  t.mem_peak.clear();
  for (auto& kv : dev::mem_stats()) t.mem_peak.push_back({kv.first, kv.second.peak});
  t.mem_peak.push_back({"__total__", dev::mem_peak()});
  t.comms.clear();
  for (int i = 0; i < Comm::registry_size(); ++i) t.comms.push_back({int(Comm::from_index(i).role()), Comm::from_index(i).size()});
}
void save_trace(const Trace& t, const Config& cfg) {
  t.save(trace_file(cfg.trace_save, my_worker()));
  if (my_worker() == 0) {
    std::ofstream mf(cfg.trace_save + ".manifest");
    mf << "xpsim-trace 1\nworld " << t.world << "\nhash " << t.config_hash << "\nargs " << t.args.size() << "\n";
    for (auto& a : t.args) mf << a << "\n";
    std::printf("xpsim: trace written: %s.manifest + %d x %s.w<worker>.xpt (%zu step ops, %zu init ops on worker 0)\n",
                cfg.trace_save.c_str(), t.world, cfg.trace_save.c_str(), t.step.size(), t.init.size());
    std::fflush(stdout);
  }
}

template <typename T, typename GT>
int train_t(const Config& cfg) {
  metrics().reset();
  metrics().sync_timers = cfg.sync_timers;
  dev::set_tf32(cfg.tf32);
  ProcessGrid g(cfg);
  double peak = cfg.peak_tflops;
  if (peak <= 0) {  // attainable GEMM rate of one worker (measured on worker 0, shared)
    double p[1] = {0};
    if (my_worker() == 0) p[0] = dev::gemm_peak_tflops(cfg.precision, dev::is_gpu() ? 8192 : 512);
    Comm::world().allreduce_host(p, 1, Cat::MISC_COMM);
    peak = p[0];
  }
  const bool tracing = cfg.replay != "off" || !cfg.trace_save.empty();
  Trace trace;
  if (tracing) {
    dev::arena_begin(arena_bytes(cfg, 0));
    trace::begin(&trace, &trace.init);  // record model construction (parameter / optimizer init)
  }
  {
    Trainer<T, GT> tr(cfg, g);
    if (tracing) trace::end();
    const int rec_iter = tracing ? cfg.warmup - 1 : -1;
    if (tracing) metrics().replay.trace_iter = rec_iter;
    std::unique_ptr<Replayer> rp;
    run_loop(cfg, [&](int it) {
      if (rp) { rp->run_step(); return; }
      if (it == rec_iter) { dev::mem_reset_peak(); trace::begin(&trace, &trace.step); }
      tr.step(it);
      if (it != rec_iter) return;
      trace::end();
      finalize_trace(trace, cfg, tr.opt_state());
      if (!cfg.trace_save.empty()) save_trace(trace, cfg);
      if (cfg.replay != "off") {
        rp = std::make_unique<Replayer>(trace, replay_mode(cfg), cfg.graph_dot);
        set_replay_info(*rp, trace, "in-process", rec_iter);
      }
    }, [&] { return tr.opt_state(); });
    rp.reset();
    report(cfg, g, peak, model_counts(cfg));
  }
  if (tracing) dev::arena_end();
  return 0;
}

// Replay a saved trace: no model is built; the init ops recreate parameters and
// optimizer state in a fresh arena, then every iteration replays the step.
int replay_file(const Config& cfg) {
  metrics().reset();
  metrics().sync_timers = cfg.sync_timers;
  dev::set_tf32(cfg.tf32);
  ProcessGrid g(cfg);
  Trace t = Trace::load(trace_file(cfg.trace_load, my_worker()));
  if (t.world != world_size()) fail("trace was recorded with " + std::to_string(t.world) + " workers, running " + std::to_string(world_size()));
  if (t.worker != my_worker()) fail("trace file belongs to worker " + std::to_string(t.worker));
  if (int(t.comms.size()) > Comm::registry_size()) fail("trace uses more communicators than this configuration creates");
  for (size_t i = 0; i < t.comms.size(); ++i)
    if (t.comms[i].first != int(Comm::from_index(int(i)).role()) || t.comms[i].second != Comm::from_index(int(i)).size())
      fail("communicator " + std::to_string(i) + " differs from the recording (configuration mismatch)");
  if (t.config_hash != hash_args(cfg.args) && my_worker() == 0)
    std::fprintf(stderr, "xpsim: warning: configuration hash differs from the recording\n");
  double peak = cfg.peak_tflops;
  if (peak <= 0) {
    double p[1] = {0};
    if (my_worker() == 0) p[0] = dev::gemm_peak_tflops(cfg.precision, dev::is_gpu() ? 8192 : 512);
    Comm::world().allreduce_host(p, 1, Cat::MISC_COMM);
    peak = p[0];
  }
  dev::arena_begin(arena_bytes(cfg, size_t(t.arena_bytes) + (size_t(64) << 20)));
  {
    Replayer rp(t, replay_mode(cfg), cfg.graph_dot);
    set_replay_info(rp, t, cfg.trace_load, -1);
    rp.run_init();
    for (auto& mp : t.mem_peak) dev::mem_note_peak(mp.first, size_t(mp.second));
    dev::OptState* opt = t.opt_state >= 0 ? reinterpret_cast<dev::OptState*>(dev::arena_base() + t.opt_state) : nullptr;
    run_loop(cfg, [&](int) { rp.run_step(); }, [&] { return opt; });
    report(cfg, g, peak, model_counts(cfg));
  }
  dev::arena_end();
  return 0;
}

}  // namespace

int run_training(const Config& cfg) {
  if (!cfg.trace_load.empty()) return replay_file(cfg);
  switch (cfg.precision) {
    case DType::F32: return train_t<float, float>(cfg);
    case DType::F64: return train_t<double, double>(cfg);
    case DType::BF16: return cfg.grad_fp32 ? train_t<bf16_t, float>(cfg) : train_t<bf16_t, bf16_t>(cfg);
    case DType::F16: return cfg.grad_fp32 ? train_t<fp16_t, float>(cfg) : train_t<fp16_t, fp16_t>(cfg);
  }
  return 1;
}

}  // namespace xpsim
