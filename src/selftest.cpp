// --selftest: verifies every collective, P2P and the DistMatrix
// redistributions against analytically known results, on the active
// transport (MPI or HIER: VMM/shared-window intra-node + MPI inter-node).
#include <cstdio>
#include <vector>
#include "xpsim/dist_matrix.hpp"
#include "xpsim/half_host.hpp"

namespace xpsim {

namespace {
thread_local int g_fail = 0, g_pass = 0;  // per worker
void check(bool ok, const char* what, const Comm& c) {
  if (ok) ++g_pass;
  else { ++g_fail; std::fprintf(stderr, "[worker %d] SELFTEST FAIL: %s (comm %s size %d)\n", my_worker(), what, role_name(c.role()), c.size()); }
}
template <class TT> std::vector<float> down(void* d, size_t n) {
  std::vector<TT> h(n); dev::memcpy_d2h(h.data(), d, n * sizeof(TT));
  std::vector<float> f(n); for (size_t i = 0; i < n; ++i) f[i] = to_float(h[i]); return f;
}
template <class TT> void* up(const std::vector<float>& f) {
  std::vector<TT> h(f.size()); for (size_t i = 0; i < f.size(); ++i) h[i] = from_float<TT>(f[i]);
  void* d = dev::alloc(f.size() * sizeof(TT)); dev::memcpy_h2d(d, h.data(), f.size() * sizeof(TT)); return d;
}

template <class TT> void test_comm(Comm& c, size_t n) {
  const int p = c.size(), r = c.rank();
  const DType dt = DTypeOf<TT>::value;
  const bool lowp = dt == DType::BF16 || dt == DType::F16;
  const float sc = lowp ? 1.f : 1000.f;
  auto mod = [&](size_t i) { return float(lowp ? i % 8 : i); };
  {  // allreduce (blocking + 3 overlapping non-blocking)
    std::vector<float> h(n); for (size_t i = 0; i < n; ++i) h[i] = sc * (r % 4) + mod(i);
    float rs = 0; for (int j = 0; j < p; ++j) rs += sc * (j % 4);
    void* d = up<TT>(h);
    c.allreduce(d, n, dt, Cat::MISC_COMM);
    auto o = down<TT>(d, n); bool ok = true;
    for (size_t i = 0; i < n; ++i) ok &= o[i] == rs + p * mod(i);
    check(ok, "allreduce", c);
    std::vector<void*> ds; std::vector<Request> rq;
    for (int k = 0; k < 3; ++k) { ds.push_back(up<TT>(h)); rq.push_back(c.iallreduce(ds.back(), n, dt, Cat::MISC_COMM)); }
    c.waitall(rq);
    for (auto* x : ds) { auto o2 = down<TT>(x, n); bool ok2 = true; for (size_t i = 0; i < n; ++i) ok2 &= o2[i] == rs + p * mod(i);
      check(ok2, "iallreduce", c); dev::free(x); }
    dev::free(d);
  }
  {  // allgather
    std::vector<float> h(n); for (size_t i = 0; i < n; ++i) h[i] = float(r) * (lowp ? 16 : 100) + mod(i);
    void* s = up<TT>(h); void* d = dev::alloc(n * p * sizeof(TT));
    c.allgather(s, d, n, dt, Cat::MISC_COMM);
    auto o = down<TT>(d, n * p); bool ok = true;
    for (int j = 0; j < p; ++j) for (size_t i = 0; i < n; ++i) ok &= o[j * n + i] == float(j) * (lowp ? 16 : 100) + mod(i);
    check(ok, "allgather", c); dev::free(s); dev::free(d);
  }
  {  // reduce-scatter
    std::vector<float> h(n * p);
    for (int j = 0; j < p; ++j) for (size_t i = 0; i < n; ++i) h[j * n + i] = float(r % 3) + float(j % 5) + mod(i);
    void* s = up<TT>(h); void* d = dev::alloc(n * sizeof(TT));
    c.reduce_scatter(s, d, n, dt, Cat::MISC_COMM);
    auto o = down<TT>(d, n); float rs = 0; for (int j = 0; j < p; ++j) rs += float(j % 3);
    bool ok = true; for (size_t i = 0; i < n; ++i) ok &= o[i] == rs + p * (float(r % 5) + mod(i));
    check(ok, "reduce_scatter", c); dev::free(s); dev::free(d);
  }
}

void test_a2a_p2p(Comm& c) {
  const int p = c.size(), r = c.rank();
  std::vector<size_t> sc(p), sd(p), rc(p), rd(p);
  size_t st = 0, rt = 0;
  for (int j = 0; j < p; ++j) { sc[j] = (r + 2 * j) % 3 + 1; sd[j] = st; st += sc[j]; rc[j] = (j + 2 * r) % 3 + 1; rd[j] = rt; rt += rc[j]; }
  std::vector<float> h(st);
  for (int j = 0; j < p; ++j) for (size_t k = 0; k < sc[j]; ++k) h[sd[j] + k] = float(r * 1000 + j * 10 + k);
  void* s = up<float>(h); void* d = dev::alloc(rt * 4);
  c.alltoallv(s, sc, sd, d, rc, rd, DType::F32, Cat::MISC_COMM);
  auto o = down<float>(d, rt); bool ok = true;
  for (int j = 0; j < p; ++j) for (size_t k = 0; k < rc[j]; ++k) ok &= o[rd[j] + k] == float(j * 1000 + r * 10 + k);
  check(ok, "alltoallv", c); dev::free(s); dev::free(d);
  // ring exchange, two tags in flight
  const size_t n = 1000;
  std::vector<float> a(n), b(n); for (size_t i = 0; i < n; ++i) { a[i] = float(r * 7 + i); b[i] = float(r * 11 + i); }
  void* sa = up<float>(a); void* sb = up<float>(b); void* ra = dev::alloc(n * 4); void* rb = dev::alloc(n * 4);
  const int nx = (r + 1) % p, pv = (r - 1 + p) % p;
  std::vector<Request> q;
  q.push_back(c.irecv(rb, n * 4, pv, 2, Cat::MISC_COMM)); q.push_back(c.irecv(ra, n * 4, pv, 1, Cat::MISC_COMM));
  q.push_back(c.isend(sa, n * 4, nx, 1, Cat::MISC_COMM)); q.push_back(c.isend(sb, n * 4, nx, 2, Cat::MISC_COMM));
  c.waitall(q);
  auto oa = down<float>(ra, n), ob = down<float>(rb, n); bool ok2 = true;
  for (size_t i = 0; i < n; ++i) ok2 &= oa[i] == float(pv * 7 + i) && ob[i] == float(pv * 11 + i);
  check(ok2, "isend/irecv", c);
  dev::free(sa); dev::free(sb); dev::free(ra); dev::free(rb);
}

void test_distmatrix(Comm& c) {
  DistGrid g(c, c.size());
  const int p = c.size();
  const Int H = 6 * p, Wd = 5;
  auto val = [](Int i, Int j) { return float(i * 100 + j); };
  DistMatrix<float, VC, STAR> A(g, H, Wd);
  { std::vector<float> h(size_t(A.LocalHeight()) * Wd);
    for (Int j = 0; j < Wd; ++j) for (Int i = 0; i < A.LocalHeight(); ++i) h[i + j * A.LocalHeight()] = val(A.GlobalRow(i), j);
    dev::memcpy_h2d(A.Matrix_().Buffer(), h.data(), h.size() * 4); }
  DistMatrix<float, STAR, STAR> B(g);
  Copy(A, B, Cat::MISC_COMM);
  std::vector<float> hb(size_t(H) * Wd); dev::memcpy_d2h(hb.data(), B.Matrix_().Buffer(), hb.size() * 4);
  bool ok = true; for (Int j = 0; j < Wd; ++j) for (Int i = 0; i < H; ++i) ok &= hb[i + j * H] == val(i, j);
  check(ok, "DistMatrix [STAR,STAR] <- [VC,STAR]", c);
  DistMatrix<float, VC, STAR> Cm(g);
  Contract(B, Cm, Cat::MISC_COMM);  // every rank holds B -> sum = p * B
  std::vector<float> hc(size_t(Cm.LocalHeight()) * Wd); dev::memcpy_d2h(hc.data(), Cm.Matrix_().Buffer(), hc.size() * 4);
  bool ok2 = true;
  for (Int j = 0; j < Wd; ++j) for (Int i = 0; i < Cm.LocalHeight(); ++i) ok2 &= hc[i + j * Cm.LocalHeight()] == p * val(Cm.GlobalRow(i), j);
  check(ok2, "Contract [VC,STAR] <- [STAR,STAR]", c);
  // [STAR,VC] <- [STAR,STAR] (filter) then back (all-gather of column blocks)
  DistMatrix<float, STAR, STAR> S(g, Wd, H);
  { std::vector<float> h(size_t(Wd) * H); for (Int j = 0; j < H; ++j) for (Int i = 0; i < Wd; ++i) h[i + j * Wd] = val(j, i);
    dev::memcpy_h2d(S.Matrix_().Buffer(), h.data(), h.size() * 4); }
  DistMatrix<float, STAR, VC> D(g);
  Copy(S, D);
  DistMatrix<float, STAR, STAR> E(g);
  Copy(D, E, Cat::MISC_COMM);
  std::vector<float> he(size_t(Wd) * H); dev::memcpy_d2h(he.data(), E.Matrix_().Buffer(), he.size() * 4);
  bool ok3 = true; for (Int j = 0; j < H; ++j) for (Int i = 0; i < Wd; ++i) ok3 &= he[i + j * Wd] == val(j, i);
  check(ok3, "DistMatrix [STAR,VC] round trip", c);
}
}  // namespace

int run_selftest() {
  auto& w = Comm::world();
  std::vector<int> even, odd;
  for (int i = 0; i < world_size(); ++i) (i % 2 ? odd : even).push_back(i);
  Comm half = Comm::create(Role::DP, my_worker() % 2 ? odd : even);
  // strided group that spans nodes when there are several
  for (Comm* c : {&w, &half}) {
    for (size_t n : {size_t(1), size_t(7), size_t(1000), size_t(65536 + 3)}) {
      test_comm<float>(*c, n); test_comm<double>(*c, n); test_comm<bf16_t>(*c, n); test_comm<fp16_t>(*c, n);
    }
    test_a2a_p2p(*c);
    test_distmatrix(*c);
  }
  double r[2] = {double(g_pass), double(g_fail)};
  w.allreduce_host(r, 2, Cat::MISC_COMM);
  if (my_worker() == 0)
    std::printf("selftest: %d workers on %d node(s), transport %s: %.0f checks passed, %.0f failed\n", world_size(),
                num_nodes(), transport() == Transport::MPI ? "mpi" : "hier", r[0], r[1]);
  return r[1] > 0 ? 1 : 0;
}

}  // namespace xpsim
