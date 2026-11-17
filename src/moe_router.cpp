#include "xpsim/moe_router.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>

namespace xpsim {

Router::Router(const Config& cfg, const ProcessGrid& g) : cfg_(cfg), g_(g) {
  E_ = cfg.experts; el_ = cfg.experts / cfg.ep;
  my_node_ = node_of(my_worker());
  owner_node_.resize(E_);
  for (int e = 0; e < E_; ++e) owner_node_[e] = node_of(g.ep_member(owner_rank(e)));
}

int Router::hops(int a, int b) const {
  if (a == b) return 0;
  switch (cfg_.topology) {
    case NetTopology::FLAT: return 1;
    case NetTopology::RING: { int n = num_nodes(), d = std::abs(a - b); return std::min(d, n - d); }
    case NetTopology::FATTREE: return (a / cfg_.nodes_per_switch == b / cfg_.nodes_per_switch) ? 2 : 4;
  }
  return 1;
}

namespace {
uint64_t mix(uint64_t a, uint64_t b) {
  uint64_t z = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull + (a << 6) + (a >> 2));
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// Weighted sampling of one index (weights zeroed for taken experts).
int pick(std::vector<double>& w, std::mt19937_64& rng) {
  double tot = 0; for (double x : w) tot += x;
  if (tot <= 0) return -1;
  double u = std::uniform_real_distribution<double>(0, tot)(rng);
  for (size_t i = 0; i < w.size(); ++i) { u -= w[i]; if (u <= 0 && w[i] > 0) return int(i); }
  for (size_t i = w.size(); i-- > 0;) if (w[i] > 0) return int(i);
  return -1;
}
}  // namespace

RoutePlan Router::route(int layer, int mb, int iter, int n_tok) const {
  const int k = cfg_.topk, ep = cfg_.ep;
  RoutePlan P; P.n_tok = n_tok; P.ep = ep; P.experts_local = el_;
  std::mt19937_64 rng(mix(mix(cfg_.seed, layer * 1000003ull + mb), mix(cfg_.router_static ? 0 : iter, my_worker())));
  const int me_ep = g_.ep_rank;
  auto near = [&](int e) {
    return cfg_.local_scope == "rank" ? owner_rank(e) == me_ep : owner_node_[e] == my_node_;
  };
  // static per-layer base weights
  std::vector<double> base(E_, 1.0);
  if (cfg_.router == RouterPolicy::ZIPF) {
    std::vector<int> perm(E_); std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 prng(mix(cfg_.seed, 77 + layer)); std::shuffle(perm.begin(), perm.end(), prng);
    for (int r = 0; r < E_; ++r) base[perm[r]] = 1.0 / std::pow(double(r + 1), cfg_.zipf_alpha);
  } else if (cfg_.router == RouterPolicy::DISTANCE) {
    for (int e = 0; e < E_; ++e) base[e] = std::exp(-cfg_.distance_beta * hops(my_node_, owner_node_[e]));
  }
  std::vector<int> nodes_with_experts(owner_node_.begin(), owner_node_.end());
  std::sort(nodes_with_experts.begin(), nodes_with_experts.end());
  nodes_with_experts.erase(std::unique(nodes_with_experts.begin(), nodes_with_experts.end()), nodes_with_experts.end());

  struct A { int dest, le, tok, e; float w; };
  std::vector<A> asg; asg.reserve(size_t(n_tok) * k);
  std::uniform_real_distribution<double> U(0, 1);
  std::vector<double> w(E_);
  std::vector<int> chosen(k);
  for (int t = 0; t < n_tok; ++t) {
    switch (cfg_.router) {
      case RouterPolicy::ROUND_ROBIN: {
        long long gt = (long long)me_ep * n_tok + t;
        for (int s = 0; s < k; ++s) chosen[s] = int((gt * k + s) % E_);
        break;
      }
      case RouterPolicy::LOCAL: case RouterPolicy::REMOTE: {
        std::vector<char> taken(E_, 0);
        for (int s = 0; s < k; ++s) {
          bool want_near = (U(rng) < cfg_.locality) == (cfg_.router == RouterPolicy::LOCAL);
          for (int pass = 0; pass < 2; ++pass) {
            for (int e = 0; e < E_; ++e) w[e] = (!taken[e] && near(e) == want_near) ? 1.0 : 0.0;
            int e = pick(w, rng);
            if (e >= 0) { chosen[s] = e; taken[e] = 1; break; }
            want_near = !want_near;  // set exhausted: fall back to the other set
          }
        }
        break;
      }
      case RouterPolicy::NODE_LIMITED: {
        std::vector<int> nodes = nodes_with_experts;
        std::shuffle(nodes.begin(), nodes.end(), rng);
        std::set<int> allowed;
        int avail = 0, i = 0;
        for (; i < int(nodes.size()) && (i < cfg_.max_nodes || avail < k); ++i) {
          allowed.insert(nodes[i]);
          for (int e = 0; e < E_; ++e) avail += owner_node_[e] == nodes[i];
        }
        for (int e = 0; e < E_; ++e) w[e] = allowed.count(owner_node_[e]) ? 1.0 : 0.0;
        for (int s = 0; s < k; ++s) { int e = pick(w, rng); chosen[s] = e; w[e] = 0; }
        break;
      }
      default: {  // UNIFORM, ZIPF, DISTANCE: weighted sampling without replacement
        w = base;
        for (int s = 0; s < k; ++s) { int e = pick(w, rng); chosen[s] = e; w[e] = 0; }
        break;
      }
    }
    // gate weights: softmax of random logits over the k choices
    double z[64], zs = 0; int kk = std::min(k, 64);
    for (int s = 0; s < kk; ++s) { z[s] = std::exp(U(rng)); zs += z[s]; }
    for (int s = 0; s < k; ++s) {
      int e = chosen[s];
      asg.push_back({owner_rank(e), e % el_, t, e, float(z[std::min(s, 63)] / zs)});
    }
  }
  // capacity (per source rank and expert)
  std::vector<long long> load(E_, 0);
  long long cap = cfg_.capacity_factor > 0 ? std::max<long long>(1, (long long)std::ceil(cfg_.capacity_factor * n_tok * k / E_)) : -1;
  std::vector<A> kept; kept.reserve(asg.size());
  for (auto& a : asg) {
    if (cap >= 0 && load[a.e] >= cap) { ++P.dropped; continue; }
    ++load[a.e]; kept.push_back(a);
  }
  std::stable_sort(kept.begin(), kept.end(), [](const A& x, const A& y) { return std::tie(x.dest, x.le) < std::tie(y.dest, y.le); });
  P.counts.assign(size_t(ep) * el_, 0);
  P.rows_to.assign(ep, 0); P.rows_off.assign(ep, 0);
  for (auto& a : kept) {
    P.token.push_back(a.tok); P.expert.push_back(a.e); P.weight.push_back(a.w);
    ++P.counts[size_t(a.dest) * el_ + a.le]; ++P.rows_to[a.dest];
  }
  for (int r = 1; r < ep; ++r) P.rows_off[r] = P.rows_off[r - 1] + P.rows_to[r - 1];
  // statistics
  auto& m = metrics();
  if (m.recording && !m.in_recompute) {
    m.moe.assignments += double(asg.size());
    m.moe.dropped += double(P.dropped);
    for (auto& a : kept) {
      int dw = g_.ep_member(a.dest);
      if (dw == my_worker()) m.moe.to_self += 1;
      else if (node_of(dw) == my_node_) m.moe.intra_node += 1;
      else m.moe.inter_node += 1;
      m.moe.distance += hops(my_node_, node_of(dw));
    }
    std::vector<std::set<int>> tn(n_tok);
    for (auto& a : kept) tn[a.tok].insert(owner_node_[a.e]);
    for (auto& s : tn) if (!s.empty()) { m.moe.tokens += 1; m.moe.token_nodes += double(s.size()); }
  }
  return P;
}

}  // namespace xpsim
