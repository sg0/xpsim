// MoE router: decides, for every local token, which top-k experts it goes to.
// Weights are random (simulation), but the *placement* of the chosen experts
// is controlled so that the expert-parallel all-to-all can be stressed in a
// controlled way:
//
//   uniform       k distinct experts uniformly at random
//   zipf          popularity-skewed experts (hot experts -> load imbalance)
//   local         with prob. --locality pick from the near set (experts on my
//                 rank or my node, --local-scope), else from the rest
//   remote|far    with prob. --locality pick from experts on *other* nodes
//                 (or other ranks with --local-scope rank)
//   distance      P(expert) ~ exp(-beta * hops(my node, expert's node)); the
//                 hop model is --topology flat|ring|fattree; beta < 0 prefers
//                 far nodes
//   node-limited  DeepSeek-V3-style group-limited routing: each token's experts
//                 lie on at most --max-nodes nodes
//   round-robin   perfectly balanced deterministic assignment (baseline)
//
// Optional capacity factor drops assignments that exceed an expert's capacity.
#pragma once
#include <vector>
#include "xpsim/config.hpp"
#include "xpsim/grid.hpp"

namespace xpsim {

struct RoutePlan {
  int n_tok = 0, ep = 1, experts_local = 1;
  // kept assignments, sorted by (destination EP rank, local expert, token)
  std::vector<int> token, expert;
  std::vector<float> weight;
  std::vector<long long> counts;           // [ep][experts_local] tokens I send
  std::vector<size_t> rows_to, rows_off;   // per destination EP rank
  long long dropped = 0;
};

class Router {
 public:
  Router(const Config& cfg, const ProcessGrid& g);
  RoutePlan route(int layer, int mb, int iter, int n_tok) const;
  int owner_rank(int e) const { return e / el_; }   // EP rank owning expert e
  int hops(int node_a, int node_b) const;

 private:
  const Config& cfg_;
  const ProcessGrid& g_;
  int E_, el_, my_node_;
  std::vector<int> owner_node_;  // per expert
};

}  // namespace xpsim
