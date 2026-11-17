#include "xpsim/grid.hpp"
#include <functional>
#include <sstream>

namespace xpsim {

int ProcessGrid::worker(const Coords& x) const {
  return (((x.pp * cfg.dp + x.dr) * cfg.fsdp + x.ds) * cfg.cp + x.c) * cfg.tp + x.t;
}
Coords ProcessGrid::coords(int w) const {
  Coords x;
  x.t = w % cfg.tp; w /= cfg.tp;
  x.c = w % cfg.cp; w /= cfg.cp;
  x.ds = w % cfg.fsdp; w /= cfg.fsdp;
  x.dr = w % cfg.dp; w /= cfg.dp;
  x.pp = w;
  return x;
}

ProcessGrid::ProcessGrid(const Config& c) : cfg(c) {
  me = coords(my_worker());
  d = me.dr * c.fsdp + me.ds;
  ep_rank = d % c.ep;
  const int W = world_size();
  // Collect the members of my group for a predicate "same group as me".
  auto group = [&](std::function<bool(const Coords&)> same) {
    std::vector<int> m;
    for (int w = 0; w < W; ++w) if (same(coords(w))) m.push_back(w);
    return m;
  };
  const Coords& M = me;
  auto dof = [&](const Coords& x) { return x.dr * c.fsdp + x.ds; };
  // Creation order is identical on every worker (required for MPI_Comm_split).
  tp = Comm::create(Role::TP, group([&](const Coords& x) { return x.pp == M.pp && x.dr == M.dr && x.ds == M.ds && x.c == M.c; }));
  cp = Comm::create(Role::CP, group([&](const Coords& x) { return x.pp == M.pp && x.dr == M.dr && x.ds == M.ds && x.t == M.t; }));
  fsdp = Comm::create(Role::FSDP, group([&](const Coords& x) { return x.pp == M.pp && x.dr == M.dr && x.c == M.c && x.t == M.t; }));
  hsdp = Comm::create(Role::HSDP, group([&](const Coords& x) { return x.pp == M.pp && x.ds == M.ds && x.t == M.t; }));
  dp = Comm::create(Role::DP, group([&](const Coords& x) { return x.pp == M.pp && x.t == M.t; }));
  pp = Comm::create(Role::PP, group([&](const Coords& x) { return x.dr == M.dr && x.ds == M.ds && x.c == M.c && x.t == M.t; }));
  ep = Comm::create(Role::EP, group([&](const Coords& x) {
    return x.pp == M.pp && x.c == M.c && x.t == M.t && dof(x) / c.ep == d / c.ep; }));
  edp = Comm::create(Role::EDP, group([&](const Coords& x) { return x.pp == M.pp && dof(x) % c.ep == ep_rank; }));
  tpGrid = std::make_unique<DistGrid>(tp, tp.size());
  cpGrid = std::make_unique<DistGrid>(cp, cp.size());
  fsdpGrid = std::make_unique<DistGrid>(fsdp, fsdp.size());
  edpGrid = std::make_unique<DistGrid>(edp, edp.size());
}

int ProcessGrid::ep_member(int e) const { return ep.members()[e]; }

std::string ProcessGrid::describe_groups() const {
  std::ostringstream o;
  auto show = [&](const char* n, const Comm& c) {
    o << "  " << n << " size=" << c.size() << " nodes=" << c.nodes_spanned() << " members=[";
    for (size_t i = 0; i < c.members().size() && i < 16; ++i) o << (i ? "," : "") << c.members()[i];
    if (c.members().size() > 16) o << ",...";
    o << "]\n";
  };
  show("tp  ", tp); show("cp  ", cp); show("fsdp", fsdp); show("hsdp", hsdp); show("dp  ", dp);
  show("pp  ", pp); show("ep  ", ep); show("edp ", edp);
  return o.str();
}

}  // namespace xpsim
