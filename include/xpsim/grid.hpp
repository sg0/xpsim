// ProcessGrid: the 5-D logical layout of workers and all parallelism groups.
//
//   worker = ((((pp * DPR + dr) * DPS + ds) * CP + c) * TP + t)
//
// TP is innermost (intra-node, highest bandwidth), then CP, the FSDP shard
// dimension, the DP replicate dimension, and PP outermost. EP is folded into
// the data-parallel index d = dr*DPS + ds: EP group = EP consecutive d values
// with the same (pp, c, t). Expert gradients are reduced over the expert-data-
// parallel group = every worker of the stage with the same EP rank.
#pragma once
#include <memory>
#include "xpsim/config.hpp"
#include "xpsim/dist_matrix.hpp"

namespace xpsim {

struct Coords { int pp = 0, dr = 0, ds = 0, c = 0, t = 0; };

class ProcessGrid {
 public:
  explicit ProcessGrid(const Config& cfg);  // collective over WORLD

  const Config& cfg;
  Coords me;
  int d = 0;       // data-parallel index dr*DPS + ds
  int ep_rank = 0; // index within the EP group

  Comm tp, cp, fsdp, hsdp, dp, pp, ep, edp;
  // Elemental-style 1-D grids for the DistMatrix-based layers.
  std::unique_ptr<DistGrid> tpGrid, cpGrid, fsdpGrid, edpGrid;

  int worker(const Coords& x) const;
  Coords coords(int worker) const;
  int ep_member(int e) const;  // global worker of EP rank e in my EP group
  bool first_stage() const { return me.pp == 0; }
  bool last_stage() const { return me.pp == cfg.pp - 1; }
  std::string describe_groups() const;
};

}  // namespace xpsim
