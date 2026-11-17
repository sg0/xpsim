// xpsim entry point.
#include <cstdio>
#include <cstring>
#include <string>
#include "xpsim/config.hpp"
#include "xpsim/trainer.hpp"

using namespace xpsim;
namespace xpsim { int run_selftest(); }

static int worker_main(int argc, char** argv) {
  Config cfg = parse_config(argc, argv);
  if (cfg.selftest) return run_selftest();
  validate(cfg, world_size());
  if (my_worker() == 0) {
    std::printf("xpsim: %d workers, %d node(s), transport %s\n%s\n", world_size(), num_nodes(),
                transport() == Transport::MPI ? "mpi" : "hier", describe(cfg).c_str());
    std::fflush(stdout);
  }
  return run_training(cfg);
}

int xpsim_main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { std::fputs(usage().c_str(), stdout); return 0; }
  Config cfg;
  try {
    cfg = parse_config(argc, argv);  // validated again per worker once the topology is known
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n\n%s", e.what(), usage().c_str());
    return 2;
  }
  return run_workers(argc, argv, cfg.rt, &worker_main);
}

#ifndef XPSIM_MPI_SHIM
int main(int argc, char** argv) { return xpsim_main(argc, argv); }
#endif
