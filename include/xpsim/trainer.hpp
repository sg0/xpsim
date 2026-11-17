#pragma once
#include "xpsim/config.hpp"

namespace xpsim {
// Runs warm-up + measured iterations on this worker and prints/writes the
// report on worker 0. Collective over WORLD.
int run_training(const Config& cfg);

// Analytical model description used by the report.
struct ModelCounts {
  double params_total = 0, params_dense = 0, params_expert = 0, params_active = 0;
  double model_flops_per_token = 0;  // 6 * active params + attention term (training)
};
ModelCounts model_counts(const Config& cfg);
}  // namespace xpsim
