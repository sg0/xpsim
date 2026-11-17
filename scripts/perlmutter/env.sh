# Source this file (login node or inside a batch job) to get the xpsim
# environment on NERSC Perlmutter (HPE Cray EX, Slingshot-11).
#   source scripts/perlmutter/env.sh gpu     # GPU nodes: 4x A100, 1x EPYC 7763 (64 cores)
#   source scripts/perlmutter/env.sh cpu     # CPU nodes: 2x EPYC 7763 (128 cores, 8 NUMA)
# Module versions follow the current default Cray PE; pin them here if needed.
XP_PART=${1:-${XP_PART:-gpu}}
export XP_PART
XP_ROOT=${XP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export XP_ROOT

module load PrgEnv-gnu >/dev/null 2>&1 || true
module load cray-libsci >/dev/null 2>&1 || true
if [ "$XP_PART" = gpu ]; then
  module load cudatoolkit >/dev/null 2>&1 || true
  module load craype-accel-nvidia80 >/dev/null 2>&1 || true   # links the Cray MPICH GPU (GTL) library
  export CRAY_ACCEL_TARGET=nvidia80
  export MPICH_GPU_SUPPORT_ENABLED=1                          # CUDA-aware Cray MPICH
  export MPICH_OFI_NIC_POLICY=GPU                             # rank -> NIC closest to its GPU
  # CUDA_HOME for the Makefile (cudatoolkit sets CUDA_HOME or CUDATOOLKIT_HOME)
  export CUDA_HOME=${CUDA_HOME:-${CUDATOOLKIT_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")")}}
else
  module load cpu >/dev/null 2>&1 || true
  unset MPICH_GPU_SUPPORT_ENABLED
fi
export XP_BIN_GPU=${XP_BIN_GPU:-$XP_ROOT/bin/xpsim-gpu}
export XP_BIN_CPU=${XP_BIN_CPU:-$XP_ROOT/bin/xpsim-cpu}
# Results go to $SCRATCH by default (fast Lustre file system, purged periodically).
export XP_OUT=${XP_OUT:-${SCRATCH:-$XP_ROOT}/xpsim-results}
mkdir -p "$XP_OUT"
