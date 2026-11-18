# xpsim -- distributed-memory ML training simulator
#
#   make                         CPU build with $(MPICXX) (MPI required)
#   make CUDA=1 SM=90            + CUDA backend (cuBLAS tensor cores, sm_80+) and
#                                  VMM/CUDA-aware-MPI hierarchical transport
#   make BLAS=openblas|mkl|blis|libsci  vendor CBLAS for CPU GEMMs (or BLAS_LIBS=...)
#   make SHIM=1                  no MPI installed: thread-based MPI shim (testing)
#   make test                    quick functional tests (uses the shim build)
#
# Variables: MPICXX, CXX, CUDA_HOME, NVCC, SM, BLAS, BLAS_LIBS, OPENMP, DEBUG

MPICXX    ?= mpicxx
CXX       ?= g++
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
SM        ?= 80
BLAS      ?= none
OPENMP    ?= 1
DEBUG     ?= 0
SHIM      ?= 0

BUILD := build$(if $(filter 1,$(SHIM)),-shim)$(if $(filter 1,$(CUDA)),-cuda)
BIN   := bin/xpsim$(if $(filter 1,$(SHIM)),-shim)

OPT      := $(if $(filter 1,$(DEBUG)),-O0 -g,-O3 -march=native)
CXXFLAGS := -std=c++17 $(OPT) -Wall -Wextra -Wno-unused-parameter -Iinclude -MMD -MP
LDLIBS   :=

ifeq ($(OPENMP),1)
  CXXFLAGS += -fopenmp
  LDLIBS   += -fopenmp
endif

ifeq ($(SHIM),1)
  CC_MPI   := $(CXX)
  CXXFLAGS += -Itools/mpishim -DXPSIM_MPI_SHIM -pthread
  LDLIBS   += -pthread
else
  CC_MPI   := $(MPICXX)
endif

# --- CPU BLAS ---------------------------------------------------------------
ifeq ($(BLAS),openblas)
  BLAS_LIBS ?= -lopenblas
else ifeq ($(BLAS),mkl)
  BLAS_LIBS ?= -lmkl_rt
else ifeq ($(BLAS),blis)
  BLAS_LIBS ?= -lblis
else ifeq ($(BLAS),system)
  BLAS_LIBS ?= -l:libblas.so.3
endif
ifneq ($(BLAS_LIBS),)
  CXXFLAGS += -DXPSIM_CBLAS
  LDLIBS   += $(BLAS_LIBS)
endif
# Cray LibSci (HPE/Cray PE, e.g. Perlmutter): the CC wrapper links it already
ifeq ($(BLAS),libsci)
  CXXFLAGS += -DXPSIM_CBLAS
endif

SRCS := src/main.cpp src/config.cpp src/device.cpp src/device_cpu.cpp src/metrics.cpp src/comm.cpp \
        src/comm_hier.cpp src/grid.cpp src/moe_router.cpp src/trainer.cpp src/report.cpp src/selftest.cpp src/trace.cpp
ifeq ($(SHIM),1)
  SRCS += tools/mpishim/mpishim.cpp
endif
OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS))

# --- CUDA -------------------------------------------------------------------
ifeq ($(CUDA),1)
  CXXFLAGS  += -DXPSIM_USE_CUDA -I$(CUDA_HOME)/include -Ithird_party/vmm
  NVCCFLAGS := -std=c++17 -O3 -DXPSIM_USE_CUDA -Iinclude -gencode arch=compute_$(SM),code=sm_$(SM) \
               -Xcompiler -fPIC --expt-relaxed-constexpr
  OBJS      += $(BUILD)/src/device_cuda.o $(BUILD)/src/comm_vmm.o
  LDLIBS    += -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lcuda
endif

.PHONY: all clean test help
all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC_MPI) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC_MPI) $(CXXFLAGS) -c $< -o $@

$(BUILD)/src/device_cuda.o: src/device_cuda.cu src/device_impl.hpp include/xpsim/device.hpp include/xpsim/types.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/src/comm_vmm.o: src/comm_vmm.cpp src/comm_internal.hpp third_party/vmm/vmm_mpi.h third_party/vmm/vmm_multigpu.h
	@mkdir -p $(dir $@)
	$(CC_MPI) $(CXXFLAGS) -c $< -o $@

test:
	$(MAKE) SHIM=1 BLAS=$(BLAS)
	./scripts/run_tests.sh bin/xpsim-shim

help:
	@sed -n '2,12p' Makefile

clean:
	rm -rf build build-* bin

-include $(OBJS:.o=.d)
