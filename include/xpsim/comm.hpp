// Worker-level communication layer.
//
// A *worker* is one simulated accelerator (a GPU, or a CPU process). The model
// code talks to groups of workers through xpsim::Comm. Two transports exist:
//
//  * Transport::MPI  : one MPI rank per worker. Collectives map 1:1 to MPI
//                      (non-blocking) collectives on sub-communicators. Device
//                      buffers are handed to MPI directly when CUDA-aware MPI
//                      is enabled, otherwise staged through pinned host memory.
//
//  * Transport::HIER : one MPI rank per node driving all of the node's workers
//                      as threads. Every worker owns a communication window:
//                      - GPU: a chunk of a vmm::DistributedBuffer (CUDA VMM:
//                        one VA range mapped on every GPU of the node), so
//                        intra-node traffic is zero-copy loads/stores and P2P
//                        DMA over NVLink/PCIe; inter-node traffic is CUDA-aware
//                        MPI_Isend/Irecv issued directly on the VMM chunks.
//                      - CPU: a shared host buffer (same algorithms).
//                      A progress engine on the rank's main thread performs the
//                      rendezvous, the MPI traffic and the intra-node moves and
//                      reductions, so all collectives are truly asynchronous.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "xpsim/metrics.hpp"
#include "xpsim/types.hpp"

namespace xpsim {

enum class Transport { MPI, HIER };

// Group "roles" (also used to build message tags in the HIER transport).
enum class Role : int { WORLD = 0, TP, CP, DP, FSDP, HSDP, PP, EP, EDP, STAGE, TPCP, P2P, COUNT };
const char* role_name(Role r);

struct RuntimeOptions {
  Transport transport = Transport::MPI;
  bool gpu = false;
  bool cuda_aware = true;        // MPI transport: pass device pointers to MPI
  int workers_per_node = 0;      // HIER: 0 = #GPUs (GPU) or 1 (CPU)
  int ranks_per_node = 0;        // MPI: override node detection (0 = MPI_COMM_TYPE_SHARED)
  size_t window_bytes = 256ull << 20;
  bool gpudirect_rdma = true;    // HIER+GPU: RDMA-capable VMM allocations
  bool allow_multi_rank_per_node = false;
  bool sync_collectives = false; // barrier before each collective (separate skew from transfer time)
};

// ---------------- topology (valid on every worker) ----------------
int world_size();
int my_worker();            // global worker id of the calling thread
int num_nodes();
int workers_per_node();
int node_of(int worker);
int local_index(int worker);
Transport transport();
const RuntimeOptions& runtime_options();

struct ReqState;
struct Request {
  std::shared_ptr<ReqState> s;
  bool active() const { return bool(s); }
};

class CommImpl;
class Comm {
 public:
  Comm() = default;
  static Comm create(Role role, const std::vector<int>& members);  // collective over WORLD
  static Comm& world();

  bool valid() const { return bool(impl_); }
  int rank() const;
  int size() const;
  Role role() const;
  const std::vector<int>& members() const;
  int nodes_spanned() const;

  // Device-buffer collectives; counts are elements of dt.
  void allreduce(void* buf, size_t n, DType dt, Cat cat);
  void allgather(const void* send, void* recv, size_t n, DType dt, Cat cat);       // recv: size*n
  void reduce_scatter(const void* send, void* recv, size_t n, DType dt, Cat cat);  // send: size*n
  void alltoallv(const void* send, const std::vector<size_t>& scounts, const std::vector<size_t>& sdispls,
                 void* recv, const std::vector<size_t>& rcounts, const std::vector<size_t>& rdispls, DType dt, Cat cat);
  void barrier(Cat cat = Cat::MISC_COMM);

  Request iallreduce(void* buf, size_t n, DType dt, Cat cat);
  Request iallgather(const void* send, void* recv, size_t n, DType dt, Cat cat);
  Request ireduce_scatter(const void* send, void* recv, size_t n, DType dt, Cat cat);
  Request isend(const void* buf, size_t bytes, int dst, int tag, Cat cat);
  Request irecv(void* buf, size_t bytes, int src, int tag, Cat cat);
  void wait(Request& r);
  void waitall(std::vector<Request>& rs);

  // Small host-side helpers (staged through a device buffer).
  void allreduce_host(double* buf, int n, Cat cat);
  void allreduce_max_host(float* buf, int n, Cat cat);
  void alltoall_host(const long long* send, long long* recv, int n_per, Cat cat);
  void allgather_host(const double* send, double* recv, int n, Cat cat);

 private:
  std::shared_ptr<CommImpl> impl_;
};

// Process-level entry: initializes MPI + topology, runs `worker_main` once per
// worker (on the main thread for MPI, on worker threads for HIER), finalizes.
int run_workers(int argc, char** argv, const RuntimeOptions& opt, int (*worker_main)(int, char**));
// Called by the process-level code before run_workers to obtain the MPI rank
// (configuration parsing happens on every rank).
int process_rank();

}  // namespace xpsim
