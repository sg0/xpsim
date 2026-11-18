// Internal declarations shared by the transport implementations.
#pragma once
#include <mpi.h>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "xpsim/comm.hpp"

namespace xpsim {

struct OpDesc {
  enum Type { AR, AG, RS, A2AV, SEND, RECV, BARRIER } type = BARRIER;
  const void* sbuf = nullptr;
  void* rbuf = nullptr;
  size_t n = 0;            // elements per block (AG/RS) or total (AR)
  DType dt = DType::F32;
  std::vector<size_t> scounts, sdispls, rcounts, rdispls;  // A2AV, elements
  int peer = -1, tag = 0;  // SEND/RECV: peer = rank in comm, bytes in `n` with dt ignored
};

struct ReqState {
  virtual ~ReqState() = default;
  Cat cat = Cat::MISC_COMM;
  bool waited = false;
  int trace_id = -1;  // request id inside a recorded trace
};

class CommImpl {
 public:
  virtual ~CommImpl() = default;
  Role role = Role::WORLD;
  std::vector<int> members;  // global worker ids, in rank order
  int rank = 0, nodes = 1;
  int reg_index = -1;  // creation index (communicator registry, used by trace replay)
  virtual std::shared_ptr<ReqState> post(const OpDesc& op) = 0;
  virtual void wait(ReqState& r) = 0;
};

// ---------------- windows (HIER transport) ----------------
class WindowSet {
 public:
  virtual ~WindowSet() = default;
  virtual char* base(int local) = 0;  // valid for every local worker (VMM / shared host)
  virtual size_t bytes() const = 0;
  virtual void isend(int local, size_t off, size_t bytes, int dst_node, int tag, std::vector<MPI_Request>& r) = 0;
  virtual void irecv(int local, size_t off, size_t bytes, int src_node, int tag, std::vector<MPI_Request>& r) = 0;
  virtual int tag_ub() const = 0;
  virtual const char* kind() const = 0;
};
std::unique_ptr<WindowSet> make_host_windows(int wpn, size_t bytes);
std::unique_ptr<WindowSet> make_vmm_windows(int wpn, size_t bytes, const RuntimeOptions& opt);  // CUDA builds

class SlotAllocator {
 public:
  explicit SlotAllocator(size_t bytes) { free_[0] = bytes; total_ = bytes; }
  size_t alloc(size_t bytes);  // blocks until space is available
  void release(size_t off, size_t bytes);
  size_t total() const { return total_; }
 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<size_t, size_t> free_;
  size_t total_;
};

class Engine;

struct NodeRuntime {
  RuntimeOptions opt;
  int rank = 0, nranks = 1;
  int world = 1, nodes = 1, wpn = 1;
  std::vector<int> node_of;   // per global worker
  std::vector<int> local_of;  // per global worker
  std::unique_ptr<WindowSet> windows;
  std::vector<std::unique_ptr<SlotAllocator>> slots;
  std::shared_ptr<Engine> engine;
  MPI_Op op_sum_bf16 = MPI_OP_NULL, op_sum_fp16 = MPI_OP_NULL;
};
extern thread_local NodeRuntime* t_rt;
extern thread_local int t_worker;

std::shared_ptr<CommImpl> make_mpi_comm(Role role, const std::vector<int>& members);
std::shared_ptr<CommImpl> make_hier_comm(Role role, const std::vector<int>& members);

// HIER engine entry points
std::shared_ptr<Engine> make_engine(NodeRuntime* rt);
void engine_run(Engine& e, int nworkers);   // main thread; returns when all workers finished
void engine_worker_finished(Engine& e);
void engine_destroy(std::shared_ptr<Engine>& e);

}  // namespace xpsim
