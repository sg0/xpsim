// GPU windows for the HIER transport, built on the user's VMM layer:
//   vmm::DistributedBuffer = one vmm::MultiGpuBuffer per node (a single VA
//   range whose chunk i is physically on GPU i and mapped read/write on every
//   GPU of the node) + CUDA-aware MPI point-to-point between nodes.
// Worker i of node r owns global chunk r*G + i. Intra-node collectives read
// peer chunks directly (zero copy over NVLink/PCIe P2P); inter-node messages
// are DistributedBuffer::isend/irecv straight from/into the VMM chunks.
#ifdef XPSIM_USE_CUDA
#include <cuda_runtime.h>
#include "comm_internal.hpp"
#include "vmm_mpi.h"

namespace xpsim {
namespace {
class VmmWindows : public WindowSet {
 public:
  VmmWindows(int wpn, size_t bytes, const RuntimeOptions& opt) {
    vmm::DistributedBufferOptions o;
    o.bytesPerDevice = bytes;
    for (int i = 0; i < wpn; ++i) o.devices.push_back(i);
    o.gpuDirectRDMA = opt.gpudirect_rdma;
    o.requireOneRankPerNode = !opt.allow_multi_rank_per_node;
    buf_ = std::make_unique<vmm::DistributedBuffer>(MPI_COMM_WORLD, o);
    void* v = nullptr; int flag = 0;
    MPI_Comm_get_attr(buf_->comm(), MPI_TAG_UB, &v, &flag);
    tagub_ = (flag && v) ? *static_cast<int*>(v) : 32767;
    // The runtime API (cudaMemcpyAsync, kernels) is used on these pointers from
    // every local device context; VMM mappings are UVA so this is valid.
  }
  char* base(int l) override { return static_cast<char*>(buf_->localPtr(buf_->firstLocalChunk() + l)); }
  size_t bytes() const override { return buf_->chunkSize(); }
  void isend(int l, size_t off, size_t b, int dst, int tag, std::vector<MPI_Request>& r) override {
    buf_->isend(buf_->firstLocalChunk() + l, off, b, dst, tag, r);
  }
  void irecv(int l, size_t off, size_t b, int src, int tag, std::vector<MPI_Request>& r) override {
    buf_->irecv(buf_->firstLocalChunk() + l, off, b, src, tag, r);
  }
  int tag_ub() const override { return tagub_; }
  const char* kind() const override { return "CUDA VMM (intra-node zero copy) + CUDA-aware MPI (inter-node)"; }
 private:
  std::unique_ptr<vmm::DistributedBuffer> buf_;
  int tagub_ = 32767;
};
}  // namespace
std::unique_ptr<WindowSet> make_vmm_windows(int wpn, size_t bytes, const RuntimeOptions& opt) {
  return std::make_unique<VmmWindows>(wpn, bytes, opt);
}
}  // namespace xpsim
#endif
