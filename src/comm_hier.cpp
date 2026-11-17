// HIER transport: one MPI rank per node, workers as threads, per-worker
// communication windows (CUDA VMM chunks on GPUs, shared host memory on CPUs)
// and a progress engine on the rank's main thread.
//
// Collective algorithms (p members, block B bytes):
//   AllGather     : every member stages its block in its window. Intra-node
//                   peers read it directly (zero copy), remote peers get it
//                   with one MPI message per (src, dst) pair, straight from
//                   the source window into the destination window.
//   ReduceScatter : like AllGather on the p input blocks; the owner of block i
//                   reduces the p contributions with one kernel that reads the
//                   local peers' windows in place.
//   AllReduce     : ReduceScatter into a window block + AllGather of the
//                   reduced blocks (bandwidth-optimal 2(p-1)/p volume).
//   AllToAllv     : direct pairwise exchange (zero-copy intra, MPI inter).
//   P2P           : intra-node matched send/recv copy, inter-node MPI.
// All inter-node messages are posted in (src member, dst member) order with
// tags unique per (role, src GPU, dst GPU, sequence, phase).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <list>
#include <set>
#include <tuple>
#include "comm_internal.hpp"

namespace xpsim {

// ------------------------------------------------------------- slot allocator
static size_t round256(size_t b) { return (b + 255) / 256 * 256; }

size_t SlotAllocator::alloc(size_t bytes) {
  if (bytes == 0) return 0;
  bytes = round256(bytes);
  XPSIM_CHECK(bytes <= total_, "communication window too small for one message: need " + std::to_string(bytes >> 20) +
                                   " MiB, window is " + std::to_string(total_ >> 20) + " MiB (raise --window-mb)");
  std::unique_lock<std::mutex> lk(mu_);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (true) {
    for (auto it = free_.begin(); it != free_.end(); ++it)
      if (it->second >= bytes) {
        size_t off = it->first, len = it->second;
        free_.erase(it);
        if (len > bytes) free_[off + bytes] = len - bytes;
        return off;
      }
    if (cv_.wait_until(lk, deadline) == std::cv_status::timeout)
      fail("communication window exhausted for 120 s (too many in-flight collectives); raise --window-mb");
  }
}
void SlotAllocator::release(size_t off, size_t bytes) {
  if (bytes == 0) return;
  bytes = round256(bytes);
  std::lock_guard<std::mutex> lk(mu_);
  auto it = free_.emplace(off, bytes).first;
  auto nx = std::next(it);
  if (nx != free_.end() && it->first + it->second == nx->first) { it->second += nx->second; free_.erase(nx); }
  if (it != free_.begin()) {
    auto pv = std::prev(it);
    if (pv->first + pv->second == it->first) { pv->second += it->second; free_.erase(it); }
  }
  cv_.notify_all();
}

// ------------------------------------------------------------- host windows
namespace {
constexpr size_t kMaxMsg = size_t(1) << 30;
class HostWindows : public WindowSet {
 public:
  HostWindows(int wpn, size_t bytes) : wpn_(wpn), bytes_(round256(bytes)) {
    mem_ = static_cast<char*>(std::aligned_alloc(256, bytes_ * wpn_));
    XPSIM_CHECK(mem_, "host window allocation failed");
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
    void* v = nullptr; int flag = 0;
    MPI_Comm_get_attr(comm_, MPI_TAG_UB, &v, &flag);
    tagub_ = (flag && v) ? *static_cast<int*>(v) : 32767;
  }
  ~HostWindows() override { std::free(mem_); int fin = 0; MPI_Finalized(&fin); if (!fin) MPI_Comm_free(&comm_); }
  char* base(int l) override { return mem_ + size_t(l) * bytes_; }
  size_t bytes() const override { return bytes_; }
  void isend(int l, size_t off, size_t b, int dst, int tag, std::vector<MPI_Request>& r) override {
    size_t done = 0;
    do { size_t n = std::min(kMaxMsg, b - done); r.emplace_back();
         MPI_Isend(base(l) + off + done, int(n), MPI_BYTE, dst, tag, comm_, &r.back()); done += n; } while (done < b);
  }
  void irecv(int l, size_t off, size_t b, int src, int tag, std::vector<MPI_Request>& r) override {
    size_t done = 0;
    do { size_t n = std::min(kMaxMsg, b - done); r.emplace_back();
         MPI_Irecv(base(l) + off + done, int(n), MPI_BYTE, src, tag, comm_, &r.back()); done += n; } while (done < b);
  }
  int tag_ub() const override { return tagub_; }
  const char* kind() const override { return "shared host memory"; }
 private:
  int wpn_; size_t bytes_; char* mem_ = nullptr; MPI_Comm comm_ = MPI_COMM_NULL; int tagub_ = 32767;
};
}  // namespace

std::unique_ptr<WindowSet> make_host_windows(int wpn, size_t bytes) { return std::make_unique<HostWindows>(wpn, bytes); }
#ifndef XPSIM_USE_CUDA
std::unique_ptr<WindowSet> make_vmm_windows(int, size_t, const RuntimeOptions&) {
  fail("VMM windows need a CUDA build (make CUDA=1)");
}
#endif

// ------------------------------------------------------------- jobs + engine
struct HJob : ReqState {
  OpDesc op;
  int worker = 0, gidx = 0, local = 0, leader = 0;
  Role role = Role::WORLD;
  uint64_t seq = 0;
  std::vector<int> members;
  size_t es = 1;
  size_t soff = 0, sb = 0, roff = 0, rb = 0, moff = 0, mb = 0;
  bool done = false;  // guarded by Engine::mu
};

class Engine {
 public:
  explicit Engine(NodeRuntime* rt) : rt_(rt) {
    const long G = rt->wpn, R = long(Role::COUNT);
    long room = long(rt->windows->tag_ub()) / (2 * R * G * G);
    nseq_ = 1; while (nseq_ * 2 <= room) nseq_ *= 2;
    XPSIM_CHECK(room >= 2, "MPI_TAG_UB too small for " + std::to_string(G) + " workers per node");
    if (nseq_ < 64 && rt->rank == 0)
      std::fprintf(stderr, "xpsim: warning: only %ld in-flight sequence numbers per group fit in MPI_TAG_UB\n", nseq_);
  }
  void submit(std::shared_ptr<HJob> j) {
    std::lock_guard<std::mutex> lk(mu_); incoming_.push_back(std::move(j)); cv_in_.notify_one();
  }
  void wait_done(HJob& j) { std::unique_lock<std::mutex> lk(mu_); cv_done_.wait(lk, [&] { return j.done; }); }
  void worker_finished() { std::lock_guard<std::mutex> lk(mu_); ++finished_; cv_in_.notify_one(); }
  void run(int nworkers);

 private:
  enum Kind { COLL, P2P_SEND, P2P_RECV };
  struct Active {
    Kind kind = COLL;
    std::vector<std::shared_ptr<HJob>> jobs;  // COLL: local members indexed by gidx (null if remote)
    int phase = 0;
    std::vector<MPI_Request> reqs;
  };

  NodeRuntime* rt_;
  std::mutex mu_;
  std::condition_variable cv_in_, cv_done_;
  std::deque<std::shared_ptr<HJob>> incoming_;
  int finished_ = 0;
  long nseq_ = 1;
  std::map<std::tuple<int, int, uint64_t>, std::vector<std::shared_ptr<HJob>>> pending_;
  std::list<Active> active_;
  std::map<std::tuple<int, int, int>, std::deque<std::shared_ptr<HJob>>> lsend_, lrecv_;

  int me() const { return rt_->rank; }
  bool here(int worker) const { return rt_->node_of[worker] == me(); }
  int devof(const HJob& j) const { return j.local; }  // worker i of a node drives GPU i
  char* S(const HJob& j) { return rt_->windows->base(j.local) + j.soff; }
  char* R(const HJob& j) { return rt_->windows->base(j.local) + j.roff; }
  char* M(const HJob& j) { return rt_->windows->base(j.local) + j.moff; }
  int tag(int role, int src, int dst, uint64_t seq, int phase) const {
    const long G = rt_->wpn;
    return int((((long(role) * G + rt_->local_of[src]) * G + rt_->local_of[dst]) * nseq_ + long(seq % nseq_)) * 2 + phase);
  }

  void handle(std::shared_ptr<HJob> j);
  void start_coll(std::vector<std::shared_ptr<HJob>>& jobs);
  void post_msgs(Active& a);
  bool local_phase(Active& a);  // returns true when the op is complete
  void finish(const std::vector<std::shared_ptr<HJob>>& jobs);
};

void Engine::finish(const std::vector<std::shared_ptr<HJob>>& jobs) {
  std::set<int> devs;
  for (auto& j : jobs) if (j) devs.insert(devof(*j));
  for (int d : devs) dev::xfer_sync(d);
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& j : jobs) if (j) j->done = true;
  cv_done_.notify_all();
}

void Engine::handle(std::shared_ptr<HJob> j) {
  const OpDesc& op = j->op;
  if (op.type == OpDesc::SEND || op.type == OpDesc::RECV) {
    const bool send = op.type == OpDesc::SEND;
    const int peer = j->members[op.peer];
    const int src = send ? j->worker : peer, dst = send ? peer : j->worker;
    if (here(peer)) {  // intra-node: match and copy window -> destination buffer
      auto key = std::make_tuple(src, dst, op.tag);
      auto& other = send ? lrecv_ : lsend_;
      auto it = other.find(key);
      if (it != other.end() && !it->second.empty()) {
        auto o = it->second.front(); it->second.pop_front();
        if (it->second.empty()) other.erase(it);
        HJob& s = send ? *j : *o; HJob& r = send ? *o : *j;
        dev::xfer_copy(devof(r), r.op.rbuf, S(s), std::min(s.op.n, r.op.n));
        finish({j, o});
      } else (send ? lsend_ : lrecv_)[key].push_back(j);
      return;
    }
    Active a; a.kind = send ? P2P_SEND : P2P_RECV; a.jobs = {j};
    int t = tag(int(Role::P2P), src, dst, uint64_t(op.tag), 0);
    if (send) rt_->windows->isend(j->local, j->soff, op.n, rt_->node_of[dst], t, a.reqs);
    else rt_->windows->irecv(j->local, j->roff, op.n, rt_->node_of[src], t, a.reqs);
    active_.push_back(std::move(a));
    return;
  }
  auto key = std::make_tuple(int(j->role), j->leader, j->seq);
  auto& v = pending_[key];
  v.push_back(j);
  int nlocal = 0;
  for (int w : j->members) nlocal += here(w);
  if (int(v.size()) == nlocal) {
    std::vector<std::shared_ptr<HJob>> jobs = std::move(v);
    pending_.erase(key);
    start_coll(jobs);
  }
}

void Engine::start_coll(std::vector<std::shared_ptr<HJob>>& jobs) {
  Active a; a.kind = COLL;
  a.jobs.assign(jobs[0]->members.size(), nullptr);
  for (auto& j : jobs) a.jobs[j->gidx] = j;
  post_msgs(a);
  active_.push_back(std::move(a));
}

// Post the inter-node messages of the current phase.
void Engine::post_msgs(Active& a) {
  HJob& any = *std::find_if(a.jobs.begin(), a.jobs.end(), [](auto& p) { return bool(p); })->get();
  const OpDesc& op0 = any.op;
  const auto& m = any.members;
  const int p = int(m.size());
  const size_t es = any.es, B = op0.n * es;
  const size_t c = (op0.n + p - 1) / std::max(p, 1);
  auto blen = [&](int j) -> size_t { long long l = (long long)op0.n - (long long)j * c; return size_t(std::clamp<long long>(l, 0, c)) * es; };
  for (int j = 0; j < p; ++j)
    for (int w = 0; w < p; ++w) {
      if (j == w || rt_->node_of[m[j]] == rt_->node_of[m[w]]) continue;
      const bool recv = here(m[w]), send = here(m[j]);
      if (!recv && !send) continue;
      const int t = tag(int(any.role), m[j], m[w], any.seq, a.phase);
      size_t soff = 0, roff = 0, len = 0;
      HJob* J = a.jobs[j].get(); HJob* Wj = a.jobs[w].get();
      switch (op0.type) {
        case OpDesc::AG: len = B; if (send) soff = J->soff; if (recv) roff = Wj->roff + j * B; break;
        case OpDesc::RS: len = B; if (send) soff = J->soff + w * B; if (recv) roff = Wj->roff + j * B; break;
        case OpDesc::AR:
          if (a.phase == 0) { len = blen(w); if (send) soff = J->soff + w * c * es; if (recv) roff = Wj->roff + j * c * es; }
          else { len = blen(j); if (send) soff = J->moff; if (recv) roff = Wj->roff + j * c * es; }
          break;
        case OpDesc::A2AV:
          if (send) { soff = J->soff + J->op.sdispls[w] * es; len = J->op.scounts[w] * es; }
          if (recv) { roff = Wj->roff + Wj->op.rdispls[j] * es; len = Wj->op.rcounts[j] * es; }
          break;
        case OpDesc::BARRIER: len = 0; if (send) soff = J->soff; if (recv) roff = Wj->roff; break;
        default: break;
      }
      if (recv) rt_->windows->irecv(Wj->local, roff, len, rt_->node_of[m[j]], t, a.reqs);
      if (send) rt_->windows->isend(J->local, soff, len, rt_->node_of[m[w]], t, a.reqs);
    }
}

// Intra-node part of the current phase (zero-copy reads of peer windows).
bool Engine::local_phase(Active& a) {
  HJob& any = *std::find_if(a.jobs.begin(), a.jobs.end(), [](auto& p) { return bool(p); })->get();
  const OpDesc& op0 = any.op;
  const int p = int(any.members.size());
  const size_t es = any.es, B = op0.n * es;
  const size_t c = (op0.n + p - 1) / std::max(p, 1);
  auto blen = [&](int j) -> size_t { long long l = (long long)op0.n - (long long)j * c; return size_t(std::clamp<long long>(l, 0, c)); };
  auto local = [&](int j) { return bool(a.jobs[j]); };
  std::vector<const void*> srcs(p);
  for (int w = 0; w < p; ++w) {
    if (!local(w)) continue;
    HJob& W = *a.jobs[w];
    char* dst = static_cast<char*>(W.op.rbuf);
    const int d = devof(W);
    switch (op0.type) {
      case OpDesc::AG:
        for (int j = 0; j < p; ++j)
          dev::xfer_copy(d, dst + j * B, local(j) ? S(*a.jobs[j]) : R(W) + j * B, B);
        break;
      case OpDesc::RS:
        for (int j = 0; j < p; ++j) srcs[j] = local(j) ? S(*a.jobs[j]) + w * B : R(W) + j * B;
        dev::xfer_reduce(d, dst, srcs.data(), p, op0.n, op0.dt, false);
        break;
      case OpDesc::AR:
        if (a.phase == 0) {
          if (blen(w) == 0) break;
          for (int j = 0; j < p; ++j) srcs[j] = local(j) ? S(*a.jobs[j]) + w * c * es : R(W) + j * c * es;
          dev::xfer_reduce(d, M(W), srcs.data(), p, blen(w), op0.dt, false);
        } else {
          for (int j = 0; j < p; ++j)
            if (blen(j)) dev::xfer_copy(d, dst + j * c * es, local(j) ? M(*a.jobs[j]) : R(W) + j * c * es, blen(j) * es);
        }
        break;
      case OpDesc::A2AV:
        for (int j = 0; j < p; ++j) {
          size_t cnt = W.op.rcounts[j] * es;
          if (!cnt) continue;
          const char* src = local(j) ? S(*a.jobs[j]) + a.jobs[j]->op.sdispls[w] * es : R(W) + W.op.rdispls[j] * es;
          dev::xfer_copy(d, dst + W.op.rdispls[j] * es, src, cnt);
        }
        break;
      default: break;
    }
  }
  if (op0.type == OpDesc::AR && a.phase == 0) {
    std::set<int> devs;
    for (auto& j : a.jobs) if (j) devs.insert(devof(*j));
    for (int dv : devs) dev::xfer_sync(dv);  // reduced blocks must be in memory before MPI reads them
    a.phase = 1; a.reqs.clear();
    post_msgs(a);
    return false;
  }
  return true;
}

void Engine::run(int nworkers) {
  while (true) {
    std::vector<std::shared_ptr<HJob>> batch;
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (incoming_.empty()) {
        if (active_.empty()) {
          if (finished_ == nworkers && pending_.empty() && lsend_.empty() && lrecv_.empty()) break;
          cv_in_.wait_for(lk, std::chrono::milliseconds(5));
        } else cv_in_.wait_for(lk, std::chrono::microseconds(20));
      }
      batch.assign(incoming_.begin(), incoming_.end());
      incoming_.clear();
    }
    for (auto& j : batch) handle(j);
    for (auto it = active_.begin(); it != active_.end();) {
      int flag = 1;
      if (!it->reqs.empty()) MPI_Testall(int(it->reqs.size()), it->reqs.data(), &flag, MPI_STATUSES_IGNORE);
      if (!flag) { ++it; continue; }
      bool complete = true;
      if (it->kind == COLL) complete = local_phase(*it);
      else if (it->kind == P2P_RECV) { HJob& r = *it->jobs[0]; dev::xfer_copy(devof(r), r.op.rbuf, R(r), r.op.n); }
      if (complete) { finish(it->jobs); it = active_.erase(it); }
      else ++it;
    }
  }
}

std::shared_ptr<Engine> make_engine(NodeRuntime* rt) { return std::make_shared<Engine>(rt); }
void engine_run(Engine& e, int n) { e.run(n); }
void engine_worker_finished(Engine& e) { e.worker_finished(); }
void engine_destroy(std::shared_ptr<Engine>& e) { e.reset(); }

// ------------------------------------------------------------- HierComm
namespace {
class HierComm : public CommImpl {
 public:
  uint64_t seq = 0;
  std::shared_ptr<ReqState> post(const OpDesc& op) override {
    auto j = std::make_shared<HJob>();
    j->op = op; j->worker = my_worker(); j->gidx = rank; j->local = local_index(j->worker);
    j->leader = members[0]; j->role = role; j->members = members;
    j->es = (op.type == OpDesc::SEND || op.type == OpDesc::RECV || op.type == OpDesc::BARRIER) ? 1 : dtype_size(op.dt);
    if (op.type != OpDesc::SEND && op.type != OpDesc::RECV) j->seq = seq++;
    const int p = int(members.size());
    const bool remote = nodes > 1;
    const size_t es = j->es, B = op.n * es, c = (op.n + p - 1) / p;
    const void* stage_src = nullptr;
    switch (op.type) {
      case OpDesc::AG: j->sb = B; j->rb = remote ? p * B : 0; stage_src = op.sbuf; break;
      case OpDesc::RS: j->sb = p * B; j->rb = remote ? p * B : 0; stage_src = op.sbuf; break;
      case OpDesc::AR: j->sb = B; j->rb = remote ? p * c * es : 0; j->mb = c * es; stage_src = op.rbuf; break;
      case OpDesc::A2AV: {
        size_t sx = 0, rx = 0;
        for (int k = 0; k < p; ++k) {
          sx = std::max(sx, op.sdispls[k] + op.scounts[k]); rx = std::max(rx, op.rdispls[k] + op.rcounts[k]);
        }
        j->sb = sx * es; j->rb = remote ? rx * es : 0; stage_src = op.sbuf; break;
      }
      case OpDesc::SEND: j->sb = op.n; stage_src = op.sbuf; break;
      case OpDesc::RECV: j->rb = node_of(members[op.peer]) != node_of(my_worker()) ? op.n : 0; break;
      case OpDesc::BARRIER: break;
    }
    auto& sl = *t_rt->slots[j->local];
    j->soff = sl.alloc(j->sb); j->roff = sl.alloc(j->rb); j->moff = sl.alloc(j->mb);
    if (j->sb) {
      dev::memcpy_d2d(t_rt->windows->base(j->local) + j->soff, stage_src, j->sb);
      dev::sync();
    }
    t_rt->engine->submit(j);
    return j;
  }
  void wait(ReqState& rs) override {
    auto& j = static_cast<HJob&>(rs);
    t_rt->engine->wait_done(j);
    auto& sl = *t_rt->slots[j.local];
    sl.release(j.soff, j.sb); sl.release(j.roff, j.rb); sl.release(j.moff, j.mb);
  }
};
}  // namespace

std::shared_ptr<CommImpl> make_hier_comm(Role role, const std::vector<int>& members) {
  auto c = std::make_shared<HierComm>();
  c->role = role; c->members = members;
  auto it = std::find(members.begin(), members.end(), my_worker());
  XPSIM_CHECK(it != members.end(), "worker not in its own group");
  c->rank = int(it - members.begin());
  std::vector<int> nodes; for (int w : members) nodes.push_back(node_of(w));
  std::sort(nodes.begin(), nodes.end()); c->nodes = int(std::unique(nodes.begin(), nodes.end()) - nodes.begin());
  return c;
}

}  // namespace xpsim
