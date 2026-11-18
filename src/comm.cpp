// Comm front-end (accounting, host helpers), runtime bring-up and the MPI
// (one-rank-per-worker) transport.
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include "comm_internal.hpp"
#include "xpsim/half_host.hpp"
#include "xpsim/trace.hpp"

namespace xpsim {

thread_local NodeRuntime* t_rt = nullptr;
thread_local int t_worker = 0;

const char* role_name(Role r) {
  static const char* n[] = {"world", "tp", "cp", "dp", "fsdp", "hsdp", "pp", "ep", "edp", "stage", "tpcp", "p2p"};
  return n[int(r)];
}

int world_size() { return t_rt->world; }
int my_worker() { return t_worker; }
int num_nodes() { return t_rt->nodes; }
int workers_per_node() { return t_rt->wpn; }
int node_of(int w) { return t_rt->node_of[w]; }
int local_index(int w) { return t_rt->local_of[w]; }
Transport transport() { return t_rt->opt.transport; }
const RuntimeOptions& runtime_options() { return t_rt->opt; }
int process_rank() { return t_rt ? t_rt->rank : 0; }

// =================================================================== Comm
static thread_local std::vector<Comm> t_registry;
Comm Comm::create(Role role, const std::vector<int>& members) {
  Comm c;
  c.impl_ = t_rt->opt.transport == Transport::MPI ? make_mpi_comm(role, members) : make_hier_comm(role, members);
  c.impl_->reg_index = int(t_registry.size());
  t_registry.push_back(c);
  return c;
}
Comm& Comm::from_index(int i) {
  if (i < 0 || i >= int(t_registry.size())) fail("communicator index " + std::to_string(i) + " not created on this worker");
  return t_registry[i];
}
int Comm::registry_size() { return int(t_registry.size()); }
int Comm::index() const { return impl_->reg_index; }
static thread_local Comm t_world;
Comm& Comm::world() { return t_world; }
int Comm::rank() const { return impl_->rank; }
int Comm::size() const { return int(impl_->members.size()); }
Role Comm::role() const { return impl_->role; }
const std::vector<int>& Comm::members() const { return impl_->members; }
int Comm::nodes_spanned() const { return impl_->nodes; }

namespace {
struct Acct { double bytes = 0, bus = 0, intra = 0, inter = 0; };

// Logical traffic of the direct algorithms (per worker): intra vs inter node.
Acct account(const CommImpl& c, const OpDesc& op) {
  Acct a;
  const int p = int(c.members.size()), me = c.members[c.rank];
  const double es = dtype_size(op.dt);
  auto split = [&](int j, double b) { (node_of(c.members[j]) == node_of(me) ? a.intra : a.inter) += b; };
  switch (op.type) {
    case OpDesc::AR: {
      a.bytes = op.n * es; a.bus = a.bytes * busbw_factor(CollKind::ALLREDUCE, p);
      for (int j = 0; j < p; ++j) if (j != c.rank) split(j, 2.0 * a.bytes / p);
      break;
    }
    case OpDesc::AG: case OpDesc::RS: {
      a.bytes = double(p) * op.n * es; a.bus = a.bytes * busbw_factor(CollKind::ALLGATHER, p);
      for (int j = 0; j < p; ++j) if (j != c.rank) split(j, op.n * es);
      break;
    }
    case OpDesc::A2AV: {
      for (int j = 0; j < p; ++j) {
        double b = op.scounts[j] * es; a.bytes += b;
        if (j != c.rank) { a.bus += b; split(j, b); }
      }
      break;
    }
    case OpDesc::SEND: a.bytes = a.bus = double(op.n); split(op.peer, double(op.n)); break;
    default: break;
  }
  return a;
}
}  // namespace

static Request post_op(CommImpl& impl, const OpDesc& op, Cat cat) {
  auto& m = metrics();
  int tid = -1;
  if (trace::active()) {  // record the call; everything below is communication internals
    tid = trace::next_request_id();
    const int p = int(impl.members.size());
    std::vector<int64_t> a = {impl.reg_index, int64_t(op.type), trace::ptr(op.sbuf), trace::ptr(op.rbuf), int64_t(op.n),
                              int64_t(op.dt), op.peer, op.tag, int64_t(cat), tid, p};
    if (op.type == OpDesc::A2AV)
      for (auto* v : {&op.scounts, &op.sdispls, &op.rcounts, &op.rdispls}) for (size_t x : *v) a.push_back(int64_t(x));
    trace::rec_vec(TOpKind::C_POST, std::move(a));
  }
  dev::Suspend suspend;
  if (t_rt->opt.sync_collectives && op.type != OpDesc::SEND && op.type != OpDesc::RECV && op.type != OpDesc::BARRIER) {
    OpDesc b; b.type = OpDesc::BARRIER;
    double t0 = now_sec();
    auto r = impl.post(b); impl.wait(*r);
    if (m.recording) { m.cat[int(Cat::SYNC_WAIT)].time += now_sec() - t0; m.cat[int(Cat::SYNC_WAIT)].calls += 1; }
  }
  if (dev::is_gpu()) dev::sync();  // MPI / the engine are not stream-ordered: data must be ready
  Acct a = account(impl, op);
  count_bytes(cat, a.bytes, a.bus, a.intra, a.inter);
  double t0 = now_sec();
  Request r{impl.post(op)};
  r.s->cat = cat;
  r.s->trace_id = tid;
  if (m.recording) { m.cat[int(cat)].time += now_sec() - t0; m.cat[int(cat)].calls += 1; }
  return r;
}
void Comm::wait(Request& r) {
  if (!r.s || r.s->waited) return;
  if (trace::active() && r.s->trace_id >= 0) trace::rec(TOpKind::C_WAIT, {impl_->reg_index, r.s->trace_id});
  dev::Suspend suspend;
  double t0 = now_sec();
  impl_->wait(*r.s);
  r.s->waited = true;
  auto& m = metrics();
  if (m.recording) m.cat[int(r.s->cat)].time += now_sec() - t0;
  r.s.reset();
}
void Comm::waitall(std::vector<Request>& rs) { for (auto& r : rs) wait(r); rs.clear(); }

Request Comm::iallreduce(void* buf, size_t n, DType dt, Cat cat) {
  OpDesc o; o.type = OpDesc::AR; o.rbuf = buf; o.n = n; o.dt = dt; return post_op(*impl_, o, cat);
}
Request Comm::iallgather(const void* s, void* r, size_t n, DType dt, Cat cat) {
  OpDesc o; o.type = OpDesc::AG; o.sbuf = s; o.rbuf = r; o.n = n; o.dt = dt; return post_op(*impl_, o, cat);
}
Request Comm::ireduce_scatter(const void* s, void* r, size_t n, DType dt, Cat cat) {
  OpDesc o; o.type = OpDesc::RS; o.sbuf = s; o.rbuf = r; o.n = n; o.dt = dt; return post_op(*impl_, o, cat);
}
Request Comm::isend(const void* buf, size_t bytes, int dst, int tag, Cat cat) {
  OpDesc o; o.type = OpDesc::SEND; o.sbuf = buf; o.n = bytes; o.peer = dst; o.tag = tag; return post_op(*impl_, o, cat);
}
Request Comm::irecv(void* buf, size_t bytes, int src, int tag, Cat cat) {
  OpDesc o; o.type = OpDesc::RECV; o.rbuf = buf; o.n = bytes; o.peer = src; o.tag = tag; return post_op(*impl_, o, cat);
}
void Comm::allreduce(void* b, size_t n, DType dt, Cat c) { if (size() == 1) return; auto r = iallreduce(b, n, dt, c); wait(r); }
void Comm::allgather(const void* s, void* r, size_t n, DType dt, Cat c) {
  if (size() == 1) { if (s != r) dev::memcpy_d2d(r, s, n * dtype_size(dt)); return; }
  auto q = iallgather(s, r, n, dt, c); wait(q);
}
void Comm::reduce_scatter(const void* s, void* r, size_t n, DType dt, Cat c) {
  if (size() == 1) { if (s != r) dev::memcpy_d2d(r, s, n * dtype_size(dt)); return; }
  auto q = ireduce_scatter(s, r, n, dt, c); wait(q);
}
void Comm::alltoallv(const void* s, const std::vector<size_t>& sc, const std::vector<size_t>& sd, void* r,
                     const std::vector<size_t>& rc, const std::vector<size_t>& rd, DType dt, Cat cat) {
  OpDesc o; o.type = OpDesc::A2AV; o.sbuf = s; o.rbuf = r; o.dt = dt;
  o.scounts = sc; o.sdispls = sd; o.rcounts = rc; o.rdispls = rd;
  auto q = post_op(*impl_, o, cat); wait(q);
}
void Comm::barrier(Cat cat) { auto q = ibarrier(cat); wait(q); }
Request Comm::ibarrier(Cat cat) { OpDesc o; o.type = OpDesc::BARRIER; return post_op(*impl_, o, cat); }
Request Comm::ialltoallv(const void* s, const std::vector<size_t>& sc, const std::vector<size_t>& sd, void* r,
                         const std::vector<size_t>& rc, const std::vector<size_t>& rd, DType dt, Cat cat) {
  OpDesc o; o.type = OpDesc::A2AV; o.sbuf = s; o.rbuf = r; o.dt = dt;
  o.scounts = sc; o.sdispls = sd; o.rcounts = rc; o.rdispls = rd;
  return post_op(*impl_, o, cat);
}
void Comm::allreduce_max(float* buf, size_t n, Cat cat) {
  if (size() == 1 || n == 0) return;
  float* all = static_cast<float*>(dev::alloc(n * size() * sizeof(float)));
  allgather(buf, all, n, DType::F32, cat);
  dev::max_blocks(buf, all, size(), n);
  dev::free(all);
}
// Host helpers are recorded as one op (inputs as constants) and replayed as calls.
static bool rec_host(const Comm& c, int which, int n, Cat cat, const void* in, size_t bytes) {
  if (!trace::active()) return false;
  trace::rec(TOpKind::C_HOST, {c.index(), which, n, int64_t(cat), trace::cst(in, bytes)});
  return true;
}

// Host helpers: stage through device memory so both transports apply.
void Comm::allreduce_host(double* buf, int n, Cat cat) {
  rec_host(*this, 0, n, cat, buf, size_t(n) * 8);
  dev::Suspend suspend;
  if (size() == 1) return;
  void* d = dev::alloc(n * 8); dev::memcpy_h2d(d, buf, n * 8);
  allreduce(d, n, DType::F64, cat);
  dev::memcpy_d2h(buf, d, n * 8); dev::free(d);
}
void Comm::allgather_host(const double* s, double* r, int n, Cat cat) {
  rec_host(*this, 1, n, cat, s, size_t(n) * 8);
  dev::Suspend suspend;
  void* d = dev::alloc(size_t(n) * 8 * (size() + 1));
  dev::memcpy_h2d(d, s, n * 8);
  allgather(d, static_cast<char*>(d) + n * 8, n, DType::F64, cat);
  dev::memcpy_d2h(r, static_cast<char*>(d) + n * 8, size_t(n) * 8 * size()); dev::free(d);
}
void Comm::allreduce_max_host(float* buf, int n, Cat cat) {
  rec_host(*this, 2, n, cat, buf, size_t(n) * 4);
  dev::Suspend suspend;
  if (size() == 1) return;
  std::vector<double> s(buf, buf + n), r(size_t(n) * size());
  allgather_host(s.data(), r.data(), n, cat);
  for (int i = 0; i < n; ++i) {
    double mx = r[i];
    for (int j = 1; j < size(); ++j) mx = std::max(mx, r[size_t(j) * n + i]);
    buf[i] = float(mx);
  }
}
void Comm::alltoall_host(const long long* s, long long* r, int n_per, Cat cat) {
  const int p = size();
  rec_host(*this, 3, n_per, cat, s, size_t(p) * n_per * 8);
  dev::Suspend suspend;
  std::vector<size_t> c(p, n_per), d(p);
  for (int j = 0; j < p; ++j) d[j] = size_t(j) * n_per;
  char* db = static_cast<char*>(dev::alloc(size_t(2) * p * n_per * 8));
  dev::memcpy_h2d(db, s, size_t(p) * n_per * 8);
  alltoallv(db, c, d, db + size_t(p) * n_per * 8, c, d, DType::F64, cat);  // raw 8-byte payload
  dev::memcpy_d2h(r, db + size_t(p) * n_per * 8, size_t(p) * n_per * 8); dev::free(db);
}

// =================================================================== MPI transport
namespace {
MPI_Datatype mpi_type(DType dt) {
  switch (dt) {
    case DType::F32: return MPI_FLOAT;
    case DType::F64: return MPI_DOUBLE;
    default: return MPI_UINT16_T;
  }
}
template <class TT> void sum_fn(void* in, void* inout, int* len, MPI_Datatype*) {
  auto* a = static_cast<TT*>(in); auto* b = static_cast<TT*>(inout);
  for (int i = 0; i < *len; ++i) b[i] = from_float<TT>(to_float(a[i]) + to_float(b[i]));
}

struct MpiReq : ReqState {
  MPI_Request req = MPI_REQUEST_NULL;
  void* hsend = nullptr; void* hrecv = nullptr; void* drecv = nullptr; size_t recv_bytes = 0;
};

int icount(size_t n) { XPSIM_CHECK(n <= size_t(INT_MAX), "message too large for MPI int count"); return int(n); }

class MpiComm : public CommImpl {
 public:
  MPI_Comm comm = MPI_COMM_NULL;
  ~MpiComm() override { if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm); }

  std::shared_ptr<ReqState> post(const OpDesc& op) override {
    auto r = std::make_shared<MpiReq>();
    const bool gpu = dev::is_gpu(), half = op.dt == DType::BF16 || op.dt == DType::F16;
    const bool reduction = op.type == OpDesc::AR || op.type == OpDesc::RS;
    // Stage through pinned host memory when MPI cannot see device memory, or
    // when a 16-bit reduction needs the (host-only) user-defined MPI_Op.
    const bool stage = gpu && (!t_rt->opt.cuda_aware || (half && reduction));
    const size_t es = dtype_size(op.dt);
    const int p = int(members.size());
    MPI_Datatype mt = mpi_type(op.dt);
    MPI_Op mop = op.dt == DType::BF16 ? t_rt->op_sum_bf16 : op.dt == DType::F16 ? t_rt->op_sum_fp16 : MPI_SUM;
    auto stage_in = [&](const void* d, size_t bytes) -> const void* {
      if (!stage) return d;
      r->hsend = dev::host_alloc(bytes); dev::memcpy_d2h(r->hsend, d, bytes); return r->hsend;
    };
    auto stage_out = [&](void* d, size_t bytes) -> void* {
      if (!stage) return d;
      r->hrecv = dev::host_alloc(bytes); r->drecv = d; r->recv_bytes = bytes; return r->hrecv;
    };
    switch (op.type) {
      case OpDesc::AR: {
        size_t b = op.n * es;
        void* buf = stage_out(op.rbuf, b);
        if (stage) dev::memcpy_d2h(buf, op.rbuf, b);
        MPI_Iallreduce(MPI_IN_PLACE, buf, icount(op.n), mt, mop, comm, &r->req);
        break;
      }
      case OpDesc::AG: {
        const void* s = stage_in(op.sbuf, op.n * es);
        void* d = stage_out(op.rbuf, op.n * es * p);
        MPI_Iallgather(s, icount(op.n), mt, d, icount(op.n), mt, comm, &r->req);
        break;
      }
      case OpDesc::RS: {
        const void* s = stage_in(op.sbuf, op.n * es * p);
        void* d = stage_out(op.rbuf, op.n * es);
        MPI_Ireduce_scatter_block(s, d, icount(op.n), mt, mop, comm, &r->req);
        break;
      }
      case OpDesc::A2AV: {
        std::vector<int> sc(p), sd(p), rc(p), rd(p);
        size_t sext = 0, rext = 0;
        for (int j = 0; j < p; ++j) {
          sc[j] = icount(op.scounts[j]); sd[j] = icount(op.sdispls[j]);
          rc[j] = icount(op.rcounts[j]); rd[j] = icount(op.rdispls[j]);
          sext = std::max(sext, op.sdispls[j] + op.scounts[j]); rext = std::max(rext, op.rdispls[j] + op.rcounts[j]);
        }
        const void* s = stage_in(op.sbuf, sext * es);
        void* d = stage_out(op.rbuf, rext * es);
        MPI_Datatype bt = op.dt == DType::F64 ? MPI_DOUBLE : mt;
        // MPI keeps the count arrays until completion only for blocking calls
        // in some implementations; complete here for portability.
        MPI_Alltoallv(s, sc.data(), sd.data(), bt, d, rc.data(), rd.data(), bt, comm);
        r->req = MPI_REQUEST_NULL;
        break;
      }
      case OpDesc::SEND: {
        const void* s = stage_in(op.sbuf, op.n);
        MPI_Isend(s, icount(op.n), MPI_BYTE, op.peer, op.tag, comm, &r->req);
        break;
      }
      case OpDesc::RECV: {
        void* d = stage_out(op.rbuf, op.n);
        MPI_Irecv(d, icount(op.n), MPI_BYTE, op.peer, op.tag, comm, &r->req);
        break;
      }
      case OpDesc::BARRIER: MPI_Ibarrier(comm, &r->req); break;
    }
    return r;
  }
  void wait(ReqState& rs) override {
    auto& r = static_cast<MpiReq&>(rs);
    if (r.req != MPI_REQUEST_NULL) MPI_Wait(&r.req, MPI_STATUS_IGNORE);
    if (r.hrecv) { dev::memcpy_h2d(r.drecv, r.hrecv, r.recv_bytes); dev::host_free(r.hrecv); }
    if (r.hsend) dev::host_free(r.hsend);
    r.hrecv = r.hsend = nullptr;
  }
};
}  // namespace

std::shared_ptr<CommImpl> make_mpi_comm(Role role, const std::vector<int>& members) {
  auto c = std::make_shared<MpiComm>();
  c->role = role; c->members = members;
  auto it = std::find(members.begin(), members.end(), my_worker());
  XPSIM_CHECK(it != members.end(), "worker not in its own group");
  c->rank = int(it - members.begin());
  MPI_Comm_split(MPI_COMM_WORLD, members[0], c->rank, &c->comm);
  std::vector<int> nodes; for (int w : members) nodes.push_back(node_of(w));
  std::sort(nodes.begin(), nodes.end()); c->nodes = int(std::unique(nodes.begin(), nodes.end()) - nodes.begin());
  return c;
}

// =================================================================== bring-up
static void init_world_comm() {
  std::vector<int> all(t_rt->world);
  for (int i = 0; i < t_rt->world; ++i) all[i] = i;
  t_world = Comm::create(Role::WORLD, all);
}

int run_workers(int argc, char** argv, const RuntimeOptions& opt, int (*worker_main)(int, char**)) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  auto rt = std::make_unique<NodeRuntime>();
  rt->opt = opt;
  MPI_Comm_rank(MPI_COMM_WORLD, &rt->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rt->nranks);
  MPI_Op_create(&sum_fn<bf16_t>, 1, &rt->op_sum_bf16);
  MPI_Op_create(&sum_fn<fp16_t>, 1, &rt->op_sum_fp16);
  t_rt = rt.get();
  int rc = 0;

  if (opt.transport == Transport::MPI) {
    rt->world = rt->nranks;
    int node = 0, local = 0;
    if (opt.ranks_per_node > 0) {
      node = rt->rank / opt.ranks_per_node; local = rt->rank % opt.ranks_per_node;
    } else {
      MPI_Comm shm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rt->rank, MPI_INFO_NULL, &shm);
      int leader = rt->rank; MPI_Comm_rank(shm, &local);
      MPI_Bcast(&leader, 1, MPI_INT, 0, shm); MPI_Comm_free(&shm);
      std::vector<int> leaders(rt->nranks);
      MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
      std::vector<int> uniq = leaders; std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      node = int(std::lower_bound(uniq.begin(), uniq.end(), leader) - uniq.begin());
    }
    std::vector<int> pair = {node, local}, all(2 * rt->nranks);
    MPI_Allgather(pair.data(), 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);
    rt->node_of.resize(rt->world); rt->local_of.resize(rt->world);
    int maxnode = 0, maxlocal = 0;
    for (int i = 0; i < rt->world; ++i) {
      rt->node_of[i] = all[2 * i]; rt->local_of[i] = all[2 * i + 1];
      maxnode = std::max(maxnode, all[2 * i]); maxlocal = std::max(maxlocal, all[2 * i + 1]);
    }
    rt->nodes = maxnode + 1; rt->wpn = maxlocal + 1;
    t_worker = rt->rank;
    try {
      if (opt.gpu) {
        int ng = dev::gpu_count(); XPSIM_CHECK(ng > 0, "no CUDA devices visible");
        dev::init(dev::Backend::GPU, local % ng);
      } else dev::init(dev::Backend::CPU, 0);
      init_world_comm();
      rc = worker_main(argc, argv);
      t_world = Comm(); t_registry.clear();
      dev::finalize();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[rank %d] error: %s\n", rt->rank, e.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } else {  // HIER: one rank per node, worker threads
    rt->nodes = rt->nranks;
    int wpn = opt.workers_per_node > 0 ? opt.workers_per_node : (opt.gpu ? dev::gpu_count() : 1);
    int mn = 0, mx = 0;
    MPI_Allreduce(&wpn, &mn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&wpn, &mx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (mn != mx || wpn < 1) {
      if (rt->rank == 0) std::fprintf(stderr, "xpsim: all nodes need the same number of workers (got %d..%d)\n", mn, mx);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    rt->wpn = wpn; rt->world = wpn * rt->nodes;
    rt->node_of.resize(rt->world); rt->local_of.resize(rt->world);
    for (int i = 0; i < rt->world; ++i) { rt->node_of[i] = i / wpn; rt->local_of[i] = i % wpn; }
    try {
      if (opt.gpu) {
        dev::init(dev::Backend::GPU, 0);  // engine thread issues copies/reductions on every local GPU
        rt->windows = make_vmm_windows(wpn, opt.window_bytes, opt);
      } else rt->windows = make_host_windows(wpn, opt.window_bytes);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[node %d] window setup failed: %s\n", rt->rank, e.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < wpn; ++i) rt->slots.push_back(std::make_unique<SlotAllocator>(rt->windows->bytes()));
    rt->engine = make_engine(rt.get());
    NodeRuntime* rtp = rt.get();
    std::vector<std::thread> threads;
    std::vector<int> rcs(wpn, 0);
    for (int i = 0; i < wpn; ++i)
      threads.emplace_back([=, &rcs]() {
        t_rt = rtp; t_worker = rtp->rank * wpn + i;
        try {
          dev::init(opt.gpu ? dev::Backend::GPU : dev::Backend::CPU, i);
          init_world_comm();
          rcs[i] = worker_main(argc, argv);
          t_world = Comm(); t_registry.clear();
          dev::finalize();
        } catch (const std::exception& e) {
          std::fprintf(stderr, "[worker %d] error: %s\n", t_worker, e.what());
          std::fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        engine_worker_finished(*rtp->engine);
      });
    engine_run(*rt->engine, wpn);
    for (auto& t : threads) t.join();
    for (int x : rcs) rc |= x;
    engine_destroy(rt->engine);
    rt->slots.clear();
    rt->windows.reset();
    if (opt.gpu) dev::finalize();
  }
  MPI_Op_free(&rt->op_sum_bf16); MPI_Op_free(&rt->op_sum_fp16);
  MPI_Finalize();
  t_rt = nullptr;
  return rc;
}

}  // namespace xpsim
