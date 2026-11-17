// mpishim implementation (see mpi.h). Collectives are executed eagerly at the
// call (non-blocking variants complete immediately); point-to-point sends are
// eager (copied into a mailbox), receives complete when a message arrives.
#include "mpi.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

int xpsim_main(int argc, char** argv);

struct mpishim_req {
  bool done = false;
  bool is_recv = false;
  void* buf = nullptr; size_t bytes = 0; int src = 0, tag = 0, ctx = 0, me = 0;
};

namespace {
struct CommInfo {
  std::vector<int> members;  // world ranks
  std::mutex mu; std::condition_variable cv;
  int count = 0; unsigned long long gen = 0;
  std::vector<std::vector<char>> tmp;  // per-rank staged inputs
  std::vector<const void*> ptr;
  std::vector<std::vector<int>> ints;
  int splits = 0;
};
std::mutex g_mu;
constexpr int kMaxComms = 1 << 16, kMaxOps = 256;
std::unique_ptr<CommInfo> g_comms[kMaxComms];  // fixed table: no reallocation while other ranks read
std::atomic<int> g_ncomms{0};
std::map<std::tuple<int, int, int, int>, int> g_split_map;  // (parent, seq, color, 0) -> handle
MPI_User_function* g_ops[kMaxOps];
std::atomic<int> g_nops{0};
int g_np = 1;
thread_local int t_rank = 0;
thread_local bool t_init = false, t_fin = false;
mpishim_req g_done_req{true};

// mailboxes
std::mutex g_mb_mu; std::condition_variable g_mb_cv;
std::map<std::tuple<int, int, int, int>, std::deque<std::vector<char>>> g_mb;  // (ctx, src, dst, tag)

CommInfo& C(MPI_Comm c) {
  if (c < 0 || c >= g_ncomms.load()) { std::fprintf(stderr, "mpishim: invalid communicator %d\n", c); std::abort(); }
  return *g_comms[c];
}
int rank_in(CommInfo& c, int w) { for (size_t i = 0; i < c.members.size(); ++i) if (c.members[i] == w) return int(i); return -1; }
size_t tsize(MPI_Datatype t) {
  switch (t) { case MPI_BYTE: case MPI_CHAR: return 1; case MPI_UINT16_T: return 2; case MPI_INT: case MPI_FLOAT: return 4; default: return 8; }
}
void barrier(CommInfo& c) {
  std::unique_lock<std::mutex> lk(c.mu);
  auto g = c.gen;
  if (++c.count == int(c.members.size())) { c.count = 0; ++c.gen; c.cv.notify_all(); }
  else c.cv.wait(lk, [&] { return c.gen != g; });
}
template <class TT> void apply_t(MPI_Op op, const TT* a, TT* b, int n) {
  for (int i = 0; i < n; ++i) {
    switch (op) {
      case MPI_SUM: b[i] = b[i] + a[i]; break;
      case MPI_MAX: b[i] = a[i] > b[i] ? a[i] : b[i]; break;
      case MPI_MIN: b[i] = a[i] < b[i] ? a[i] : b[i]; break;
      case MPI_LAND: b[i] = (a[i] && b[i]) ? 1 : 0; break;
    }
  }
}
void apply(MPI_Op op, MPI_Datatype t, const void* a, void* b, int n) {
  if (op >= 100) { MPI_Datatype tt = t; int nn = n; g_ops[op - 100](const_cast<void*>(a), b, &nn, &tt); return; }
  switch (t) {
    case MPI_INT: apply_t(op, (const int*)a, (int*)b, n); break;
    case MPI_FLOAT: apply_t(op, (const float*)a, (float*)b, n); break;
    case MPI_DOUBLE: apply_t(op, (const double*)a, (double*)b, n); break;
    case MPI_UNSIGNED_LONG_LONG: apply_t(op, (const unsigned long long*)a, (unsigned long long*)b, n); break;
    case MPI_LONG_LONG: apply_t(op, (const long long*)a, (long long*)b, n); break;
    default: std::fprintf(stderr, "mpishim: unsupported reduction type\n"); std::abort();
  }
}
// Stage `bytes` of my input, rendezvous, run f(me), rendezvous.
template <class F> void collective(MPI_Comm comm, const void* in, size_t bytes, F f) {
  CommInfo& c = C(comm);
  int me = rank_in(c, t_rank);
  {
    std::lock_guard<std::mutex> lk(c.mu);
    if (c.tmp.size() != c.members.size()) { c.tmp.resize(c.members.size()); c.ptr.resize(c.members.size()); c.ints.resize(c.members.size()); }
  }
  c.tmp[me].assign(static_cast<const char*>(in), static_cast<const char*>(in) + bytes);
  barrier(c);
  f(c, me);
  barrier(c);
}
}  // namespace

int MPI_Init_thread(int*, char***, int, int* provided) { t_init = true; if (provided) *provided = MPI_THREAD_MULTIPLE; return 0; }
int MPI_Init(int*, char***) { t_init = true; return 0; }
int MPI_Finalize() { MPI_Barrier(MPI_COMM_WORLD); t_fin = true; return 0; }
int MPI_Initialized(int* f) { *f = t_init; return 0; }
int MPI_Finalized(int* f) { *f = t_fin; return 0; }
int MPI_Abort(MPI_Comm, int code) { std::fprintf(stderr, "mpishim: MPI_Abort(%d) from rank %d\n", code, t_rank); std::fflush(stderr); std::_Exit(code ? code : 1); }
int MPI_Comm_rank(MPI_Comm c, int* r) { *r = rank_in(C(c), t_rank); return 0; }
int MPI_Comm_size(MPI_Comm c, int* s) { *s = int(C(c).members.size()); return 0; }
int MPI_Error_string(int, char* s, int* l) { std::strcpy(s, "mpishim error"); *l = int(std::strlen(s)); return 0; }
int MPI_Op_create(MPI_User_function* f, int, MPI_Op* op) {
  std::lock_guard<std::mutex> lk(g_mu); int i = g_nops.load(); g_ops[i] = f; g_nops.store(i + 1); *op = 100 + i; return 0;
}
int MPI_Op_free(MPI_Op* op) { *op = MPI_OP_NULL; return 0; }
int MPI_Comm_get_attr(MPI_Comm, int key, void* val, int* flag) {
  static int tagub = (1 << 30) - 1;
  if (key == MPI_TAG_UB) { *static_cast<void**>(val) = &tagub; *flag = 1; } else *flag = 0;
  return 0;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* out) {
  int ck[2] = {color, key};
  int seq = 0;
  std::vector<int> members;
  collective(comm, ck, sizeof(ck), [&](CommInfo& c, int me) {
    seq = c.splits;  // read before increment by the last finisher below
    std::vector<std::tuple<int, int, int>> v;  // key, oldrank, world
    for (size_t i = 0; i < c.members.size(); ++i) {
      int col, k; std::memcpy(&col, c.tmp[i].data(), 4); std::memcpy(&k, c.tmp[i].data() + 4, 4);
      if (col == color) v.emplace_back(k, int(i), c.members[i]);
    }
    std::sort(v.begin(), v.end());
    for (auto& t : v) members.push_back(std::get<2>(t));
    (void)me;
  });
  {
    CommInfo& c = C(comm);
    barrier(c);
    if (rank_in(c, t_rank) == 0) c.splits++;
    barrier(c);
  }
  std::lock_guard<std::mutex> lk(g_mu);
  auto k = std::make_tuple(comm, seq, color, 0);
  auto it = g_split_map.find(k);
  if (it == g_split_map.end()) {
    int idx = g_ncomms.load();
    if (idx >= kMaxComms) { std::fprintf(stderr, "mpishim: too many communicators\n"); std::abort(); }
    g_comms[idx] = std::make_unique<CommInfo>();
    g_comms[idx]->members = members;
    g_ncomms.store(idx + 1);
    it = g_split_map.emplace(k, idx).first;
  }
  *out = it->second;
  return 0;
}
int MPI_Comm_dup(MPI_Comm c, MPI_Comm* out) { int r; MPI_Comm_rank(c, &r); return MPI_Comm_split(c, 0, r, out); }
int MPI_Comm_split_type(MPI_Comm c, int, int key, MPI_Info, MPI_Comm* out) { return MPI_Comm_split(c, 0, key, out); }
int MPI_Comm_free(MPI_Comm* c) { *c = MPI_COMM_NULL; return 0; }

int MPI_Barrier(MPI_Comm c) { barrier(C(c)); return 0; }
int MPI_Bcast(void* buf, int n, MPI_Datatype t, int root, MPI_Comm comm) {
  size_t b = n * tsize(t);
  collective(comm, buf, b, [&](CommInfo& c, int me) { if (me != root) std::memcpy(buf, c.tmp[root].data(), b); });
  return 0;
}
int MPI_Allreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm comm) {
  size_t b = n * tsize(t);
  const void* in = s == MPI_IN_PLACE ? r : s;
  collective(comm, in, b, [&](CommInfo& c, int) {
    std::memcpy(r, c.tmp[0].data(), b);
    for (size_t j = 1; j < c.members.size(); ++j) apply(op, t, c.tmp[j].data(), r, n);
  });
  return 0;
}
int MPI_Allgather(const void* s, int n, MPI_Datatype t, void* r, int, MPI_Datatype, MPI_Comm comm) {
  size_t b = n * tsize(t);
  collective(comm, s, b, [&](CommInfo& c, int) {
    for (size_t j = 0; j < c.members.size(); ++j) std::memcpy(static_cast<char*>(r) + j * b, c.tmp[j].data(), b);
  });
  return 0;
}
int MPI_Allgatherv(const void* s, int n, MPI_Datatype t, void* r, const int* rc, const int* rd, MPI_Datatype, MPI_Comm comm) {
  size_t es = tsize(t);
  collective(comm, s, n * es, [&](CommInfo& c, int) {
    for (size_t j = 0; j < c.members.size(); ++j) std::memcpy(static_cast<char*>(r) + rd[j] * es, c.tmp[j].data(), rc[j] * es);
  });
  return 0;
}
int MPI_Alltoallv(const void* s, const int* sc, const int* sd, MPI_Datatype t, void* r, const int* rc, const int* rd,
                  MPI_Datatype, MPI_Comm comm) {
  CommInfo& cc = C(comm);
  size_t es = tsize(t), p = cc.members.size(), ext = 0;
  for (size_t j = 0; j < p; ++j) ext = std::max(ext, size_t(sd[j] + sc[j]));
  std::vector<int> meta(2 * p);
  for (size_t j = 0; j < p; ++j) { meta[2 * j] = sd[j]; meta[2 * j + 1] = sc[j]; }
  int me0 = rank_in(cc, t_rank);
  {
    std::lock_guard<std::mutex> lk(cc.mu);
    if (cc.ints.size() != p) { cc.tmp.resize(p); cc.ptr.resize(p); cc.ints.resize(p); }
  }
  cc.ints[me0] = meta;
  collective(comm, s, ext * es, [&](CommInfo& c, int me) {
    for (size_t j = 0; j < p; ++j) {
      int off = c.ints[j][2 * me], cnt = c.ints[j][2 * me + 1];
      if (cnt != rc[j]) { std::fprintf(stderr, "mpishim: alltoallv count mismatch\n"); std::abort(); }
      std::memcpy(static_cast<char*>(r) + rd[j] * es, c.tmp[j].data() + off * es, cnt * es);
    }
  });
  return 0;
}
int MPI_Iallreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c, MPI_Request* q) {
  MPI_Allreduce(s, r, n, t, op, c); *q = &g_done_req; return 0;
}
int MPI_Iallgather(const void* s, int n, MPI_Datatype t, void* r, int rn, MPI_Datatype rt, MPI_Comm c, MPI_Request* q) {
  MPI_Allgather(s, n, t, r, rn, rt, c); *q = &g_done_req; return 0;
}
int MPI_Ireduce_scatter_block(const void* s, void* r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm comm, MPI_Request* q) {
  CommInfo& cc = C(comm);
  size_t b = n * tsize(t), p = cc.members.size();
  collective(comm, s, b * p, [&](CommInfo& c, int me) {
    std::memcpy(r, c.tmp[0].data() + me * b, b);
    for (size_t j = 1; j < p; ++j) apply(op, t, c.tmp[j].data() + me * b, r, n);
  });
  *q = &g_done_req; return 0;
}
int MPI_Ibarrier(MPI_Comm c, MPI_Request* q) { MPI_Barrier(c); *q = &g_done_req; return 0; }

int MPI_Isend(const void* buf, int n, MPI_Datatype t, int dst, int tag, MPI_Comm comm, MPI_Request* q) {
  CommInfo& c = C(comm);
  size_t b = n * tsize(t);
  std::vector<char> msg(static_cast<const char*>(buf), static_cast<const char*>(buf) + b);
  {
    std::lock_guard<std::mutex> lk(g_mb_mu);
    g_mb[std::make_tuple(int(comm), rank_in(c, t_rank), dst, tag)].push_back(std::move(msg));
  }
  g_mb_cv.notify_all();
  *q = &g_done_req;
  return 0;
}
int MPI_Irecv(void* buf, int n, MPI_Datatype t, int src, int tag, MPI_Comm comm, MPI_Request* q) {
  auto* r = new mpishim_req;
  r->is_recv = true; r->buf = buf; r->bytes = n * tsize(t); r->src = src; r->tag = tag; r->ctx = comm;
  r->me = rank_in(C(comm), t_rank);
  *q = r;
  return 0;
}
static bool try_complete(mpishim_req* r) {
  if (r->done) return true;
  auto it = g_mb.find(std::make_tuple(r->ctx, r->src, r->me, r->tag));
  if (it == g_mb.end() || it->second.empty()) return false;
  auto& m = it->second.front();
  if (m.size() != r->bytes) { std::fprintf(stderr, "mpishim: message size mismatch (%zu vs %zu)\n", m.size(), r->bytes); std::abort(); }
  std::memcpy(r->buf, m.data(), m.size());
  it->second.pop_front();
  r->done = true;
  return true;
}
int MPI_Test(MPI_Request* q, int* flag, MPI_Status*) {
  if (*q == MPI_REQUEST_NULL || *q == &g_done_req) { *flag = 1; *q = MPI_REQUEST_NULL; return 0; }
  std::lock_guard<std::mutex> lk(g_mb_mu);
  *flag = try_complete(*q);
  if (*flag) { delete *q; *q = MPI_REQUEST_NULL; }
  return 0;
}
int MPI_Wait(MPI_Request* q, MPI_Status*) {
  if (*q == MPI_REQUEST_NULL || *q == &g_done_req) { *q = MPI_REQUEST_NULL; return 0; }
  std::unique_lock<std::mutex> lk(g_mb_mu);
  g_mb_cv.wait(lk, [&] { return try_complete(*q); });
  delete *q; *q = MPI_REQUEST_NULL;
  return 0;
}
int MPI_Waitall(int n, MPI_Request* q, MPI_Status*) { for (int i = 0; i < n; ++i) MPI_Wait(&q[i], nullptr); return 0; }
int MPI_Testall(int n, MPI_Request* q, int* flag, MPI_Status*) {
  *flag = 1;
  for (int i = 0; i < n; ++i) { int f; MPI_Test(&q[i], &f, nullptr); if (!f) *flag = 0; }
  return 0;
}
double MPI_Wtime() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  int np = 1;
  if (const char* e = std::getenv("MPISHIM_NP")) np = std::atoi(e);
  std::vector<char*> args;
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "-np" && i + 1 < argc) { np = std::atoi(argv[++i]); continue; }
    args.push_back(argv[i]);
  }
  args.push_back(nullptr);
  g_np = np;
  g_comms[0] = std::make_unique<CommInfo>();
  for (int i = 0; i < np; ++i) g_comms[0]->members.push_back(i);
  g_ncomms.store(1);
  std::vector<std::thread> th; std::vector<int> rc(np);
  for (int i = 0; i < np; ++i)
    th.emplace_back([&, i] {
      t_rank = i;
      std::vector<char*> a = args;
      rc[i] = xpsim_main(int(a.size()) - 1, a.data());
    });
  for (auto& t : th) t.join();
  int r = 0; for (int x : rc) r |= x;
  return r;
}
