#include "xpsim/config.hpp"
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>

namespace xpsim {

const char* router_name(RouterPolicy r) {
  switch (r) {
    case RouterPolicy::UNIFORM: return "uniform";
    case RouterPolicy::ZIPF: return "zipf";
    case RouterPolicy::LOCAL: return "local";
    case RouterPolicy::REMOTE: return "remote";
    case RouterPolicy::DISTANCE: return "distance";
    case RouterPolicy::NODE_LIMITED: return "node-limited";
    case RouterPolicy::ROUND_ROBIN: return "round-robin";
  }
  return "?";
}

static void apply_preset(Config& c, const std::string& p) {
  auto set = [&](int L, int h, int nh, int nkv, int f, int s, int V) {
    c.layers = L; c.hidden = h; c.heads = nh; c.kv_heads = nkv; c.ffn = f; c.seq = s; c.vocab = V;
  };
  if (p == "tiny") { set(2, 256, 4, 4, 1024, 128, 2048); c.micro_batch = 2; }
  else if (p == "small") set(4, 512, 8, 8, 2048, 256, 8192);
  else if (p == "gpt3-1.3b") set(24, 2048, 16, 16, 8192, 2048, 50304);
  else if (p == "gpt3-13b") set(40, 5120, 40, 40, 20480, 2048, 50304);
  else if (p == "llama2-7b") set(32, 4096, 32, 32, 11008, 4096, 32000);
  else if (p == "llama3-8b") set(32, 4096, 32, 8, 14336, 8192, 128256);
  else if (p == "llama3-70b") set(80, 8192, 64, 8, 28672, 8192, 128256);
  else if (p == "mixtral-8x7b") { set(32, 4096, 32, 8, 14336, 4096, 32000); c.experts = 8; c.topk = 2; }
  else if (p == "moe-64e") { set(16, 2048, 16, 16, 8192, 4096, 32000); c.experts = 64; c.topk = 6; c.expert_ffn = 1408; }
  else if (p == "moe-tiny") { set(2, 256, 4, 4, 1024, 128, 2048); c.experts = 8; c.topk = 2; }
  else fail("unknown preset '" + p + "'");
}

Config parse_config(int argc, char** argv) {
  Config c;
  std::vector<std::pair<std::string, std::string>> kv;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) fail("unexpected argument '" + a + "' (see --help)");
    a = a.substr(2);
    std::string v;
    auto eq = a.find('=');
    if (eq != std::string::npos) { v = a.substr(eq + 1); a = a.substr(0, eq); }
    else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) v = argv[++i];
    else v = "1";
    kv.emplace_back(a, v);
  }
  for (auto& [k, v] : kv) if (k == "preset") apply_preset(c, v);  // presets first, flags override
  auto I = [](const std::string& v) { return std::stoi(v); };
  auto D = [](const std::string& v) { return std::stod(v); };
  auto B = [](const std::string& v) { return !(v == "0" || v == "false" || v == "no" || v == "off"); };
  std::map<std::string, std::function<void(const std::string&)>> h = {
      {"preset", [&](auto&) {}},
      {"help", [&](auto&) { fail("help"); }},
      {"layers", [&](auto& v) { c.layers = I(v); }}, {"hidden", [&](auto& v) { c.hidden = I(v); }},
      {"heads", [&](auto& v) { c.heads = I(v); }}, {"kv-heads", [&](auto& v) { c.kv_heads = I(v); }},
      {"ffn", [&](auto& v) { c.ffn = I(v); }}, {"seq", [&](auto& v) { c.seq = I(v); }},
      {"vocab", [&](auto& v) { c.vocab = I(v); }}, {"micro-batch", [&](auto& v) { c.micro_batch = I(v); }},
      {"microbatches", [&](auto& v) { c.microbatches = I(v); }}, {"causal", [&](auto& v) { c.causal = B(v); }},
      {"experts", [&](auto& v) { c.experts = I(v); }}, {"topk", [&](auto& v) { c.topk = I(v); }},
      {"moe-every", [&](auto& v) { c.moe_every = I(v); }}, {"expert-ffn", [&](auto& v) { c.expert_ffn = I(v); }},
      {"capacity-factor", [&](auto& v) { c.capacity_factor = D(v); }},
      {"router", [&](auto& v) {
         static const std::map<std::string, RouterPolicy> m = {
             {"uniform", RouterPolicy::UNIFORM}, {"zipf", RouterPolicy::ZIPF}, {"local", RouterPolicy::LOCAL},
             {"remote", RouterPolicy::REMOTE}, {"far", RouterPolicy::REMOTE}, {"distance", RouterPolicy::DISTANCE},
             {"node-limited", RouterPolicy::NODE_LIMITED}, {"round-robin", RouterPolicy::ROUND_ROBIN}};
         auto it = m.find(v); if (it == m.end()) fail("unknown router policy " + v); c.router = it->second; }},
      {"zipf-alpha", [&](auto& v) { c.zipf_alpha = D(v); }}, {"locality", [&](auto& v) { c.locality = D(v); }},
      {"local-scope", [&](auto& v) { if (v != "rank" && v != "node") fail("--local-scope rank|node"); c.local_scope = v; }},
      {"distance-beta", [&](auto& v) { c.distance_beta = D(v); }}, {"max-nodes", [&](auto& v) { c.max_nodes = I(v); }},
      {"topology", [&](auto& v) {
         if (v == "flat") c.topology = NetTopology::FLAT; else if (v == "ring") c.topology = NetTopology::RING;
         else if (v == "fattree") c.topology = NetTopology::FATTREE; else fail("--topology flat|ring|fattree"); }},
      {"nodes-per-switch", [&](auto& v) { c.nodes_per_switch = I(v); }},
      {"router-static", [&](auto& v) { c.router_static = B(v); }},
      {"tp", [&](auto& v) { c.tp = I(v); }}, {"pp", [&](auto& v) { c.pp = I(v); }}, {"dp", [&](auto& v) { c.dp = I(v); }},
      {"fsdp", [&](auto& v) { c.fsdp = I(v); }}, {"cp", [&](auto& v) { c.cp = I(v); }}, {"ep", [&](auto& v) { c.ep = I(v); }},
      {"zero", [&](auto& v) { c.zero = I(v); }}, {"sp", [&](auto& v) { c.sp = B(v); }},
      {"overlap", [&](auto& v) { c.overlap = B(v); }}, {"recompute", [&](auto& v) { c.recompute = B(v); }},
      {"pp-schedule", [&](auto& v) {
         if (v == "gpipe") c.schedule = PPSchedule::GPIPE; else if (v == "1f1b") c.schedule = PPSchedule::ONE_F_ONE_B;
         else fail("--pp-schedule gpipe|1f1b"); }},
      {"device", [&](auto& v) { if (v != "cpu" && v != "gpu") fail("--device cpu|gpu"); c.gpu = v == "gpu"; }},
      {"precision", [&](auto& v) { c.precision = parse_dtype(v); }},
      {"grad-dtype", [&](auto& v) { if (v != "fp32" && v != "compute") fail("--grad-dtype fp32|compute"); c.grad_fp32 = v == "fp32"; }},
      {"tf32", [&](auto& v) { c.tf32 = B(v); }}, {"loss-scale", [&](auto& v) { c.loss_scale = D(v); }},
      {"clip-grad", [&](auto& v) { c.clip_grad = D(v); }}, {"lr", [&](auto& v) { c.lr = D(v); }},
      {"transport", [&](auto& v) {
         if (v == "mpi") c.rt.transport = Transport::MPI;
         else if (v == "hier" || v == "vmm") c.rt.transport = Transport::HIER;
         else fail("--transport mpi|hier"); }},
      {"cuda-aware", [&](auto& v) { c.rt.cuda_aware = B(v); }},
      {"workers-per-node", [&](auto& v) { c.rt.workers_per_node = I(v); }},
      {"ranks-per-node", [&](auto& v) { c.rt.ranks_per_node = I(v); }},
      {"window-mb", [&](auto& v) { c.rt.window_bytes = size_t(D(v) * (1 << 20)); }},
      {"gdr", [&](auto& v) { c.rt.gpudirect_rdma = B(v); }},
      {"allow-multi-rank-per-node", [&](auto& v) { c.rt.allow_multi_rank_per_node = B(v); }},
      {"sync-collectives", [&](auto& v) { c.rt.sync_collectives = B(v); }},
      {"iters", [&](auto& v) { c.iters = I(v); }}, {"warmup", [&](auto& v) { c.warmup = I(v); }},
      {"seed", [&](auto& v) { c.seed = std::stoull(v); }}, {"peak-tflops", [&](auto& v) { c.peak_tflops = D(v); }},
      {"sync-timers", [&](auto& v) { c.sync_timers = B(v); }}, {"verbose", [&](auto& v) { c.verbose = B(v); }}, {"selftest", [&](auto& v) { c.selftest = B(v); }},
      {"json", [&](auto& v) { c.json = v; }}, {"csv", [&](auto& v) { c.csv = v; }}, {"label", [&](auto& v) { c.label = v; }},
  };
  for (auto& [k, v] : kv) {
    auto it = h.find(k);
    if (it == h.end()) fail("unknown option --" + k + " (see --help)");
    it->second(v);
  }
  if (c.kv_heads == 0) c.kv_heads = c.heads;
  if (c.ffn == 0) c.ffn = 4 * c.hidden;
  if (c.expert_ffn == 0) c.expert_ffn = c.ffn;
  if (c.loss_scale == 0) c.loss_scale = c.precision == DType::F16 ? 65536.0 : 1.0;
  if (c.precision == DType::F64) c.grad_fp32 = false;  // fp64 grads
  if (c.selftest && c.tp * c.pp * c.dp * c.fsdp * c.cp == 1) c.dp = -1;  // any world size
  c.rt.gpu = c.gpu;
  return c;
}

void validate(const Config& c, int world) {
  auto req = [](bool ok, const std::string& m) { if (!ok) fail("invalid configuration: " + m); };
  req(c.world() == world, "tp*pp*dp*fsdp*cp = " + std::to_string(c.world()) + " but " + std::to_string(world) +
                              " workers are running");
  req(c.hidden % c.heads == 0, "hidden must be divisible by heads");
  req(c.heads % c.kv_heads == 0, "heads must be divisible by kv-heads (GQA groups)");
  req(c.heads % c.tp == 0 && c.kv_heads % c.tp == 0, "heads and kv-heads must be divisible by tp");
  req(c.ffn % c.tp == 0, "ffn must be divisible by tp");
  req(c.vocab % c.tp == 0, "vocab must be divisible by tp (vocab-parallel embedding/head)");
  req(c.seq % c.cp == 0, "seq must be divisible by cp");
  const long long tok = (long long)c.micro_batch * c.seq / c.cp;
  req(tok % c.tp == 0, "micro-batch*seq/cp must be divisible by tp (sequence parallel / MoE token slicing)");
  req(c.layers >= c.pp, "need at least one layer per pipeline stage");
  req(c.microbatches >= 1, "need >= 1 microbatch");
  req(c.zero_stage() >= 0 && c.zero_stage() <= 3, "zero stage must be 0..3");
  if (c.experts > 0) {
    req(c.experts % c.ep == 0, "experts must be divisible by ep");
    req(c.dp_total() % c.ep == 0, "ep must divide dp*fsdp (expert parallelism is folded into data parallelism)");
    req(c.topk >= 1 && c.topk <= c.experts, "topk must be in [1, experts]");
  } else req(c.ep == 1, "ep > 1 needs --experts");
  req(c.iters >= 1, "iters must be >= 1");
}

std::string usage() {
  return R"(xpsim - distributed-memory ML training simulator (MPI + C++, CPU or CUDA)

usage: mpirun -np <W> ./xpsim [options]        (transport mpi: one rank per worker)
       mpirun -np <nodes> ./xpsim --transport hier [options]   (one rank per node)

model     --preset tiny|small|gpt3-1.3b|gpt3-13b|llama2-7b|llama3-8b|llama3-70b|mixtral-8x7b|moe-64e|moe-tiny
          --layers L --hidden H --heads A --kv-heads G --ffn F --seq S --vocab V
          --micro-batch b --microbatches m --causal 0|1
MoE       --experts E --topk k --moe-every n --expert-ffn F --capacity-factor c (0 = dropless)
router    --router uniform|zipf|local|remote|distance|node-limited|round-robin
          --zipf-alpha a       popularity skew (zipf)
          --locality p         prob. of drawing from the near set (local) / far set (remote)
          --local-scope rank|node   near set = experts on own rank, or on own node
          --distance-beta b    P(expert) ~ exp(-b*hops(node_src,node_dst)); b<0 prefers far
          --max-nodes M        node-limited routing: experts of a token span <= M nodes
          --topology flat|ring|fattree --nodes-per-switch n   hop model for distances
          --router-static      reuse the same routing every iteration
parallel  --tp --pp --dp --fsdp --cp --ep  (world = pp*dp*fsdp*cp*tp; EP folded into dp*fsdp)
          --zero 0|1|2|3 (default 3 if fsdp>1)  --sp  --overlap  --recompute
          --pp-schedule gpipe|1f1b
device    --device cpu|gpu --precision fp32|bf16|fp16|fp64 --grad-dtype fp32|compute --tf32
          --loss-scale s --clip-grad g --lr x
transport --transport mpi|hier  --cuda-aware 0|1  --workers-per-node n  --ranks-per-node n
          --window-mb MB  --gdr 0|1  --allow-multi-rank-per-node  --sync-collectives
run       --iters N --warmup W --seed s --peak-tflops T (0 = measure) --sync-timers 0|1
          --json file --csv file --label name --verbose
          --selftest           verify all collectives / P2P / DistMatrix redistributions and exit
)";
}

std::string describe(const Config& c) {
  std::ostringstream o;
  o << "model: L=" << c.layers << " h=" << c.hidden << " heads=" << c.heads << " kv=" << c.kv_heads
    << " ffn=" << c.ffn << " seq=" << c.seq << " vocab=" << c.vocab;
  if (c.experts) o << " experts=" << c.experts << " top" << c.topk << " every=" << c.moe_every << " eff=" << c.expert_ffn
                   << " cf=" << c.capacity_factor;
  o << "\nbatch: micro=" << c.micro_batch << " x " << c.microbatches << " microbatches x dp " << c.dp_total()
    << " = " << c.global_batch() << " seqs (" << c.tokens_per_iter() << " tokens/iter)";
  o << "\nparallel: tp=" << c.tp << (c.sp ? "+sp" : "") << " pp=" << c.pp << "("
    << (c.schedule == PPSchedule::GPIPE ? "gpipe" : "1f1b") << ") dp=" << c.dp << " fsdp=" << c.fsdp << " cp=" << c.cp
    << " ep=" << c.ep << " zero=" << c.zero_stage() << (c.overlap ? " overlap" : "") << (c.recompute ? " recompute" : "");
  if (c.experts)
    o << "\nrouter: " << router_name(c.router) << " (alpha=" << c.zipf_alpha << " locality=" << c.locality << "/"
      << c.local_scope << " beta=" << c.distance_beta << " max-nodes=" << c.max_nodes << " topo="
      << (c.topology == NetTopology::FLAT ? "flat" : c.topology == NetTopology::RING ? "ring" : "fattree") << ")";
  o << "\ndevice: " << (c.gpu ? "gpu" : "cpu") << " precision=" << dtype_name(c.precision)
    << " grads=" << (c.grad_fp32 ? "fp32" : dtype_name(c.precision)) << " master=" << dtype_name(master_dtype(c.precision))
    << (c.tf32 ? " tf32" : "") << " transport=" << (c.rt.transport == Transport::MPI ? "mpi" : "hier")
    << (c.rt.transport == Transport::MPI && c.gpu ? (c.rt.cuda_aware ? "(cuda-aware)" : "(host-staged)") : "");
  return o.str();
}

}  // namespace xpsim
