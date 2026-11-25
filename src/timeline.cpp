// Timeline export: Chrome trace-event JSON (open in https://ui.perfetto.dev or
// chrome://tracing). One track per worker (pid = node, tid = worker), sorted by
// pipeline stage so that pipeline bubbles line up vertically.
//   <prefix>.w<k>.json  per worker      <prefix>.json  merged by worker 0
// Events: pipeline ops ("F mb3 c1", "B ..."), gradient sync, optimizer step,
// compute kernels by category, communication post/wait (exposed time),
// replay segments, and one span per iteration.
#include <cstdio>
#include <fstream>
#include <sstream>
#include "xpsim/grid.hpp"

namespace xpsim {

namespace {
std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o;
}
}  // namespace

void write_timeline(const Config& cfg, const ProcessGrid& g) {
  if (cfg.timeline.empty()) return;
  auto& tl = metrics().timeline;
  const int w = my_worker(), node = node_of(w);
  const std::string mine = cfg.timeline + ".w" + std::to_string(w) + ".json";
  {
    std::ofstream o(mine);
    if (!o) fail("cannot write timeline file " + mine);
    char label[160];
    std::snprintf(label, sizeof label, "w%d  pp%d tp%d cp%d dp%d/%d", w, g.me.pp, g.me.t, g.me.c, g.me.dr, g.me.ds);
    o << "{\"traceEvents\":[\n";
    o << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << node << ",\"args\":{\"name\":\"node " << node << "\"}}\n";
    o << ",{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" << node << ",\"args\":{\"sort_index\":" << node << "}}\n";
    o << ",{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << node << ",\"tid\":" << w << ",\"args\":{\"name\":\"" << esc(label) << "\"}}\n";
    o << ",{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":" << node << ",\"tid\":" << w
      << ",\"args\":{\"sort_index\":" << (g.me.pp * 100000 + w) << "}}\n";
    for (const TLEvent& e : tl.ev) {
      std::string name = e.name;
      const char* kind = e.cat == NCAT ? "pipeline" : is_comm(Cat(e.cat)) ? "comm" : "compute";
      if (e.cat == NCAT && name == "iteration") name += " " + std::to_string(e.mb);
      else if (e.mb >= 0) { name += " mb" + std::to_string(e.mb); if (cfg.pp_virtual > 1 && e.chunk >= 0) name += " c" + std::to_string(e.chunk); }
      if (e.phase == 'w') name += " (wait)";
      o << ",{\"name\":\"" << esc(name) << "\",\"cat\":\"" << kind << "\",\"ph\":\"X\",\"ts\":"
        << (e.t0 - tl.epoch) * 1e6 << ",\"dur\":" << e.dur * 1e6 << ",\"pid\":" << node << ",\"tid\":" << w;
      o << ",\"args\":{";
      bool first = true;
      auto arg = [&](const char* k, const std::string& v) { o << (first ? "" : ",") << "\"" << k << "\":" << v; first = false; };
      if (e.mb >= 0 && name.rfind("iteration", 0) != 0) { arg("microbatch", std::to_string(e.mb)); arg("chunk", std::to_string(e.chunk)); }
      if (e.cat < NCAT) arg("category", "\"" + std::string(cat_name(Cat(e.cat))) + "\"");
      if (e.phase) arg("phase", e.phase == 'p' ? "\"post\"" : "\"wait\"");
      o << "}}\n";
    }
    o << "]}\n";
  }
  const size_t nev = tl.ev.size();
  Comm::world().barrier();
  if (w != 0) return;
  // merge (needs a file system shared by all nodes; per-worker files stay either way)
  std::ofstream out(cfg.timeline + ".json");
  out << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n";
  bool first = true;
  int missing = 0;
  for (int k = 0; k < world_size(); ++k) {
    std::ifstream in(cfg.timeline + ".w" + std::to_string(k) + ".json");
    if (!in) { ++missing; continue; }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == ']' || line.rfind("{\"traceEvents\"", 0) == 0) continue;
      if (line[0] == ',') line.erase(0, 1);
      out << (first ? "" : ",") << line << "\n";
      first = false;
    }
  }
  out << "]}\n";
  std::printf("xpsim: timeline written to %s.json (%d workers, %zu events on worker 0%s); open in ui.perfetto.dev\n",
              cfg.timeline.c_str(), world_size() - missing, nev,
              missing ? ", some per-worker files not visible: merge them on a shared file system" : "");
  std::fflush(stdout);
}

}  // namespace xpsim
