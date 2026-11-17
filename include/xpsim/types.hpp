// xpsim: distributed-memory ML training simulator
// Basic scalar types. Half-precision types are 16-bit storage structs that are
// bit-compatible with __half / __nv_bfloat16 so the same buffers can be handed
// to CUDA kernels and cuBLAS. Host-side conversions live in half_host.hpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace xpsim {

using Int = long long;

enum class DType : int { F32 = 0, BF16 = 1, F16 = 2, F64 = 3 };

struct bf16_t { uint16_t x; };
struct fp16_t { uint16_t x; };

inline size_t dtype_size(DType t) { return t == DType::F64 ? 8 : t == DType::F32 ? 4 : 2; }
inline const char* dtype_name(DType t) {
  switch (t) { case DType::F32: return "fp32"; case DType::BF16: return "bf16"; case DType::F16: return "fp16"; case DType::F64: return "fp64"; }
  return "?";
}
inline DType parse_dtype(const std::string& s) {
  if (s == "fp32" || s == "f32" || s == "float") return DType::F32;
  if (s == "fp64" || s == "f64" || s == "double") return DType::F64;
  if (s == "bf16") return DType::BF16;
  if (s == "fp16" || s == "f16" || s == "half") return DType::F16;
  throw std::runtime_error("unknown dtype: " + s);
}

template <typename T> struct DTypeOf;
template <> struct DTypeOf<float>  { static constexpr DType value = DType::F32; };
template <> struct DTypeOf<bf16_t> { static constexpr DType value = DType::BF16; };
template <> struct DTypeOf<fp16_t> { static constexpr DType value = DType::F16; };
template <> struct DTypeOf<double> { static constexpr DType value = DType::F64; };
// Master-weight / optimizer-state precision for a given compute precision.
inline DType master_dtype(DType compute) { return compute == DType::F64 ? DType::F64 : DType::F32; }

enum class Orientation { NORMAL, TRANSPOSE };
constexpr Orientation N = Orientation::NORMAL;
constexpr Orientation T = Orientation::TRANSPOSE;
constexpr Orientation TR = Orientation::TRANSPOSE;  // usable inside templates whose type parameter is T

enum class ReduceOp { SUM, MAX };

[[noreturn]] inline void fail(const std::string& msg) { throw std::runtime_error("xpsim: " + msg); }
#define XPSIM_CHECK(cond, msg) do { if (!(cond)) ::xpsim::fail(std::string(msg) + "  [" #cond "] at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

}  // namespace xpsim
