// Host-side (CPU) conversions for the 16-bit storage types. Only included by
// translation units compiled by the host C++ compiler (not nvcc).
#pragma once
#include <cstring>
#include "xpsim/types.hpp"

namespace xpsim {

inline float to_float(float v) { return v; }
inline float to_float(double v) { return float(v); }
inline float to_float(bf16_t v) {
  uint32_t u = uint32_t(v.x) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
inline float to_float(fp16_t v) {
  uint32_t h = v.x, sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ffu, u;
  if (exp == 0) {
    if (man == 0) u = sign;
    else {  // subnormal
      int e = -1; do { ++e; man <<= 1; } while (!(man & 0x400u));
      u = sign | uint32_t(127 - 15 - e) << 23 | (man & 0x3ffu) << 13;
    }
  } else if (exp == 31) u = sign | 0x7f800000u | man << 13;
  else u = sign | (exp + 112) << 23 | man << 13;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Exact widening (keeps fp64 values intact; 16-bit types widen to fp32).
inline double to_float_exact(double v) { return v; }
inline float to_float_exact(float v) { return v; }
inline float to_float_exact(bf16_t v) { return to_float(v); }
inline float to_float_exact(fp16_t v) { return to_float(v); }

template <typename T> inline T from_float(float f);
template <> inline float from_float<float>(float f) { return f; }
template <> inline double from_float<double>(float f) { return f; }
template <> inline bf16_t from_float<bf16_t>(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return bf16_t{uint16_t((u >> 16) | 0x40)};  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);                                                    // RNE
  return bf16_t{uint16_t(u >> 16)};
}
template <> inline fp16_t from_float<fp16_t>(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u; int32_t exp = int32_t((u >> 23) & 0xff) - 127 + 15;
  uint32_t man = u & 0x7fffffu;
  if (((u >> 23) & 0xff) == 0xff) return fp16_t{uint16_t(sign | 0x7c00u | (man ? 0x200u : 0))};
  if (exp >= 31) return fp16_t{uint16_t(sign | 0x7c00u)};  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return fp16_t{uint16_t(sign)};
    man |= 0x800000u; uint32_t shift = uint32_t(14 - exp);
    uint32_t h = man >> shift, rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return fp16_t{uint16_t(sign | h)};
  }
  uint32_t h = sign | uint32_t(exp) << 10 | man >> 13, rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return fp16_t{uint16_t(h)};
}

}  // namespace xpsim
