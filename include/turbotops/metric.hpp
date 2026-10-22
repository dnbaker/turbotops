// SPDX-License-Identifier: MIT
// metric.hpp — Minkowski (Lp, 1 <= p <= inf) distances on f32 coordinates.
//
//   l1, l2, linf      exact SIMD kernels (sum |d|, sqrt(sum d^2), max |d|)
//   lp:<p>            1 <= p < inf. The sum is taken over (|d_k| / M)^p with
//                     M = max_k |d_k|, so no term exceeds 1 and nothing
//                     overflows for any p or coordinate scale:
//                         d_p = M * (sum_k (|d_k| / M)^p)^(1/p).
//                     Integer p <= 8 uses repeated multiplication; other p a
//                     vectorized exp2/log2 power (relative error < 1e-5).
//
// Every kernel is a pure function of its two inputs, so d(i, j) == d(j, i)
// bit for bit and results do not depend on threading.
#ifndef TURBOTOPS_METRIC_HPP
#define TURBOTOPS_METRIC_HPP

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core.hpp"
#ifndef TURBOTOPS_NO_SIMD
#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"
#endif

namespace tops {

namespace kern {

#ifndef TURBOTOPS_NO_SIMD
inline f32 hsum8(simde__m256 v) {
    alignas(32) f32 t[8];
    simde_mm256_store_ps(t, v);
    return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
}
inline f32 hmax8(simde__m256 v) {
    alignas(32) f32 t[8];
    simde_mm256_store_ps(t, v);
    f32 m = t[0];
    for (int i = 1; i < 8; ++i) m = std::max(m, t[i]);
    return m;
}
inline simde__m256 abs8(simde__m256 v) {
    return simde_mm256_and_ps(v, simde_mm256_castsi256_ps(simde_mm256_set1_epi32(0x7fffffff)));
}

// x^p for x in [0, 1], p >= 1 (exp2(p * log2 x)); 0 maps to 0.
inline simde__m256 pow01_8(simde__m256 x, simde__m256 p8) {
    const simde__m256 one = simde_mm256_set1_ps(1.0f);
    const simde__m256 two = simde_mm256_set1_ps(2.0f);
    simde__m256i xi = simde_mm256_castps_si256(x);
    simde__m256 e = simde_mm256_cvtepi32_ps(simde_mm256_srli_epi32(
        simde_mm256_and_si256(xi, simde_mm256_set1_epi32(0x7f800000)), 23));
    simde__m256 m = simde_mm256_castsi256_ps(simde_mm256_or_si256(
        simde_mm256_and_si256(xi, simde_mm256_set1_epi32(0x007fffff)),
        simde_mm256_castps_si256(one)));
    // log2(m), m in [1, 2): 2 atanh((m-1)/(m+1)) / ln 2 as an odd series.
    simde__m256 r = simde_mm256_rcp_ps(simde_mm256_add_ps(m, one));
    r = simde_mm256_mul_ps(r, simde_mm256_fnmadd_ps(simde_mm256_add_ps(m, one), r, two));
    simde__m256 t = simde_mm256_mul_ps(simde_mm256_sub_ps(m, one), r);
    simde__m256 t2 = simde_mm256_mul_ps(t, t);
    simde__m256 S = simde_mm256_set1_ps(1.0f / 13.0f);
    S = simde_mm256_fmadd_ps(t2, S, simde_mm256_set1_ps(1.0f / 11.0f));
    S = simde_mm256_fmadd_ps(t2, S, simde_mm256_set1_ps(1.0f / 9.0f));
    S = simde_mm256_fmadd_ps(t2, S, simde_mm256_set1_ps(1.0f / 7.0f));
    S = simde_mm256_fmadd_ps(t2, S, simde_mm256_set1_ps(1.0f / 5.0f));
    S = simde_mm256_fmadd_ps(t2, S, simde_mm256_set1_ps(1.0f / 3.0f));
    S = simde_mm256_fmadd_ps(t2, S, one);
    S = simde_mm256_mul_ps(S, t);
    simde__m256 lg = simde_mm256_add_ps(simde_mm256_sub_ps(e, simde_mm256_set1_ps(127.0f)),
                                        simde_mm256_mul_ps(S, simde_mm256_set1_ps(2.885390081777927f)));
    // exp2(y) with y = p * log2(x) <= 0; clamp to avoid denormal exponents.
    simde__m256 y = simde_mm256_max_ps(simde_mm256_mul_ps(p8, lg), simde_mm256_set1_ps(-126.0f));
    simde__m256 k = simde_mm256_round_ps(y, SIMDE_MM_FROUND_TO_NEAREST_INT | SIMDE_MM_FROUND_NO_EXC);
    simde__m256 f = simde_mm256_mul_ps(simde_mm256_sub_ps(y, k), simde_mm256_set1_ps(0.6931471805599453f));
    simde__m256 P = simde_mm256_set1_ps(1.0f / 720.0f);
    P = simde_mm256_fmadd_ps(f, P, simde_mm256_set1_ps(1.0f / 120.0f));
    P = simde_mm256_fmadd_ps(f, P, simde_mm256_set1_ps(1.0f / 24.0f));
    P = simde_mm256_fmadd_ps(f, P, simde_mm256_set1_ps(1.0f / 6.0f));
    P = simde_mm256_fmadd_ps(f, P, simde_mm256_set1_ps(0.5f));
    P = simde_mm256_fmadd_ps(f, P, one);
    P = simde_mm256_fmadd_ps(f, P, one);
    simde__m256i pi = simde_mm256_add_epi32(simde_mm256_castps_si256(P),
                                            simde_mm256_slli_epi32(simde_mm256_cvtps_epi32(k), 23));
    const simde__m256 tiny = simde_mm256_cmp_ps(x, simde_mm256_set1_ps(1e-30f), SIMDE_CMP_LT_OQ);
    return simde_mm256_blendv_ps(simde_mm256_castsi256_ps(pi), simde_mm256_setzero_ps(), tiny);
}
#endif

inline f32 l2(const f32* a, const f32* b, u32 dim) {
    u32 k = 0;
    f32 s = 0.0f;
#ifndef TURBOTOPS_NO_SIMD
    if (dim >= 16) {
        simde__m256 acc0 = simde_mm256_setzero_ps(), acc1 = simde_mm256_setzero_ps();
        for (; k + 16 <= dim; k += 16) {
            simde__m256 d0 = simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k), simde_mm256_loadu_ps(b + k));
            simde__m256 d1 = simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k + 8), simde_mm256_loadu_ps(b + k + 8));
            acc0 = simde_mm256_fmadd_ps(d0, d0, acc0);
            acc1 = simde_mm256_fmadd_ps(d1, d1, acc1);
        }
        s = hsum8(simde_mm256_add_ps(acc0, acc1));
    }
#endif
    for (; k < dim; ++k) {
        const f32 d = a[k] - b[k];
        s = std::fma(d, d, s);
    }
    return std::sqrt(s);
}

// d(a, p + 3t) for t < count, one vector row at a time. Bit-identical to
// l2(a, p + 3t, 3) lane by lane: same multiply/fma chain (dx*dx, then
// fma with dy*dy, then fma with dz*dz) and the same sqrt. Only used where
// SIMDe maps fmadd to a hardware fused multiply-add; elsewhere (SSE2
// without FMA) simde would emulate fmadd as mul+add, two roundings, and
// the batch would diverge from the scalar std::fma chain, so the scalar
// loop runs instead.
inline void l2_row3(const f32* a, const f32* p, u64 count, f32* out) {
    u64 t = 0;
#if !defined(TURBOTOPS_NO_SIMD) && defined(SIMDE_X86_FMA_NATIVE)
    const simde__m256 ax = simde_mm256_set1_ps(a[0]);
    const simde__m256 ay = simde_mm256_set1_ps(a[1]);
    const simde__m256 az = simde_mm256_set1_ps(a[2]);
    for (; t + 8 <= count; t += 8) {
        const f32* q = p + 3 * t;
        const simde__m256 x = simde_mm256_setr_ps(q[0], q[3], q[6], q[9], q[12], q[15], q[18], q[21]);
        const simde__m256 y = simde_mm256_setr_ps(q[1], q[4], q[7], q[10], q[13], q[16], q[19], q[22]);
        const simde__m256 z = simde_mm256_setr_ps(q[2], q[5], q[8], q[11], q[14], q[17], q[20], q[23]);
        const simde__m256 dx = simde_mm256_sub_ps(x, ax);
        const simde__m256 dy = simde_mm256_sub_ps(y, ay);
        const simde__m256 dz = simde_mm256_sub_ps(z, az);
        simde__m256 s = simde_mm256_mul_ps(dx, dx);
        s = simde_mm256_fmadd_ps(dy, dy, s);
        s = simde_mm256_fmadd_ps(dz, dz, s);
        simde_mm256_storeu_ps(out + t, simde_mm256_sqrt_ps(s));
    }
#endif
    for (; t < count; ++t) out[t] = l2(a, p + 3 * t, 3);
}

inline f32 l1(const f32* a, const f32* b, u32 dim) {
    u32 k = 0;
    f32 s = 0.0f;
#ifndef TURBOTOPS_NO_SIMD
    if (dim >= 16) {
        simde__m256 acc0 = simde_mm256_setzero_ps(), acc1 = simde_mm256_setzero_ps();
        for (; k + 16 <= dim; k += 16) {
            acc0 = simde_mm256_add_ps(acc0, abs8(simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k), simde_mm256_loadu_ps(b + k))));
            acc1 = simde_mm256_add_ps(acc1, abs8(simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k + 8), simde_mm256_loadu_ps(b + k + 8))));
        }
        s = hsum8(simde_mm256_add_ps(acc0, acc1));
    }
#endif
    for (; k < dim; ++k) s += std::fabs(a[k] - b[k]);
    return s;
}

inline f32 linf(const f32* a, const f32* b, u32 dim) {
    u32 k = 0;
    f32 m = 0.0f;
#ifndef TURBOTOPS_NO_SIMD
    if (dim >= 8) {
        simde__m256 acc = simde_mm256_setzero_ps();
        for (; k + 8 <= dim; k += 8)
            acc = simde_mm256_max_ps(acc, abs8(simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k), simde_mm256_loadu_ps(b + k))));
        m = hmax8(acc);
    }
#endif
    for (; k < dim; ++k) m = std::max(m, std::fabs(a[k] - b[k]));
    return m;
}

inline f32 ipow(f32 x, u32 p) {
    f32 r = 1.0f;
    for (; p; p >>= 1, x *= x)
        if (p & 1) r *= x;
    return r;
}

inline f32 lp(const f32* a, const f32* b, u32 dim, f32 p) {
    const f32 M = linf(a, b, dim);
    if (M == 0.0f) return 0.0f;
    const f32 inv = 1.0f / M;
    const bool integral = p == std::floor(p) && p <= 8.0f;
    u32 k = 0;
    f32 s = 0.0f;
#ifndef TURBOTOPS_NO_SIMD
    if (!integral && dim >= 8) {
        const simde__m256 p8 = simde_mm256_set1_ps(p), iv = simde_mm256_set1_ps(inv);
        simde__m256 acc = simde_mm256_setzero_ps();
        for (; k + 8 <= dim; k += 8) {
            simde__m256 x = simde_mm256_mul_ps(
                abs8(simde_mm256_sub_ps(simde_mm256_loadu_ps(a + k), simde_mm256_loadu_ps(b + k))), iv);
            acc = simde_mm256_add_ps(acc, pow01_8(simde_mm256_min_ps(x, simde_mm256_set1_ps(1.0f)), p8));
        }
        s = hsum8(acc);
    }
#endif
    for (; k < dim; ++k) {
        const f32 x = std::min(1.0f, std::fabs(a[k] - b[k]) * inv);
        s += integral ? ipow(x, u32(p)) : std::pow(x, p);
    }
    return M * std::pow(s, 1.0f / p);
}

}  // namespace kern

// ---------------------------------------------------------------------------
// Metric: a parsed Minkowski exponent.
// ---------------------------------------------------------------------------
struct Metric {
    enum Kind { L1, L2, Linf, Lp } kind = L2;
    f32 p = 2.0f;

    f32 operator()(const f32* a, const f32* b, u32 dim) const {
        switch (kind) {
            case L1: return kern::l1(a, b, dim);
            case L2: return kern::l2(a, b, dim);
            case Linf: return kern::linf(a, b, dim);
            default: return kern::lp(a, b, dim, p);
        }
    }

    static Metric minkowski(f32 p) {
        if (!(p >= 1.0f))
            throw std::invalid_argument("metric: p must be in [1, inf]");
        Metric m;
        m.p = p;
        if (p == 1.0f) m.kind = L1;
        else if (p == 2.0f) m.kind = L2;
        else if (std::isinf(p)) m.kind = Linf;
        else m.kind = Lp;
        return m;
    }

    // "l1" | "l2" | "linf" | "euclidean" | "manhattan" | "chebyshev" | "lp:<p>"
    static Metric parse(const std::string& s) {
        if (s == "l2" || s == "euclidean") return minkowski(2.0f);
        if (s == "l1" || s == "manhattan") return minkowski(1.0f);
        if (s == "linf" || s == "chebyshev") return minkowski(INFINITY);
        if (s.rfind("lp:", 0) == 0) {
            const std::string v = s.substr(3);
            if (v == "inf") return minkowski(INFINITY);
            char* end = nullptr;
            const f32 p = std::strtof(v.c_str(), &end);
            if (end && *end == '\0' && !v.empty()) return minkowski(p);
        }
        throw std::invalid_argument("metric: unknown '" + s +
                                    "' (expected l1 | l2 | linf | lp:<p>, p >= 1)");
    }

    std::string name() const {
        switch (kind) {
            case L1: return "l1";
            case L2: return "l2";
            case Linf: return "linf";
            default: {
                char buf[32];
                std::snprintf(buf, sizeof buf, "lp:%g", double(p));
                return buf;
            }
        }
    }
};

// Distances between rows of a row-major point array.
struct PointCloud {
    const f32* x = nullptr;
    u32 n = 0, dim = 0;
    Metric metric;

    f32 operator()(u32 i, u32 j) const {
        return metric(x + u64(i) * dim, x + u64(j) * dim, dim);
    }
};

}  // namespace tops

#endif  // TURBOTOPS_METRIC_HPP
