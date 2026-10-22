// SPDX-License-Identifier: MIT
// rips.hpp — implicit Vietoris–Rips persistent cohomology in any dimension.
//
// Algorithm: the cohomology reduction of Ripser (U. Bauer, "Ripser: efficient
// computation of Vietoris–Rips persistence barcodes", J. Appl. Comput. Topol.
// 2021): simplices are addressed by the combinatorial number system and never
// stored as vertex lists; coboundary columns are enumerated on demand and
// reduced in a binary heap with lazy mod-2 cancellation; reduced columns are
// stored as lists of the columns added to them; apparent and emergent pairs
// skip reduction entirely; clearing removes every column that is already a
// pivot of the previous dimension.
//
// Two complex policies share one reducer:
//   DenseComplex  — compressed lower-triangular distance matrix (any metric);
//                   with an infinite threshold the filtration is cut at the
//                   enclosing radius, beyond which the complex is a cone.
//   SparseComplex — a CsrGraph (rows sorted by neighbor id); cofacets are
//                   the k-way intersection of the vertices' neighbor rows.
//
// Index width: simplex indices are u64 when C(n, maxdim + 2) fits, otherwise
// unsigned __int128. A complex whose indices do not fit in 128 bits is
// rejected with std::overflow_error.
//
// Filtration order: (diameter ascending, index descending). Persistence pairs
// of zero length are not reported.
#ifndef TURBOTOPS_RIPS_HPP
#define TURBOTOPS_RIPS_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "core.hpp"
#ifndef TURBOTOPS_NO_SIMD
#include "simde/x86/avx2.h"
#endif

namespace tops::rips {

using u128 = unsigned __int128;

// ---------------------------------------------------------------------------
// Barcode: finite and essential (death = +inf) bars per dimension.
// ---------------------------------------------------------------------------
struct Barcode {
    std::vector<std::vector<f32>> birth, death;

    void ensure(u32 dim) {
        if (birth.size() <= dim) {
            birth.resize(dim + 1);
            death.resize(dim + 1);
        }
    }
    void add(u32 dim, f32 b, f32 d) {
        ensure(dim);
        birth[dim].push_back(b);
        death[dim].push_back(d);
    }
    u32 ndims() const { return u32(birth.size()); }
};

// ---------------------------------------------------------------------------
// Binomial coefficients C(i, k) for 0 <= i <= n, 0 <= k <= K, overflow-checked.
// ---------------------------------------------------------------------------
template <typename I>
struct Binomial {
    u32 n = 0, K = 0;
    std::vector<I> tab;   // column-major: C(i, k) at k * (n + 1) + i

    Binomial(u32 nn, u32 KK) : n(nn), K(KK), tab(u64(nn + 1) * (KK + 1), I(0)) {
        for (u32 i = 0; i <= n; ++i) {
            at(i, 0) = 1;
            for (u32 k = 1; k <= std::min(i, K); ++k) {
                I s;
                if (__builtin_add_overflow(at(i - 1, k - 1), at(i - 1, k), &s))
                    throw std::overflow_error("rips: simplex index overflow");
                at(i, k) = s;
            }
        }
    }
    I& at(u32 i, u32 k) { return tab[u64(k) * (n + 1) + i]; }
    I operator()(u32 i, u32 k) const {
        return k > K ? I(0) : tab[u64(k) * (n + 1) + i];
    }
    const I* column(u32 k) const { return tab.data() + u64(k) * (n + 1); }
};

// True when every simplex index up to dimension maxdim + 1 fits in I with the
// all-ones value left free as a sentinel.
template <typename I>
inline bool index_fits(u32 n, u32 maxdim) {
    try {
        Binomial<I> b(n, maxdim + 2);
        return b(n, maxdim + 2) < std::numeric_limits<I>::max();
    } catch (const std::overflow_error&) {
        return false;
    }
}

// Packed: 12 bytes for 64-bit indices, 20 for 128-bit.
template <typename I>
struct __attribute__((packed)) Entry {
    I idx;
    f32 d;
};

// a precedes b in the filtration
template <typename I>
inline bool precedes(const Entry<I>& a, const Entry<I>& b) {
    return a.d < b.d || (a.d == b.d && a.idx > b.idx);
}

template <typename I>
inline constexpr I kNone = std::numeric_limits<I>::max();

// ---------------------------------------------------------------------------
// Heap whose top is the earliest entry in filtration order; mod-2
// cancellation happens when equal entries meet at the top.
// ---------------------------------------------------------------------------
template <typename I>
struct ColumnHeap {
    std::vector<Entry<I>> h;

    bool empty() const { return h.empty(); }
    void clear() { h.clear(); }

    void push(const Entry<I>& e) {
        h.push_back(e);
        size_t i = h.size() - 1;
        while (i > 0) {
            size_t p = (i - 1) >> 1;
            if (!precedes(h[i], h[p])) break;
            std::swap(h[i], h[p]);
            i = p;
        }
    }
    // Bulk-load a column: identical pop sequence to pushing the same entries
    // one by one (a binary heap always pops in filtration order regardless of
    // its internal arrangement), at O(n) instead of O(n log n).
    void heapify(std::vector<Entry<I>>& v) {
        std::make_heap(v.begin(), v.end(),
                       [](const Entry<I>& a, const Entry<I>& b) { return precedes(b, a); });
        h.swap(v);
    }
    void pop() {
        h.front() = h.back();
        h.pop_back();
        const size_t n = h.size();
        size_t i = 0;
        for (;;) {
            size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < n && precedes(h[l], h[m])) m = l;
            if (r < n && precedes(h[r], h[m])) m = r;
            if (m == i) break;
            std::swap(h[i], h[m]);
            i = m;
        }
    }
    // Remove and return the surviving earliest entry (kNone if the column is
    // zero).
    Entry<I> pop_pivot() {
        Entry<I> piv{kNone<I>, 0.0f};
        while (!h.empty()) {
            if (piv.idx == kNone<I>) {
                piv = h.front();
                pop();
            } else if (h.front().idx == piv.idx) {
                pop();
                piv.idx = kNone<I>;
            } else {
                break;
            }
        }
        return piv;
    }
    Entry<I> get_pivot() {
        Entry<I> p = pop_pivot();
        if (p.idx != kNone<I>) push(p);
        return p;
    }
};

// ---------------------------------------------------------------------------
// Open-addressing map: simplex index -> column number.
// ---------------------------------------------------------------------------
template <typename I>
struct PivotMap {
    std::vector<I> keys;
    std::vector<u64> vals;
    u64 mask = 0, count = 0;

    PivotMap() { reset(1024); }
    void reset(u64 cap) {
        u64 c = 16;
        while (c < cap) c <<= 1;
        keys.assign(c, kNone<I>);
        vals.assign(c, 0);
        mask = c - 1;
        count = 0;
    }
    static u64 hash(I k) {
        u64 x = u64(k);
        if constexpr (sizeof(I) > 8) x ^= u64(k >> 64) * 0xC2B2AE3D27D4EB4Full;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return x;
    }
    u64 find(I k) const {
        for (u64 s = hash(k) & mask;; s = (s + 1) & mask) {
            if (keys[s] == k) return vals[s];
            if (keys[s] == kNone<I>) return ~0ull;
        }
    }
    bool contains(I k) const { return find(k) != ~0ull; }
    void insert(I k, u64 v) {
        if (2 * (count + 1) > keys.size()) grow();
        u64 s = hash(k) & mask;
        while (keys[s] != kNone<I> && keys[s] != k) s = (s + 1) & mask;
        if (keys[s] == kNone<I>) ++count;
        keys[s] = k;
        vals[s] = v;
    }
    void grow() {
        std::vector<I> ok;
        std::vector<u64> ov;
        ok.swap(keys);
        ov.swap(vals);
        reset(ok.size() * 2);
        for (u64 i = 0; i < ok.size(); ++i)
            if (ok[i] != kNone<I>) insert(ok[i], ov[i]);
    }
};

// ---------------------------------------------------------------------------
// Complex policies.
//   dist(i, j)                  — distance, +inf if the edge is absent
//   cofacets(vd, k, diam, all, cap, emit)
//       vd: the k vertices of a simplex in DESCENDING order; enumerates the
//       cofacets with diameter <= cap (cap <= threshold) in descending order
//       of the added vertex w, calling emit(w, t, cofacet_diameter) where t
//       is the number of vertices of vd greater than w. all == false
//       restricts to w > vd[0]. emit returns false to stop.
// ---------------------------------------------------------------------------

// Dense distances in one of two layouts:
//   full  — n x n row-major; each simplex vertex streams its own contiguous
//           row, so cofacet scans vectorize over blocks of candidates;
//   lower — compressed lower triangle, d(i, j) for i > j at i (i - 1) / 2 + j
//           (half the memory; used when the full matrix exceeds the budget).
// Both layouts store identical values, so results do not depend on the
// choice.
struct DenseComplex {
    u32 n = 0;
    f32 threshold = INFINITY;
    bool full = false;
    std::vector<f32> mat;

    f32 dist(u32 i, u32 j) const {
        if (i == j) return 0.0f;
        if (full) return mat[u64(i) * n + j];
        if (i < j) std::swap(i, j);
        return mat[u64(i) * (i - 1) / 2 + j];
    }

    template <typename Emit>
    void cofacets(const u32* vd, u32 k, f32 diam, bool all, f32 cap, Emit&& emit) const {
        if (full) cofacets_full(vd, k, diam, all, cap, emit);
        else cofacets_lower(vd, k, diam, all, cap, emit);
    }

    template <typename Emit>
    void cofacets_full(const u32* vd, u32 k, f32 diam, bool all, f32 cap, Emit&& emit) const {
        constexpr u32 BLK = 64;
        constexpr u32 KMAX = 32;
        const f32* row[KMAX];
        for (u32 i = 0; i < k; ++i) row[i] = mat.data() + u64(vd[i]) * n;
        alignas(64) f32 mx[BLK];
        const u32 stop = all ? 0 : vd[0] + 1;   // candidates w in [stop, n)
        u32 t = 0;
#ifndef TURBOTOPS_NO_SIMD
        // 8-wide: max over the k rows, compare with cap, skip empty blocks.
        // cap >= diam, so cd <= cap iff the row max is <= cap.
        {
            const simde__m256 capv = simde_mm256_set1_ps(cap);
            u32 hi = n;
            while (hi >= stop + 8) {
                const u32 lo = hi - 8;
                simde__m256 m = simde_mm256_loadu_ps(row[0] + lo);
                for (u32 i = 1; i < k; ++i)
                    m = simde_mm256_max_ps(m, simde_mm256_loadu_ps(row[i] + lo));
                u32 mask = u32(simde_mm256_movemask_ps(
                    simde_mm256_cmp_ps(m, capv, SIMDE_CMP_LE_OQ)));
                if (mask) {
                    simde_mm256_store_ps(mx, m);
                    for (u32 j = 8; j-- > 0;) {
                        if (!((mask >> j) & 1u)) continue;
                        const u32 w = lo + j;
                        while (t < k && vd[t] > w) ++t;
                        if (t < k && vd[t] == w) continue;
                        if (!emit(w, t, std::max(diam, mx[j]))) return;
                    }
                }
                hi = lo;
            }
            for (u32 w = hi; w-- > stop;) {
                f32 m = row[0][w];
                for (u32 i = 1; i < k; ++i) m = std::max(m, row[i][w]);
                if (m > cap) continue;
                while (t < k && vd[t] > w) ++t;
                if (t < k && vd[t] == w) continue;
                if (!emit(w, t, std::max(diam, m))) return;
            }
            return;
        }
#endif
        for (u32 hi = n; hi > stop;) {
            const u32 lo = hi - std::min<u32>(BLK, hi - stop);
            const u32 len = hi - lo;
            const f32* r0 = row[0] + lo;
            for (u32 j = 0; j < len; ++j) mx[j] = r0[j];
            for (u32 i = 1; i < k; ++i) {
                const f32* ri = row[i] + lo;
                for (u32 j = 0; j < len; ++j) mx[j] = std::max(mx[j], ri[j]);
            }
            for (u32 j = len; j-- > 0;) {
                const u32 w = lo + j;
                while (t < k && vd[t] > w) ++t;
                if (t < k && vd[t] == w) continue;
                const f32 cd = std::max(diam, mx[j]);
                if (cd <= cap && !emit(w, t, cd)) return;
            }
            hi = lo;
        }
    }

    template <typename Emit>
    void cofacets_lower(const u32* vd, u32 k, f32 diam, bool all, f32 cap, Emit&& emit) const {
        u32 t = 0;
        for (i64 wi = i64(n) - 1; wi >= 0; --wi) {
            const u32 w = u32(wi);
            while (t < k && vd[t] > w) ++t;
            if (!all && t > 0) return;
            if (t < k && vd[t] == w) continue;
            f32 cd = diam;
            const u64 row = u64(w) * (w - 1) / 2;
            bool ok = true;
            for (u32 i = 0; i < k; ++i) {
                const u32 v = vd[i];
                const f32 dv = v < w ? mat[row + v] : mat[u64(v) * (v - 1) / 2 + w];
                if (dv > cd) {
                    cd = dv;
                    if (cd > cap) { ok = false; break; }
                }
            }
            if (ok && !emit(w, t, cd)) return;
        }
    }

    // Enclosing radius: min over i of max over j of d(i, j).
    f32 enclosing_radius() const {
        std::vector<f32> rowmax(n, 0.0f);
        parallel_for(n, [&](u64 i) {
            f32 m = 0.0f;
            for (u32 j = 0; j < n; ++j) m = std::max(m, dist(u32(i), j));
            rowmax[i] = m;
        });
        f32 best = INFINITY;
        for (f32 m : rowmax) best = std::min(best, m);
        return best;
    }
};

// Bytes needed by the full and lower layouts.
inline u64 dense_bytes(u32 n, bool full) {
    return full ? u64(n) * n * sizeof(f32) : u64(n) * (n > 0 ? n - 1 : 0) / 2 * sizeof(f32);
}

// Build a dense complex from any symmetric distance functor d(i, j). The
// full layout is used when it fits in budget_bytes; otherwise the lower
// layout (throws std::length_error if neither fits).
template <typename Dist>
inline DenseComplex make_dense(u32 n, Dist&& d, u64 budget_bytes = ~0ull) {
    DenseComplex c;
    c.n = n;
    if (dense_bytes(n, true) <= budget_bytes) {
        c.full = true;
        c.mat.assign(u64(n) * n, 0.0f);
        parallel_for(n, [&](u64 i) {
            for (u32 j = 0; j < i; ++j) c.mat[i * n + j] = d(u32(i), j);
        });
        parallel_for(n, [&](u64 i) {
            for (u32 j = u32(i) + 1; j < n; ++j) c.mat[i * n + j] = c.mat[u64(j) * n + i];
        });
    } else if (dense_bytes(n, false) <= budget_bytes) {
        c.mat.resize(dense_bytes(n, false) / sizeof(f32));
        parallel_for(n, [&](u64 i) {
            const u64 row = i * (i - 1) / 2;
            for (u32 j = 0; j < i; ++j) c.mat[row + j] = d(u32(i), j);
        });
    } else {
        throw std::length_error(
            "rips: the dense distance matrix exceeds the memory budget; use a "
            "finite threshold or a sparse graph");
    }
    return c;
}

struct SparseComplex {
    const CsrGraph* g = nullptr;
    u32 n = 0;
    f32 threshold = INFINITY;

    explicit SparseComplex(const CsrGraph& gr, f32 thr = INFINITY)
        : g(&gr), n(gr.n), threshold(thr) {}

    f32 dist(u32 i, u32 j) const { return i == j ? 0.0f : g->weight(i, j); }

    template <typename Emit>
    void cofacets(const u32* vd, u32 k, f32 diam, bool all, f32 cap, Emit&& emit) const {
        constexpr u32 KMAX = 32;
        u64 p[KMAX], lo[KMAX];
        const u32* col = g->col_idx.data();
        const f32* wt = g->weights.data();
        for (u32 i = 0; i < k; ++i) {
            lo[i] = g->begin(vd[i]);
            u64 e = g->end(vd[i]);
            if (e == lo[i]) return;
            p[i] = e - 1;
        }
        u32 t = 0;
        for (;;) {
            u32 x = col[p[0]];
            bool agree = true;
            for (u32 i = 1; i < k; ++i) {
                while (col[p[i]] > x) {
                    if (p[i] == lo[i]) return;
                    --p[i];
                }
                if (col[p[i]] < x) {
                    x = col[p[i]];
                    agree = false;
                    break;
                }
            }
            if (!agree) {
                while (col[p[0]] > x) {
                    if (p[0] == lo[0]) return;
                    --p[0];
                }
                continue;
            }
            const u32 w = x;
            while (t < k && vd[t] > w) ++t;
            if (!all && t > 0) return;
            f32 cd = diam;
            for (u32 i = 0; i < k; ++i) cd = std::max(cd, wt[p[i]]);
            if (cd <= cap && !emit(w, t, cd)) return;
            if (p[0] == lo[0]) return;
            --p[0];
        }
    }
};

// ---------------------------------------------------------------------------
// Statistics (optional).
// ---------------------------------------------------------------------------
struct RipsStats {
    bool wide_index = false;     // u128 simplex indices
    f32 threshold = INFINITY;    // effective threshold
    u64 edges = 0;
    std::vector<u64> columns;    // columns reduced per dimension
    std::vector<u64> apparent;   // columns skipped as apparent/emergent
    std::vector<u64> emergent;   // columns closed by the emergent-pair shortcut
};

// ---------------------------------------------------------------------------
// Reducer.
// ---------------------------------------------------------------------------
template <typename I, typename Cx>
struct Reducer {
    using E = Entry<I>;
    static constexpr u32 KMAX = 32;

    const Cx& cx;
    const u32 n, maxdim;
    const Binomial<I> B;
    Barcode& out;
    RipsStats& st;

    Reducer(const Cx& c, u32 md, Barcode& o, RipsStats& s)
        : cx(c), n(c.n), maxdim(md), B(c.n, md + 2), out(o), st(s) {
        init_kfact();
    }

    // ---- simplex index arithmetic -------------------------------------
    // Largest v < top with C(v, k) <= idx. Initial guess from
    // C(v, k) ~ (v - (k - 1) / 2)^k / k!, exact galloping fix-up, bisection
    // to finish. The guess only affects speed, never the result.
    u32 max_vertex(I idx, u32 k, u32 top) const {
        const I* col = B.column(k);
        u32 lo = k - 1, hi = top;   // C(lo, k) <= idx; C(hi, k) > idx or hi == top
        // No transcendental calls: closed-form roots for k <= 4; for k >= 5 no
        // guess (g = -1 skips straight to the exact bisection below).
        const double x = kfact[k] * double(idx);
        const double g = k == 1 ? x + 0.0 : k == 2 ? std::sqrt(x) + 0.5 : k == 3 ? std::cbrt(x) + 1.0
                         : k == 4 ? std::sqrt(std::sqrt(x)) + 1.5 : -1.0;
        if (g > double(lo) && g < double(hi)) {
            const u32 v = u32(g);
            if (col[v] <= idx) {
                lo = v;
                for (u32 s = 1; hi - lo > 1; s <<= 1) {
                    const u32 nx = u32(std::min<u64>(u64(lo) + s, hi - 1));
                    if (col[nx] <= idx) lo = nx;
                    else { hi = nx; break; }
                }
            } else {
                hi = v;
                for (u32 s = 1; hi - lo > 1; s <<= 1) {
                    const u32 nx = s >= hi - lo ? lo + 1 : hi - s;
                    if (col[nx] > idx) hi = nx;
                    else { lo = nx; break; }
                }
            }
        }
        while (hi - lo > 1) {
            const u32 mid = lo + (hi - lo) / 2;
            if (col[mid] <= idx) lo = mid;
            else hi = mid;
        }
        return lo;
    }
    double kfact[KMAX + 1];   // k!
    void init_kfact() {
        kfact[0] = 1.0;
        for (u32 k = 1; k <= KMAX; ++k) kfact[k] = kfact[k - 1] * k;
    }
    // vertices of a simplex with nv vertices, descending
    void decode(I idx, u32 nv, u32* vd) const {
        if (nv == 2 && idx < (I(1) << 52)) {   // closed form, exact in double
            const double x = double(u64(idx));
            u64 v = u64((1.0 + std::sqrt(1.0 + 8.0 * x)) * 0.5);
            while (v * (v - 1) / 2 > u64(idx)) --v;
            while ((v + 1) * v / 2 <= u64(idx)) ++v;
            vd[0] = u32(v);
            vd[1] = u32(u64(idx) - v * (v - 1) / 2);
            return;
        }
        u32 top = n;
        for (u32 i = 0; i < nv; ++i) {
            const u32 pos = nv - i;
            const u32 v = max_vertex(idx, pos, top);
            vd[i] = v;
            idx -= B(v, pos);
            top = v;
        }
    }

    // ---- cofacets of (vd, nv vertices, diam) as entries ---------------
    template <typename F>
    void cofacet_entries(const u32* vd, u32 nv, f32 diam, bool all, f32 cap, F&& f) const {
        // A[t] = sum_{i<t} C(vd[i], nv+1-i); Bs[t] = sum_{i>=t} C(vd[i], nv-i)
        I A[KMAX + 1], Bs[KMAX + 1];
        A[0] = 0;
        for (u32 i = 0; i < nv; ++i) A[i + 1] = A[i] + B(vd[i], nv + 1 - i);
        Bs[nv] = 0;
        for (u32 i = nv; i-- > 0;) Bs[i] = Bs[i + 1] + B(vd[i], nv - i);
        cx.cofacets(vd, nv, diam, all, cap, [&](u32 w, u32 t, f32 cd) {
            return f(E{A[t] + B(w, nv + 1 - t) + Bs[t], cd});
        });
    }

    // Zero pivot cofacet of the simplex with vertices vd (nv, descending) and
    // diameter d: the first cofacet in descending index order whose diameter
    // is d. Its vertices are written to cv (nv + 1, descending).
    E zero_pivot_cofacet_v(const u32* vd, u32 nv, f32 d, u32* cv) const {
        E found{kNone<I>, 0.0f};
        u32 fw = 0, ft = 0;
        I A[KMAX + 1], Bs[KMAX + 1];
        A[0] = 0;
        for (u32 i = 0; i < nv; ++i) A[i + 1] = A[i] + B(vd[i], nv + 1 - i);
        Bs[nv] = 0;
        for (u32 i = nv; i-- > 0;) Bs[i] = Bs[i + 1] + B(vd[i], nv - i);
        cx.cofacets(vd, nv, d, true, d, [&](u32 w, u32 t, f32 cd) {
            found = E{A[t] + B(w, nv + 1 - t) + Bs[t], cd};
            fw = w;
            ft = t;
            return false;   // the cap admits only cofacets of diameter d
        });
        if (found.idx != kNone<I>) {
            for (u32 i = 0; i < ft; ++i) cv[i] = vd[i];
            cv[ft] = fw;
            for (u32 i = ft; i < nv; ++i) cv[i + 1] = vd[i];
        }
        return found;
    }

    // Zero pivot facet: the first facet in ascending index order whose
    // diameter equals d. Its vertices are written to fv (nv - 1, descending).
    E zero_pivot_facet_v(const u32* vd, u32 nv, f32 d, u32* fv) const {
        if (nv < 2) return E{kNone<I>, 0.0f};
        f32 dm[KMAX][KMAX];
        for (u32 a = 0; a < nv; ++a)
            for (u32 b = a + 1; b < nv; ++b) dm[a][b] = dm[b][a] = cx.dist(vd[a], vd[b]);
        I pre = 0;   // sum_{i<j} C(vd[i], nv-1-i)
        I suf = 0;   // sum_{i>j} C(vd[i], nv-i)
        for (u32 i = 1; i < nv; ++i) suf += B(vd[i], nv - i);
        // facet j (vertex j removed) has diameter d iff some edge of length d
        // avoids j; facets never exceed d, so only the edges of length d matter
        u32 ea[KMAX * KMAX], eb[KMAX * KMAX], ne = 0;
        for (u32 a = 0; a < nv; ++a)
            for (u32 b = a + 1; b < nv; ++b)
                if (dm[a][b] == d) { ea[ne] = a; eb[ne] = b; ++ne; }
        for (u32 j = 0; j < nv; ++j) {
            bool hit = false;
            for (u32 q = 0; q < ne && !hit; ++q) hit = ea[q] != j && eb[q] != j;
            if (hit) {
                const f32 fd = d;
                for (u32 i = 0, o = 0; i < nv; ++i)
                    if (i != j) fv[o++] = vd[i];
                return E{pre + suf, fd};
            }
            pre += B(vd[j], nv - 1 - j);
            if (j + 1 < nv) suf -= B(vd[j + 1], nv - (j + 1));
        }
        return E{kNone<I>, 0.0f};
    }

    E zero_apparent_facet_v(const u32* vd, u32 nv, const E& s) const {
        u32 fv[KMAX], cv[KMAX];
        E f = zero_pivot_facet_v(vd, nv, s.d, fv);
        if (f.idx == kNone<I>) return f;
        return zero_pivot_cofacet_v(fv, nv - 1, f.d, cv).idx == s.idx ? f : E{kNone<I>, 0.0f};
    }
    E zero_apparent_cofacet_v(const u32* vd, u32 nv, const E& s) const {
        u32 fv[KMAX], cv[KMAX];
        E c = zero_pivot_cofacet_v(vd, nv, s.d, cv);
        if (c.idx == kNone<I>) return c;
        return zero_pivot_facet_v(cv, nv + 1, c.d, fv).idx == s.idx ? c : E{kNone<I>, 0.0f};
    }
    E zero_apparent_facet(const E& s, u32 nv) const {
        u32 vd[KMAX];
        decode(s.idx, nv, vd);
        return zero_apparent_facet_v(vd, nv, s);
    }
    E zero_apparent_cofacet(const E& s, u32 nv) const {
        u32 vd[KMAX];
        decode(s.idx, nv, vd);
        return zero_apparent_cofacet_v(vd, nv, s);
    }
    bool in_zero_apparent_pair(const E& s, u32 nv) const {
        u32 vd[KMAX];
        decode(s.idx, nv, vd);
        return zero_apparent_cofacet_v(vd, nv, s).idx != kNone<I> ||
               zero_apparent_facet_v(vd, nv, s).idx != kNone<I>;
    }

    // ---- state carried across dimensions ------------------------------
    std::vector<E> simplices;   // all simplices of the current dimension
    std::vector<E> columns;     // columns to reduce, descending filtration order
    PivotMap<I> pivots;
    // reduction scratch, reused by every dimension (sized once, cleared per
    // dimension, so the allocator does not reshuffle the heap between the
    // compute_pairs and assemble phases)
    std::vector<u64> cp_off{0};
    std::vector<E> cp_red;      // stored reduction columns
    std::vector<E> cp_wr;       // reduction column: append-only, drained by sort
    std::vector<E> cp_buf;      // staging for the initial coboundary heap
    ColumnHeap<I> cp_wc;        // working coboundary

    static void sort_desc(std::vector<E>& v) {
        std::sort(v.begin(), v.end(), [](const E& a, const E& b) { return precedes(b, a); });
    }

    // ---- dimension 0 ---------------------------------------------------
    void dim0() {
        const u32 nt = std::max<u32>(1, num_threads());
        const u32 nb = std::max<u32>(1, std::min<u32>(n ? n : 1, 4 * nt));
        // Visit every kept edge once per pass below: sqrt-balanced row cuts
        // for the dense triangle, vertex-balanced cuts for a CSR graph.
        std::vector<u64> cut(nb + 1, n), vcut(nb + 1, n);
        for (u32 b = 0; b < nb; ++b) {
            cut[b] = u64(double(n) * std::sqrt(double(b) / nb));
            vcut[b] = u64(n) * b / nb;
        }
        auto enum_block = [&](u64 b, auto&& fn) {
            if constexpr (std::is_same_v<Cx, DenseComplex>) {
                for (u64 i = cut[b]; i < cut[b + 1]; ++i) {
                    const f32* row = cx.full ? cx.mat.data() + u64(i) * n
                                             : cx.mat.data() + u64(i) * (i - 1) / 2;
                    for (u32 j = 0; j < i; ++j) {
                        const f32 d = row[j];
                        if (d <= cx.threshold) fn(i, u64(j), d);
                    }
                }
            } else {
                const CsrGraph& g = *cx.g;
                for (u64 v = vcut[b]; v < vcut[b + 1]; ++v)
                    for (u64 t = g.begin(u32(v)); t != g.end(u32(v)); ++t) {
                        const u32 u = g.col_idx[t];
                        if (u < u32(v) && g.weights[t] <= cx.threshold)
                            fn(v, u64(u), g.weights[t]);
                    }
            }
        };

        // Pass A: count kept edges and their diameter range. Dense counts are
        // branchless; the range uses integer compares on the float bit
        // pattern (monotone for non-negative distances) and clamps the top to
        // the threshold, which keeps every kept edge inside [dmin, dmax].
        std::vector<u64> ecnt(nb + 1, 0);
        u64 total = 0;
        f32 dmin = 0.0f, dmax = 0.0f;
        if constexpr (std::is_same_v<Cx, DenseComplex>) {
            std::vector<u32> bmin(nb, ~0u), bmax(nb, 0u);
            parallel_for(nb, [&](u64 b) {
                u64 c = 0;
                u32 mn = ~0u, mx = 0u;
                for (u64 i = cut[b]; i < cut[b + 1]; ++i) {
                    const f32* row = cx.full ? cx.mat.data() + u64(i) * n
                                             : cx.mat.data() + u64(i) * (i - 1) / 2;
                    for (u32 j = 0; j < i; ++j) c += row[j] <= cx.threshold;
                    for (u32 j = 0; j < i; ++j) {
                        const u32 bits = std::bit_cast<u32>(row[j]);
                        mn = std::min(mn, bits);
                        mx = std::max(mx, bits);
                    }
                }
                ecnt[b + 1] = c;
                bmin[b] = mn;
                bmax[b] = mx;
            });
            for (u32 b = 0; b < nb; ++b) ecnt[b + 1] += ecnt[b];
            total = ecnt[nb];
            u32 mnbits = ~0u, mxbits = 0u;
            for (u32 b = 0; b < nb; ++b) {
                if (ecnt[b + 1] == ecnt[b]) continue;
                mnbits = std::min(mnbits, bmin[b]);
                mxbits = std::max(mxbits, bmax[b]);
            }
            if (total) {
                dmin = std::bit_cast<f32>(mnbits);
                dmax = std::min(std::bit_cast<f32>(mxbits), cx.threshold);
            }
        } else {
            const CsrGraph& g = *cx.g;
            for (u32 v = 0; v < n; ++v)
                for (u64 t = g.begin(v); t != g.end(v); ++t) {
                    const u32 u = g.col_idx[t];
                    const f32 w = g.weights[t];
                    if (u < v && w <= cx.threshold) {
                        ++total;
                        dmin = std::min(dmin, w);
                        dmax = std::max(dmax, w);
                    }
                }
            dmax = std::min(dmax, cx.threshold);
        }
        st.edges = total;

        std::vector<E> edges;
        auto sort_plain = [&] {
            std::sort(edges.begin(), edges.end(),
                      [](const E& a, const E& b) { return precedes(a, b); });
        };
        if (total < 131072 || !(dmax > dmin)) {
            // Small (or all-equal-diameter) input: collect, then sort.
            if constexpr (std::is_same_v<Cx, DenseComplex>) {
                edges.resize(total);
                parallel_for(nb, [&](u64 b) {
                    u64 o = ecnt[b];
                    enum_block(b, [&](u64 i, u64 j, f32 d) {
                        edges[o++] = E{I(i * (i - 1) / 2 + j), d};
                    });
                });
            } else {
                edges.reserve(total);
                const CsrGraph& g = *cx.g;
                for (u32 v = 0; v < n; ++v)
                    for (u64 t = g.begin(v); t != g.end(v); ++t) {
                        const u32 u = g.col_idx[t];
                        if (u < v && g.weights[t] <= cx.threshold)
                            edges.push_back(E{I(u64(v) * (v - 1) / 2 + u), g.weights[t]});
                    }
            }
            sort_plain();
        } else {
            // Large input: histogram by diameter, scatter each edge directly
            // into its final slot (bucket-major), then sort inside each
            // bucket. Buckets ascend in d and the per-bucket sort applies the
            // full (diameter, index) order, so the result equals the plain
            // std::sort order while comparing ~n log(n/nbk) hot-cache items.
            const u32 nbk = u32(std::min<u64>(2048, std::max<u64>(256, total / 1024)));
            const f32 inv = 1.0f / (dmax - dmin);
            auto bucket_of = [&](f32 d) {
                const int b = int((d - dmin) * inv * f32(nbk));
                return b < 0 ? 0u : (u32(b) >= nbk ? nbk - 1 : u32(b));
            };
            std::vector<std::vector<u64>> cnt(nb, std::vector<u64>(nbk, 0));
            parallel_for(nb, [&](u64 b) {
                u64* h = cnt[b].data();
                enum_block(b, [&](u64, u64, f32 d) { ++h[bucket_of(d)]; });
            });
            std::vector<u64> start(nbk + 1, 0);
            for (u32 k = 0; k < nbk; ++k) {
                u64 o = start[k];
                for (u32 b = 0; b < nb; ++b) {
                    const u64 t = cnt[b][k];
                    cnt[b][k] = o;
                    o += t;
                }
                start[k + 1] = o;
            }
            edges.resize(total);
            parallel_for(nb, [&](u64 b) {
                u64* h = cnt[b].data();
                enum_block(b, [&](u64 i, u64 j, f32 d) {
                    edges[h[bucket_of(d)]++] = E{I(i * (i - 1) / 2 + j), d};
                });
            });
            std::atomic<u32> next{0};
            parallel_for(nt, [&](u64) {
                for (;;) {
                    const u32 k = next.fetch_add(1, std::memory_order_relaxed);
                    if (k >= nbk) break;
                    std::sort(edges.begin() + start[k], edges.begin() + start[k + 1],
                              [](const E& a, const E& b) { return precedes(a, b); });
                }
            });
        }

        // Edges in a zero-apparent pair never become columns. The test is
        // independent per edge, so it runs in parallel before the serial
        // union-find.
        std::vector<u8> apparent(maxdim >= 1 ? edges.size() : 0, 0);
        if (maxdim >= 1)
            parallel_ranges(edges.size(), num_threads(), [&](u32, u64 lo, u64 hi) {
                for (u64 i = lo; i < hi; ++i)
                    apparent[i] = zero_apparent_cofacet(edges[i], 2).idx != kNone<I>;
            });
        UnionFind uf(n);
        columns.clear();
        u64 skipped = 0;
        for (u64 i = 0; i < edges.size(); ++i) {
            const E e = edges[i];
            u32 vd[2];
            decode(e.idx, 2, vd);
            if (uf.unite(vd[0], vd[1])) {
                if (e.d > 0.0f) out.add(0, 0.0f, e.d);
            } else if (maxdim >= 1) {
                if (!apparent[i]) columns.push_back(e);
                else ++skipped;
            }
        }
        std::vector<u8>().swap(apparent);
        for (u32 v = 0; v < n; ++v)
            if (uf.find(v) == v) out.add(0, 0.0f, INFINITY);
        std::reverse(columns.begin(), columns.end());
        if (maxdim >= 2) simplices.swap(edges);
        st.apparent.push_back(skipped);
    }

    // ---- reduction of one dimension ------------------------------------
    // columns: simplices of dimension dim (nv = dim + 1 vertices).
    void compute_pairs(u32 dim) {
        const u32 nv = dim + 1;
        std::vector<u64>& off = cp_off;
        std::vector<E>& red = cp_red;      // stored reduction columns
        std::vector<E>& wr = cp_wr;        // reduction column: append-only, drained by sort
        std::vector<E>& buf = cp_buf;      // staging for the initial coboundary heap
        ColumnHeap<I>& wc = cp_wc;         // working coboundary
        pivots.reset(columns.size() + 16);
        off.clear();
        off.push_back(0);
        off.reserve(columns.size() + 1);
        red.clear();
        u64 emergent_skips = 0;
        u32 vd[KMAX];

        auto push_coboundary = [&](const E& s) {
            decode(s.idx, nv, vd);
            cofacet_entries(vd, nv, s.d, true, cx.threshold, [&](const E& c) {
                wc.push(c);
                return true;
            });
        };

        for (u64 c = 0; c < columns.size(); ++c) {
            const E col = columns[c];
            const f32 birth = col.d;
            wc.clear();
            wr.clear();

            // initial coboundary, with the emergent-pair shortcut
            E pivot{kNone<I>, 0.0f};
            bool emergent = false;
            {
                bool check = true;
                buf.clear();
                decode(col.idx, nv, vd);
                cofacet_entries(vd, nv, col.d, true, cx.threshold, [&](const E& cf) {
                    if (check && cf.d == col.d) {
                        if (!pivots.contains(cf.idx) &&
                            zero_apparent_facet(cf, nv + 1).idx == kNone<I>) {
                            pivot = cf;
                            emergent = true;
                            ++emergent_skips;
                            return false;
                        }
                        check = false;
                    }
                    buf.push_back(cf);
                    return true;
                });
                if (!emergent) {
                    wc.heapify(buf);
                    pivot = wc.get_pivot();
                }
            }

            bool fresh = emergent;   // pivot already known to be unpaired
            for (;;) {
                if (pivot.idx == kNone<I>) {
                    out.add(dim, birth, INFINITY);
                    break;
                }
                if (fresh) {
                    pivots.insert(pivot.idx, c);
                    break;
                }
                const u64 j = pivots.find(pivot.idx);
                if (j != ~0ull) {
                    const E add = columns[j];
                    wr.push_back(add);
                    push_coboundary(add);
                    for (u64 t = off[j]; t < off[j + 1]; ++t) {
                        wr.push_back(red[t]);
                        push_coboundary(red[t]);
                    }
                    pivot = wc.get_pivot();
                    continue;
                }
                const E fa = zero_apparent_facet(pivot, nv + 1);
                if (fa.idx != kNone<I>) {
                    wr.push_back(fa);
                    push_coboundary(fa);
                    pivot = wc.get_pivot();
                    continue;
                }
                if (pivot.d > birth) out.add(dim, birth, pivot.d);
                pivots.insert(pivot.idx, c);
                // Drain the reduction column. Sorting gives the same survivor
                // sequence as repeated pop_pivot (sorted order IS pop order;
                // equal indices are adjacent, so pairing runs linearly).
                std::sort(wr.begin(), wr.end(),
                          [](const E& a, const E& b) { return precedes(a, b); });
                for (size_t x = 0; x < wr.size();) {
                    if (x + 1 < wr.size() && wr[x].idx == wr[x + 1].idx)
                        x += 2;   // equal pair cancels
                    else
                        red.push_back(wr[x++]);
                }
                break;
            }
            off.push_back(red.size());
        }
        st.columns.push_back(columns.size());
        st.emergent.push_back(emergent_skips);
    }

    // ---- columns of dimension dim + 1 from the simplices of dimension dim
    void assemble(u32 dim) {
        const u32 nv = dim + 1;         // vertices of the current simplices
        const bool keep_next = dim + 1 < maxdim;
        const u32 nt = num_threads();
        std::vector<std::vector<E>> cols(nt), nexts(nt);
        std::vector<u64> skipped(nt, 0);
        parallel_ranges(simplices.size(), nt, [&](u32 tid, u64 lo, u64 hi) {
            u32 vd[KMAX];
            auto& oc = cols[tid];
            auto& on = nexts[tid];
            for (u64 s = lo; s < hi; ++s) {
                decode(simplices[s].idx, nv, vd);
                I A[KMAX + 1], Bs[KMAX + 1];
                A[0] = 0;
                for (u32 i = 0; i < nv; ++i) A[i + 1] = A[i] + B(vd[i], nv + 1 - i);
                Bs[nv] = 0;
                for (u32 i = nv; i-- > 0;) Bs[i] = Bs[i + 1] + B(vd[i], nv - i);
                cx.cofacets(vd, nv, simplices[s].d, false, cx.threshold, [&](u32 w, u32 t, f32 cd) {
                    const E cf{A[t] + B(w, nv + 1 - t) + Bs[t], cd};
                    if (keep_next) on.push_back(cf);
                    if (pivots.contains(cf.idx)) return true;
                    u32 cv[KMAX];                       // cofacet vertices, descending
                    for (u32 i = 0; i < t; ++i) cv[i] = vd[i];
                    cv[t] = w;
                    for (u32 i = t; i < nv; ++i) cv[i + 1] = vd[i];
                    if (zero_apparent_cofacet_v(cv, nv + 1, cf).idx != kNone<I> ||
                        zero_apparent_facet_v(cv, nv + 1, cf).idx != kNone<I>) { ++skipped[tid]; return true; }
                    oc.push_back(cf);
                    return true;
                });
            }
        });
        columns.clear();
        for (auto& v : cols) columns.insert(columns.end(), v.begin(), v.end());
        std::vector<E> next;
        if (keep_next)
            for (auto& v : nexts) next.insert(next.end(), v.begin(), v.end());
        simplices.swap(next);
        sort_desc(columns);
        u64 sk = 0;
        for (u64 x : skipped) sk += x;
        st.apparent.push_back(sk);
    }

    void run() {
        dim0();
        for (u32 dim = 1; dim <= maxdim; ++dim) {
            compute_pairs(dim);
            if (dim < maxdim) assemble(dim);
        }
    }
};

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------
// force_wide selects 128-bit indices even when 64 bits suffice (testing).
template <typename Cx>
inline Barcode compute(const Cx& cx, u32 maxdim, RipsStats* stats = nullptr,
                       bool force_wide = false) {
    if (maxdim + 2 > 32)
        throw std::invalid_argument("rips: maxdim must be <= 30");
    Barcode out;
    out.ensure(maxdim);
    RipsStats st;
    st.threshold = cx.threshold;
    if (cx.n == 0) {
        if (stats) *stats = st;
        return out;
    }
    if (!force_wide && index_fits<u64>(cx.n, maxdim)) {
        Reducer<u64, Cx> r(cx, maxdim, out, st);
        r.run();
    } else if (index_fits<u128>(cx.n, maxdim)) {
        st.wide_index = true;
        Reducer<u128, Cx> r(cx, maxdim, out, st);
        r.run();
    } else {
        throw std::overflow_error(
            "rips: C(n, maxdim + 2) exceeds 128-bit simplex indices; lower maxdim");
    }
    if (stats) *stats = st;
    return out;
}

// Dense full or thresholded Rips from a distance functor. An infinite
// threshold is replaced by the enclosing radius.
template <typename Dist>
inline Barcode compute_dense(u32 n, Dist&& d, f32 threshold, u32 maxdim,
                             RipsStats* stats = nullptr, bool force_wide = false,
                             u64 budget_bytes = ~0ull) {
    DenseComplex cx = make_dense(n, d, budget_bytes);
    cx.threshold = std::isinf(threshold) ? cx.enclosing_radius() : threshold;
    return compute(cx, maxdim, stats, force_wide);
}

// Flag complex of a weighted graph (rows sorted by neighbor id).
inline Barcode compute_graph(const CsrGraph& g, u32 maxdim,
                             f32 threshold = INFINITY, RipsStats* stats = nullptr,
                             bool force_wide = false) {
    SparseComplex cx(g, threshold);
    return compute(cx, maxdim, stats, force_wide);
}

}  // namespace tops::rips

#endif  // TURBOTOPS_RIPS_HPP
