// SPDX-License-Identifier: MIT
// redzed.hpp — Vietoris–Rips persistent homology by reduction to zero
// differentials with active enumeration (RedZeD).
//
// Algorithm: C. Kapulkin and N. Kershaw, "RedZeD", arXiv 2606.06310; this
// file is a port of the MIT-licensed reference implementation
// (github.com/nkershaw01/RedZeD @ 22ef1bf; see THIRD_PARTY_NOTICES.md).
// The algorithm is unchanged. Two geometry policies supply the only parts
// that depend on how distances are stored:
//
//   DenseGeom   — a rips::DenseComplex (full Rips cut at its threshold). The
//                 growing edge graph and the radius graph are n x n bitsets,
//                 as in the reference code.
//   SparseGeom  — a CsrGraph (flag complex of a weighted graph). Each CSR
//                 entry stores its edge's filtration rank, so "edge (a, b)
//                 is present at step k" is a rank comparison; candidate sets
//                 are ascending vertex lists instead of bitsets.
//
// Enumeration order (ascending vertex ids, ties broken by combinatorial
// index) is the same for both, so the two produce identical diagrams on the
// same complex, and both match rips::compute bar for bar (the diagram of a
// filtration does not depend on the tie-breaking order).
//
// SIMD (via SIMDe, as in rips.hpp): the dense edge pass compares and buckets
// 8 distances at a time, and every bitset intersection (edge-driven
// generation, the leftover-coface scan) runs 256 bits at a time.
//
// Single-threaded. Simplex indices are 64-bit; complexes whose indices do not
// fit throw std::overflow_error (rips::compute handles those with 128 bits).
#ifndef TURBOTOPS_REDZED_HPP
#define TURBOTOPS_REDZED_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core.hpp"
#include "rips.hpp"
#ifndef TURBOTOPS_NO_SIMD
#include "simde/x86/avx2.h"
#endif

namespace tops::redzed {

namespace detail {

using value_t = f32;
using comb_t = int64_t;    // combinatorial index of a simplex (ascending vertices)
using rank_t = int64_t;    // position of a simplex in its dimension's filtration order
using slot_t = int64_t;

inline constexpr value_t VINF = std::numeric_limits<value_t>::infinity();

inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }

// out[t] = a[t] & b[t] for t < len; returns whether any bit is set.
inline bool and_words(uint64_t* out, const uint64_t* a, const uint64_t* b, int len) {
    int t = 0;
    uint64_t any = 0;
#ifndef TURBOTOPS_NO_SIMD
    simde__m256i acc = simde_mm256_setzero_si256();
    for (; t + 4 <= len; t += 4) {
        const simde__m256i r = simde_mm256_and_si256(simde_mm256_loadu_si256(a + t),
                                                     simde_mm256_loadu_si256(b + t));
        simde_mm256_storeu_si256(out + t, r);
        acc = simde_mm256_or_si256(acc, r);
    }
    any = !simde_mm256_testz_si256(acc, acc);
#endif
    for (; t < len; ++t) {
        out[t] = a[t] & b[t];
        any |= out[t];
    }
    return any != 0;
}

// ---------------------------------------------------------------------------
// FastMap: rank_t key (>= 0) -> slot_t value. Open addressing with linear
// probing over one array of interleaved (key, value) pairs, so a probe
// touches one cache line; Fibonacci hashing; maximum load 1/2 (expected
// probes ~1.5 hit / ~2.5 miss, vs ~32 for a miss at load 7/8); backward-
// shift deletion, so there are no tombstones. Key -1 marks a hole.
// ---------------------------------------------------------------------------
struct FastMap {
    struct Cell {
        rank_t k;
        slot_t v;
    };
    std::vector<Cell> cells;
    size_t mask = 15, count = 0, limit = 8;
    int shift = 60;
    explicit FastMap(size_t cap = 16) { init(cap); }
    void init(size_t cap) {
        size_t c = 16;
        while (c < 2 * cap) c <<= 1;
        cells.assign(c, Cell{-1, 0});
        mask = c - 1;
        count = 0;
        limit = c >> 1;
        shift = shift_for(c);
    }
    static int shift_for(size_t c) {
        int b = 0;
        while ((size_t(1) << b) < c) ++b;
        return 64 - b;
    }
    size_t hsh(rank_t k) const { return size_t((uint64_t(k) * 0x9E3779B97F4A7C15ull) >> shift); }
    slot_t get(rank_t k) const {
        for (size_t i = hsh(k);; i = (i + 1) & mask) {
            const Cell& c = cells[i];
            if (c.k == k) return c.v;
            if (c.k == -1) return 0;
        }
    }
    void grow() {
        std::vector<Cell> old;
        old.swap(cells);
        const size_t c = old.size() << 1;
        cells.assign(c, Cell{-1, 0});
        mask = c - 1;
        limit = c >> 1;
        shift = shift_for(c);
        for (const Cell& o : old) {
            if (o.k == -1) continue;
            size_t i = hsh(o.k);
            while (cells[i].k != -1) i = (i + 1) & mask;
            cells[i] = o;
        }
    }
    void set(rank_t k, slot_t v) {
        for (size_t i = hsh(k);; i = (i + 1) & mask) {
            Cell& c = cells[i];
            if (c.k == k) { c.v = v; return; }
            if (c.k == -1) {
                c = Cell{k, v};
                if (++count > limit) grow();
                return;
            }
        }
    }
    // k's value if present (else 0), and the hole where k belongs
    slot_t find_or_hole(rank_t k, size_t& hole) const {
        for (size_t i = hsh(k);; i = (i + 1) & mask) {
            const Cell& c = cells[i];
            if (c.k == k) return c.v;
            if (c.k == -1) { hole = i; return 0; }
        }
    }
    void insert_at(size_t hole, rank_t k, slot_t v) {
        cells[hole] = Cell{k, v};
        if (++count > limit) grow();
    }
    void erase(rank_t k) {
        size_t i = hsh(k);
        for (;; i = (i + 1) & mask) {
            if (cells[i].k == -1) return;
            if (cells[i].k == k) break;
        }
        for (size_t j = i;;) {
            j = (j + 1) & mask;
            const rank_t kj = cells[j].k;
            if (kj == -1) break;
            if (((j - hsh(kj)) & mask) >= ((j - i) & mask)) {
                cells[i] = cells[j];
                i = j;
            }
        }
        cells[i].k = -1;
        --count;
    }
    template <class F>
    void for_each_key(F&& f) const {
        for (const Cell& c : cells)
            if (c.k != -1) f(c.k);
    }
};

// ---------------------------------------------------------------------------
// Sorted-vector sets (ascending).
// ---------------------------------------------------------------------------
using SSet = std::vector<rank_t>;

inline void sset_remove(SSet& v, rank_t x) {
    auto it = std::lower_bound(v.begin(), v.end(), x);
    if (it != v.end() && *it == x) v.erase(it);
}
inline void sset_toggle(SSet& v, rank_t x) {
    auto it = std::lower_bound(v.begin(), v.end(), x);
    if (it != v.end() && *it == x) v.erase(it);
    else v.insert(it, x);
}
// out = a XOR b
inline void sset_xor2(SSet& out, const SSet& a, const SSet& b) {
    const size_t na = a.size(), nb = b.size();
    if (out.size() < na + nb) out.resize(na + nb);
    size_t i = 0, j = 0, o = 0;
    while (i < na && j < nb) {
        const rank_t x = a[i], y = b[j];
        if (x < y) out[o++] = x, ++i;
        else if (x > y) out[o++] = y, ++j;
        else ++i, ++j;
    }
    while (i < na) out[o++] = a[i++];
    while (j < nb) out[o++] = b[j++];
    out.resize(o);
}
// dst ^= src
inline void sset_xor(SSet& dst, const SSet& src, SSet& buf) {
    const size_t ni = dst.size(), nj = src.size();
    if (nj == 0) return;
    if (ni == 0) { dst = src; return; }
    if (buf.size() < ni + nj) buf.resize(ni + nj);
    size_t i = 0, j = 0, o = 0;
    while (i < ni && j < nj) {
        const rank_t a = dst[i], b = src[j];
        if (a < b) buf[o++] = a, ++i;
        else if (a > b) buf[o++] = b, ++j;
        else ++i, ++j;
    }
    while (i < ni) buf[o++] = dst[i++];
    while (j < nj) buf[o++] = src[j++];
    dst.assign(buf.begin(), buf.begin() + o);
}

// ---------------------------------------------------------------------------
// SetDict: rank -> sorted set, pooled. sing[slot] caches the element of a
// singleton set (0 = not a singleton). Optionally stores each key's diameter
// and vertex list.
// ---------------------------------------------------------------------------
struct SetDict {
    FastMap map{64};
    std::vector<SSet> sets;
    std::vector<slot_t> freelist;
    std::vector<rank_t> sing;
    int side_vpc = 0;
    std::vector<value_t> sdiam;
    std::vector<int32_t> svert;

    void set_side(slot_t s, value_t d, const int32_t* v) {
        if (!side_vpc) return;
        sdiam[s - 1] = d;
        std::memcpy(&svert[size_t(s - 1) * side_vpc], v, sizeof(int32_t) * side_vpc);
    }
    value_t side_diam(rank_t k) const { return sdiam[map.get(k) - 1]; }
    const int32_t* side_vert(rank_t k) const { return &svert[size_t(map.get(k) - 1) * side_vpc]; }

    slot_t slot(rank_t k) const { return map.get(k); }
    SSet& at(slot_t s) { return sets[s - 1]; }
    const SSet& at(slot_t s) const { return sets[s - 1]; }
    void touch(slot_t s) {
        const SSet& v = sets[s - 1];
        sing[s - 1] = v.size() == 1 ? v[0] : 0;
    }
    slot_t create(rank_t k) {
        size_t hole = 0;
        slot_t s = map.find_or_hole(k, hole);
        if (s) {
            sets[s - 1].clear();
            sing[s - 1] = 0;
            return s;
        }
        if (freelist.empty()) {
            sets.emplace_back();
            sing.push_back(0);
            if (side_vpc) {
                sdiam.push_back(0);
                svert.resize(sets.size() * size_t(side_vpc));
            }
            s = slot_t(sets.size());
        } else {
            s = freelist.back();
            freelist.pop_back();
            sets[s - 1].clear();
        }
        sing[s - 1] = 0;
        map.insert_at(hole, k, s);
        return s;
    }
    slot_t singleton(rank_t k, rank_t v) {
        const slot_t s = create(k);
        sets[s - 1].push_back(v);
        sing[s - 1] = v;
        return s;
    }
    void erase(rank_t k) {
        const slot_t s = map.get(k);
        if (!s) return;
        map.erase(k);
        sets[s - 1].clear();
        sing[s - 1] = 0;
        freelist.push_back(s);
    }
    void remove_elem(slot_t s, rank_t y) {
        sset_remove(sets[s - 1], y);
        touch(s);
    }
    void xor_into(slot_t s, const SSet& src, SSet& scratch) {
        SSet& dst = sets[s - 1];
        if (src.empty()) return;
        if (dst.empty()) { dst = src; touch(s); return; }
        sset_xor2(scratch, dst, src);
        dst.swap(scratch);
        touch(s);
    }
    void keys(std::vector<rank_t>& out) const {
        out.clear();
        map.for_each_key([&](rank_t k) { out.push_back(k); });
    }
};

// True when XOR-ing x's set with the seeds would cancel to zero through
// singletons alone (the common case), so the full XOR can be skipped.
inline bool seeded_is_noop(const SetDict& sd, slot_t xslot, const std::vector<slot_t>& seeds) {
    int cnt = 0;
    rank_t f = 0;
    if (xslot) {
        f = sd.sing[xslot - 1];
        if (!f) return false;
        cnt = 1;
    }
    for (slot_t s : seeds) {
        const rank_t v = sd.sing[s - 1];
        if (!v) return false;
        if (cnt == 0) f = v;
        else if (v != f) return false;
        ++cnt;
    }
    return (cnt & 1) == 0;
}

// Union-find for dimension 0 whose root is always the oldest vertex.
struct RootForest {
    std::vector<int32_t> parent;
    explicit RootForest(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int32_t find(int32_t x) {
        int32_t r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) {
            const int32_t nx = parent[x];
            parent[x] = r;
            x = nx;
        }
        return r;
    }
};

// Combinatorial index -> pool slot (0 = absent): a direct array while the
// index universe is at most 2^24, a hash map beyond.
inline constexpr int64_t DENSE_INDEX_CAP = int64_t(1) << 24;

struct FaceSlot {
    std::vector<slot_t> dense;
    FastMap sparse;
    bool isdense = true;
    void init(int64_t universe) {
        std::vector<slot_t>().swap(dense);
        sparse.init(16);
        isdense = universe >= 0 && universe <= DENSE_INDEX_CAP;
        if (isdense) dense.assign(size_t(universe), 0);
    }
    slot_t get(comb_t k) const { return isdense ? dense[size_t(k)] : sparse.get(k); }
    void prefetch(comb_t k) const {
        if (isdense) __builtin_prefetch(&dense[size_t(k)]);
        else __builtin_prefetch(&sparse.cells[sparse.hsh(k)]);
    }
    void set(comb_t k, slot_t v) {
        if (isdense) dense[size_t(k)] = v;
        else if (v == 0) sparse.erase(k);
        else sparse.set(k, v);
    }
};

// Binomial coefficients C(n, k), k <= kmax, overflow-checked.
struct BinomialTable {
    std::vector<comb_t> data;
    int kmax = 0;
    BinomialTable() = default;
    BinomialTable(int nmax, int kmaxv) : kmax(kmaxv) {
        data.assign(size_t(kmax + 1) * size_t(nmax + 1), 0);
        for (int n = 0; n <= nmax; ++n) {
            data[size_t(n) * (kmax + 1)] = 1;
            for (int k = 1; k <= std::min(n, kmax); ++k) {
                comb_t s;
                if (__builtin_add_overflow(data[size_t(n - 1) * (kmax + 1) + (k - 1)],
                                           data[size_t(n - 1) * (kmax + 1) + k], &s))
                    throw std::overflow_error("redzed: simplex index exceeds 64 bits");
                data[size_t(n) * (kmax + 1) + k] = s;
            }
        }
    }
    comb_t operator()(int n, int k) const {
        if (k > n || k < 0 || n < 0) return 0;
        return data[size_t(n) * (kmax + 1) + k];
    }
    // no checks: requires n >= 0 and 0 <= k <= kmax (entries with k > n hold 0)
    comb_t at(int n, int k) const { return data[size_t(n) * (kmax + 1) + k]; }
};

inline comb_t comb_index(const int32_t* v, int m, const BinomialTable& b) {
    comb_t idx = 0;
    for (int k = 1; k <= m; ++k) idx += b.at(v[k - 1], k);
    return idx;
}
// indices of all m faces of v (ascending, m vertices): out[p] omits v[p].
// Prefix/suffix sums, O(m) instead of O(m^2).
inline void face_indices(const int32_t* v, int m, const BinomialTable& b, comb_t* out) {
    comb_t suf = 0;   // sum_{k > p} C(v[k], k)      (0-based k: shifted ranks)
    for (int k = m - 1; k >= 0; --k) {
        out[k] = suf;
        suf += b.at(v[k], k);
    }
    comb_t pre = 0;   // sum_{k < p} C(v[k], k + 1)
    for (int p = 0; p < m; ++p) {
        out[p] += pre;
        pre += b.at(v[p], p + 1);
    }
}
// index of the face that omits position omit (1-based)
inline comb_t face_index_without(const int32_t* v, int m, int omit, const BinomialTable& b) {
    comb_t idx = 0;
    for (int k = 1; k <= omit - 1; ++k) idx += b(v[k - 1], k);
    for (int k = omit + 1; k <= m; ++k) idx += b(v[k - 1], k - 1);
    return idx;
}

// Edges in filtration order; edge r (1-based) is verts[2r-2], verts[2r-1].
struct EdgeList {
    size_t count = 0;
    std::vector<value_t> diam;
    std::vector<int32_t> verts;
    const int32_t* vp(size_t r) const { return &verts[2 * (r - 1)]; }
    value_t dm(size_t r) const { return diam[r - 1]; }
};

// ---------------------------------------------------------------------------
// Bucket sort into filtration order: (diameter, combinatorial index).
// ---------------------------------------------------------------------------
struct SortScratch {
    std::vector<rank_t> cnt;
    std::vector<value_t> dbuf, sdbuf;
    std::vector<int32_t> vbuf, svbuf;
    std::vector<std::pair<comb_t, rank_t>> kv;
    std::vector<value_t> kvd;
};

inline int bucket_of(value_t d, value_t lo, value_t inv, int nb) {
    const int b = int((d - lo) * inv * nb);
    return b < 0 ? 0 : (b >= nb ? nb - 1 : b);
}

// positions a..b (1-based, inclusive); comparison sort, or insertion sort
inline void sort_slice_small(std::vector<value_t>& diam, std::vector<int32_t>& verts, int vpc,
                             size_t a, size_t b, const BinomialTable& bt, SortScratch& sc,
                             bool comparison) {
    const size_t n = b - a + 1;
    if (n <= 1) return;
    if (sc.kv.size() < n) { sc.kv.resize(n); sc.kvd.resize(n); }
    if (sc.sdbuf.size() < n) sc.sdbuf.resize(n);
    if (sc.svbuf.size() < n * size_t(vpc)) sc.svbuf.resize(n * size_t(vpc));
    for (size_t t = 0; t < n; ++t) {
        const size_t p = a + t - 1;
        sc.kvd[t] = diam[p];
        sc.kv[t] = {comb_index(&verts[p * vpc], vpc, bt), rank_t(p)};
    }
    std::vector<rank_t>& ord = sc.cnt;
    if (ord.size() < n) ord.resize(n);
    for (size_t t = 0; t < n; ++t) ord[t] = rank_t(t);
    auto less = [&](rank_t x, rank_t y) {
        if (sc.kvd[x] != sc.kvd[y]) return sc.kvd[x] < sc.kvd[y];
        return sc.kv[x].first < sc.kv[y].first;
    };
    if (comparison) {
        std::sort(ord.begin(), ord.begin() + n, less);
    } else {
        for (size_t i = 1; i < n; ++i) {
            const rank_t x = ord[i];
            size_t j = i;
            while (j > 0 && less(x, ord[j - 1])) { ord[j] = ord[j - 1]; --j; }
            ord[j] = x;
        }
    }
    for (size_t t = 0; t < n; ++t) {
        const size_t p = size_t(sc.kv[ord[t]].second);
        sc.sdbuf[t] = diam[p];
        std::memcpy(&sc.svbuf[t * vpc], &verts[p * vpc], sizeof(int32_t) * vpc);
    }
    for (size_t t = 0; t < n; ++t) {
        diam[a + t - 1] = sc.sdbuf[t];
        std::memcpy(&verts[(a + t - 1) * vpc], &sc.svbuf[t * vpc], sizeof(int32_t) * vpc);
    }
}

inline void sort_slice(std::vector<value_t>& diam, std::vector<int32_t>& verts, int vpc,
                       size_t a, size_t b, const BinomialTable& bt, SortScratch& sc, int depth) {
    const size_t n = b - a + 1;
    if (n <= 1) return;
    if (n <= 32 || depth >= 5) { sort_slice_small(diam, verts, vpc, a, b, bt, sc, n > 32); return; }
    value_t dmin = VINF, dmax = -VINF;
    for (size_t i = a; i <= b; ++i) {
        dmin = std::min(dmin, diam[i - 1]);
        dmax = std::max(dmax, diam[i - 1]);
    }
    if (dmax == dmin) { sort_slice_small(diam, verts, vpc, a, b, bt, sc, true); return; }
    const int nb = std::max(2, int(n >> 2));
    const value_t inv = value_t(1.0 / (double(dmax) - double(dmin)));
    std::vector<rank_t> cnt(nb + 1, 0);
    for (size_t i = a; i <= b; ++i) cnt[bucket_of(diam[i - 1], dmin, inv, nb) + 1] += 1;
    for (int t = 1; t <= nb; ++t) cnt[t] += cnt[t - 1];
    const std::vector<rank_t> bounds(cnt);
    if (sc.dbuf.size() < n) sc.dbuf.resize(n);
    if (sc.vbuf.size() < n * size_t(vpc)) sc.vbuf.resize(n * size_t(vpc));
    for (size_t i = a; i <= b; ++i) {
        const value_t d = diam[i - 1];
        const rank_t p = cnt[bucket_of(d, dmin, inv, nb)]++;
        sc.dbuf[p] = d;
        std::memcpy(&sc.vbuf[size_t(p) * vpc], &verts[(i - 1) * vpc], sizeof(int32_t) * vpc);
    }
    for (size_t t = 0; t < n; ++t) {
        diam[a + t - 1] = sc.dbuf[t];
        std::memcpy(&verts[(a + t - 1) * vpc], &sc.vbuf[t * vpc], sizeof(int32_t) * vpc);
    }
    for (int bk = 0; bk < nb; ++bk) {
        const size_t sa = a + size_t(bounds[bk]);
        const size_t sb = a + size_t(bounds[bk + 1]) - 1;
        if (sb > sa) sort_slice(diam, verts, vpc, sa, sb, bt, sc, depth + 1);
    }
}

// Edges with diameter <= threshold, in filtration order, in parallel with a
// result identical to the serial pass: rows are cut into contiguous blocks
// (balanced by row size); each block histograms its rows into the diameter
// buckets; a block's slots in bucket k follow the earlier blocks' slots, so
// the scatter reproduces the serial (row-major) order exactly; each bucket is
// then sorted independently (deterministic), buckets handed out dynamically.
template <typename G>
inline void build_edges(const G& geo, value_t threshold, const BinomialTable& bt, EdgeList& out) {
    const int n = geo.n;
    const int nb = int(std::clamp<uint64_t>(geo.pair_capacity() >> 2, 1024, 1 << 16));
    const value_t inv = (std::isfinite(threshold) && threshold > 0) ? value_t(1.0 / threshold) : 0;
    out.count = 0;
    if (n < 2) return;
    const u32 B = std::min<u32>(num_threads(), u32(n - 1));
    std::vector<int> cut(B + 1, n);
    {
        double total = 0;
        for (int j = 1; j < n; ++j) total += double(geo.row_weight(j));
        double acc = 0;
        u32 b = 1;
        cut[0] = 1;
        for (int j = 1; j < n && b < B; ++j) {
            acc += double(geo.row_weight(j));
            while (b < B && acc >= total * b / B) cut[b++] = j + 1;
        }
        while (b < B) cut[b++] = n;
    }
    std::vector<std::vector<rank_t>> cnt(B, std::vector<rank_t>(nb, 0));
    parallel_ranges(B, B, [&](u32, u64 lo, u64 hi) {
        for (u64 b = lo; b < hi; ++b) {
            rank_t* c = cnt[b].data();
            for (int j = cut[b]; j < cut[b + 1]; ++j)
                geo.bucket_row(j, threshold, inv, nb, [&](int, value_t, int k) { ++c[k]; });
        }
    });
    // starts[k] = first slot of bucket k; cnt[b][k] becomes block b's first slot in it
    std::vector<rank_t> starts(nb + 1, 0);
    for (int k = 0; k < nb; ++k) {
        rank_t o = starts[k];
        for (u32 b = 0; b < B; ++b) {
            const rank_t c = cnt[b][k];
            cnt[b][k] = o;
            o += c;
        }
        starts[k + 1] = o;
    }
    const size_t count = size_t(starts[nb]);
    out.count = count;
    if (count == 0) return;
    out.diam.resize(count);
    out.verts.resize(2 * count);
    parallel_ranges(B, B, [&](u32, u64 lo, u64 hi) {
        for (u64 b = lo; b < hi; ++b) {
            rank_t* c = cnt[b].data();
            for (int j = cut[b]; j < cut[b + 1]; ++j)
                geo.bucket_row(j, threshold, inv, nb, [&](int i, value_t d, int k) {
                    const rank_t p = c[k]++;
                    out.diam[p] = d;
                    out.verts[2 * size_t(p)] = i;
                    out.verts[2 * size_t(p) + 1] = j;
                });
        }
    });
    std::vector<std::vector<rank_t>>().swap(cnt);
    std::atomic<int> next{0};
    const u32 nt = std::min<u32>(num_threads(), u32(nb));
    parallel_ranges(nt, nt, [&](u32, u64, u64) {
        SortScratch sc;
        for (;;) {
            const int k = next.fetch_add(1, std::memory_order_relaxed);
            if (k >= nb) break;
            const size_t a = size_t(starts[k]) + 1, e = size_t(starts[k + 1]);
            if (e > a) sort_slice(out.diam, out.verts, 2, a, e, bt, sc, 1);
        }
    });
}

// ---------------------------------------------------------------------------
// Edge-driven generation of the simplices of dimension d >= 2: each simplex
// is generated once, from its longest edge e, as e plus a clique of the
// common neighbourhood of e's endpoints in G_<e (the edges before e).
// Simplices from a tie group of edges are emitted in combinatorial order.
// ---------------------------------------------------------------------------
template <class F>
struct StreamOut {
    const BinomialTable* bt = nullptr;
    int vpc = 0;
    value_t diam = 0;
    int32_t base[2] = {0, 0};
    std::vector<int32_t> stack, merged;
    bool accumulate = false;
    std::vector<comb_t> akey;
    std::vector<int32_t> avert;
    F* emit = nullptr;

    // merge {u, v} with stack[0..k-1] (descending) into ascending order
    void put(int k) {
        int32_t* m = merged.data();
        int i = 0, j = k - 1, t = 0;
        while (i < 2 && j >= 0) m[t++] = (base[i] < stack[j]) ? base[i++] : stack[j--];
        while (i < 2) m[t++] = base[i++];
        while (j >= 0) m[t++] = stack[j--];
        if (accumulate) {
            akey.push_back(comb_index(m, vpc, *bt));
            avert.insert(avert.end(), m, m + vpc);
        } else {
            (*emit)(diam, m);
        }
    }
    void flush() {
        const size_t c = akey.size();
        if (!c) return;
        std::vector<uint32_t> ord(c);
        for (size_t i = 0; i < c; ++i) ord[i] = uint32_t(i);
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return akey[a] < akey[b]; });
        for (size_t i = 0; i < c; ++i) (*emit)(diam, &avert[size_t(ord[i]) * vpc]);
        akey.clear();
        avert.clear();
    }
};

// ---------------------------------------------------------------------------
// Geometry policies.
// ---------------------------------------------------------------------------

// Full Rips over a dense distance matrix; bitset adjacency.
struct DenseGeom {
    const rips::DenseComplex* cx = nullptr;
    int n = 0, words = 0;
    // radius graph (edges with rank < rptr), grown by add_radius
    std::vector<uint64_t> rbits;
    std::vector<int32_t> rlo, rhi;

    explicit DenseGeom(const rips::DenseComplex& c) : cx(&c), n(int(c.n)), words((int(c.n) + 63) / 64) {}

    value_t dist(int i, int j) const { return cx->dist(u32(i), u32(j)); }
    uint64_t pair_capacity() const { return uint64_t(n) * uint64_t(n > 0 ? n - 1 : 0) / 2; }
    uint64_t row_weight(int j) const { return uint64_t(j) + 1; }
    // f(i, d, bucket) for every i < j with d(i, j) <= thr. Row j's entries
    // below the diagonal are contiguous in both layouts.
    template <class F>
    void bucket_row(int j, value_t thr, value_t inv, int nb, F&& f) const {
        const f32* row = cx->full ? cx->mat.data() + u64(j) * cx->n
                                  : cx->mat.data() + u64(j) * (j - 1) / 2;
        int i = 0;
#ifndef TURBOTOPS_NO_SIMD
        const simde__m256 tv = simde_mm256_set1_ps(thr), iv = simde_mm256_set1_ps(inv),
                          nv = simde_mm256_set1_ps(value_t(nb));
        const simde__m256i lo = simde_mm256_setzero_si256(), hi = simde_mm256_set1_epi32(nb - 1);
        alignas(32) int32_t bk[8];
        for (; i + 8 <= j; i += 8) {
            const simde__m256 d = simde_mm256_loadu_ps(row + i);
            u32 mask = u32(simde_mm256_movemask_ps(simde_mm256_cmp_ps(d, tv, SIMDE_CMP_LE_OQ)));
            if (!mask) continue;
            // same operations and order as bucket_of: int((d - 0) * inv * nb)
            simde__m256i b = simde_mm256_cvttps_epi32(simde_mm256_mul_ps(simde_mm256_mul_ps(d, iv), nv));
            b = simde_mm256_min_epi32(simde_mm256_max_epi32(b, lo), hi);
            simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(bk), b);
            while (mask) {
                const int t = __builtin_ctz(mask);
                mask &= mask - 1;
                f(i + t, row[i + t], bk[t]);
            }
        }
#endif
        for (; i < j; ++i)
            if (row[i] <= thr) f(i, row[i], bucket_of(row[i], 0, inv, nb));
    }
    void set_ranks(const EdgeList&) {}

    void init_radius() {
        rbits.assign(size_t(words) * n, 0);
        rlo.assign(n, words);
        rhi.assign(n, -1);
    }
    void add_radius(int i, int j) {
        const int wj = j >> 6, wi = i >> 6;
        rbits[size_t(i) * words + wj] |= uint64_t(1) << (j & 63);
        rbits[size_t(j) * words + wi] |= uint64_t(1) << (i & 63);
        rlo[i] = std::min(rlo[i], wj);
        rhi[i] = std::max(rhi[i], wj);
        rlo[j] = std::min(rlo[j], wi);
        rhi[j] = std::max(rhi[j], wi);
    }
    bool radius_adj(int a, int b, size_t) const {
        return (rbits[size_t(a) * words + (b >> 6)] >> (b & 63)) & 1u;
    }
    // Calls ready(w) for every vertex w adjacent in the radius graph to all of
    // xverts and not marked in seenbits, until ready returns true.
    template <class Ready>
    bool any_common_radius_neighbor(const int32_t* xverts, int m, size_t,
                                    const std::vector<uint64_t>& seenbits,
                                    const std::vector<char>&, Ready&& ready) const {
        int wlo = 0, whi = words - 1;
        for (int i = 0; i < m; ++i) {
            wlo = std::max(wlo, rlo[xverts[i]]);
            whi = std::min(whi, rhi[xverts[i]]);
        }
        auto visit = [&](int w, uint64_t acc) {
            while (acc) {
                const int cand = w * 64 + ctz64(acc);
                acc &= acc - 1;
                if (cand >= n) return false;
                if (ready(int32_t(cand))) return true;
            }
            return false;
        };
        int w = wlo;
#ifndef TURBOTOPS_NO_SIMD
        alignas(32) uint64_t blk[4];
        for (; w + 4 <= whi + 1; w += 4) {
            simde__m256i acc = simde_mm256_loadu_si256(&rbits[size_t(xverts[0]) * words + w]);
            for (int i = 1; i < m; ++i)
                acc = simde_mm256_and_si256(acc, simde_mm256_loadu_si256(&rbits[size_t(xverts[i]) * words + w]));
            acc = simde_mm256_andnot_si256(simde_mm256_loadu_si256(&seenbits[w]), acc);
            if (simde_mm256_testz_si256(acc, acc)) continue;
            simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(blk), acc);
            for (int t = 0; t < 4; ++t)
                if (visit(w + t, blk[t])) return true;
        }
#endif
        for (; w <= whi; ++w) {
            uint64_t acc = rbits[size_t(xverts[0]) * words + w];
            for (int i = 1; i < m && acc; ++i) acc &= rbits[size_t(xverts[i]) * words + w];
            if (visit(w, acc & ~seenbits[w])) return true;
        }
        return false;
    }

    template <class F>
    void generate(const EdgeList& edges, int d, const BinomialTable& bt, F&& emit) const {
        if (edges.count == 0) return;
        const int need = d - 1;
        std::vector<uint64_t> g(size_t(words) * n, 0);   // G_<e
        std::vector<uint64_t> scratch(size_t(need + 1) * words, 0);
        StreamOut<std::remove_reference_t<F>> out;
        out.bt = &bt;
        out.vpc = d + 1;
        out.stack.assign(need + 1, 0);
        out.merged.assign(d + 2, 0);
        out.emit = &emit;
        // cand has no bits at or above word hiw
        auto rec = [&](auto&& self, uint64_t* cand, int hiw, int left, int depth) -> void {
            for (int w = 0; w < hiw; ++w) {
                uint64_t wb = cand[w];
                while (wb) {
                    const int x = w * 64 + ctz64(wb);
                    wb &= wb - 1;
                    if (x >= n) break;
                    out.stack[depth] = int32_t(x);
                    if (left == 1) {
                        out.put(depth + 1);
                        continue;
                    }
                    uint64_t* nxt = &scratch[size_t(depth + 1) * words];
                    const uint64_t* gx = g.data() + size_t(x) * words;
                    const int cw = x >> 6;                   // keep only vertices below x
                    and_words(nxt, cand, gx, cw + 1);
                    const int rem = x & 63;
                    nxt[cw] &= rem ? ((uint64_t(1) << rem) - 1) : 0;
                    self(self, nxt, cw + 1, left - 1, depth + 1);
                }
            }
        };
        const size_t E = edges.count;
        for (size_t i = 0; i < E;) {
            const value_t r = edges.diam[i];
            size_t jend = i;
            while (jend < E && edges.diam[jend] == r) ++jend;
            out.diam = r;
            out.accumulate = jend != i + 1;
            for (size_t k = i; k < jend; ++k) {
                const int u = edges.verts[2 * k], v = edges.verts[2 * k + 1];
                const uint64_t* gu = g.data() + size_t(u) * words;
                const uint64_t* gv = g.data() + size_t(v) * words;
                uint64_t* cand = scratch.data();
                const bool any = and_words(cand, gu, gv, words);
                out.base[0] = int32_t(u);
                out.base[1] = int32_t(v);
                if (any) rec(rec, cand, words, need, 0);
                g[size_t(u) * words + (v >> 6)] |= uint64_t(1) << (v & 63);
                g[size_t(v) * words + (u >> 6)] |= uint64_t(1) << (u & 63);
            }
            if (out.accumulate) out.flush();
            i = jend;
        }
    }
};

// Flag complex of a CsrGraph; each CSR entry carries its edge's rank
// (1-based filtration position, 0 = above the threshold). Rank is u32 when the
// graph has fewer than 2^32 - 1 edges (8 bytes per undirected edge instead of
// 16), u64 otherwise.
template <typename Rank>
struct SparseGeom {
    static_assert(std::is_unsigned_v<Rank>);
    const CsrGraph* g = nullptr;
    int n = 0;
    std::vector<Rank> rank;

    explicit SparseGeom(const CsrGraph& gr) : g(&gr), n(int(gr.n)) {}

    int64_t find(int i, int j) const {
        const u32* b = g->col_idx.data() + g->begin(u32(i));
        const u32* e = g->col_idx.data() + g->end(u32(i));
        const u32* it = std::lower_bound(b, e, u32(j));
        return (it != e && *it == u32(j)) ? int64_t(it - g->col_idx.data()) : -1;
    }
    value_t dist(int i, int j) const {
        if (i == j) return 0;
        const int64_t p = find(i, j);
        return p < 0 ? VINF : g->weights[size_t(p)];
    }
    int64_t rank_of(int i, int j) const {
        const int64_t p = find(i, j);
        return p < 0 ? 0 : rank[size_t(p)];
    }
    uint64_t pair_capacity() const { return g->col_idx.size() / 2; }
    uint64_t row_weight(int j) const { return g->degree(u32(j)) + 1; }
    template <class F>
    void bucket_row(int j, value_t thr, value_t inv, int nb, F&& f) const {
        for (u64 p = g->begin(u32(j)); p < g->end(u32(j)); ++p) {
            const u32 i = g->col_idx[p];
            if (i >= u32(j)) break;
            const value_t d = g->weights[p];
            if (d <= thr) f(int(i), d, bucket_of(d, 0, inv, nb));
        }
    }
    void set_ranks(const EdgeList& ed) {
        rank.assign(g->col_idx.size(), 0);
        parallel_ranges(ed.count, num_threads(), [&](u32, u64 lo, u64 hi) {   // disjoint writes
            for (u64 r = lo + 1; r <= hi; ++r) {
                const int u = ed.verts[2 * (r - 1)], v = ed.verts[2 * (r - 1) + 1];
                rank[size_t(find(u, v))] = Rank(r);
                rank[size_t(find(v, u))] = Rank(r);
            }
        });
    }

    void init_radius() {}
    void add_radius(int, int) {}
    bool radius_adj(int a, int b, size_t rptr) const {
        const int64_t r = rank_of(a, b);
        return r != 0 && size_t(r) < rptr;
    }
    template <class Ready>
    bool any_common_radius_neighbor(const int32_t* xverts, int m, size_t rptr,
                                    const std::vector<uint64_t>&,
                                    const std::vector<char>& seen, Ready&& ready) const {
        int best = 0;   // scan the shortest row
        for (int i = 1; i < m; ++i)
            if (g->degree(u32(xverts[i])) < g->degree(u32(xverts[best]))) best = i;
        const u32 v0 = u32(xverts[best]);
        for (u64 p = g->begin(v0); p < g->end(v0); ++p) {
            const int64_t rk = rank[p];
            if (rk == 0 || size_t(rk) >= rptr) continue;
            const int cand = int(g->col_idx[p]);
            if (seen[cand]) continue;
            bool ok = true;
            for (int i = 0; i < m && ok; ++i)
                if (i != best && (xverts[i] == cand || !radius_adj(xverts[i], cand, rptr))) ok = false;
            if (ok && ready(int32_t(cand))) return true;
        }
        return false;
    }

    template <class F>
    void generate(const EdgeList& edges, int d, const BinomialTable& bt, F&& emit) const {
        if (edges.count == 0) return;
        const int need = d - 1;
        StreamOut<std::remove_reference_t<F>> out;
        out.bt = &bt;
        out.vpc = d + 1;
        out.stack.assign(need + 1, 0);
        out.merged.assign(d + 2, 0);
        out.emit = &emit;
        std::vector<std::vector<int32_t>> lists(need + 1);
        int64_t kcur = 0;   // edges of rank <= kcur form G_<e
        auto rec = [&](auto&& self, int left, int depth) -> void {
            const std::vector<int32_t>& cand = lists[depth];
            for (size_t t = 0; t < cand.size(); ++t) {
                const int x = cand[t];
                out.stack[depth] = int32_t(x);
                if (left == 1) {
                    out.put(depth + 1);
                    continue;
                }
                std::vector<int32_t>& nxt = lists[depth + 1];
                nxt.clear();
                // cand is ascending: entries before t are those below x; merge
                // them against x's sorted row
                u64 p = g->begin(u32(x));
                const u64 pe = g->end(u32(x));
                size_t q = 0;
                while (q < t && p < pe) {
                    const int32_t a = int32_t(g->col_idx[p]), c = cand[q];
                    if (a < c) ++p;
                    else if (c < a) ++q;
                    else {
                        const int64_t r = rank[p];
                        if (r != 0 && r <= kcur) nxt.push_back(c);
                        ++p;
                        ++q;
                    }
                }
                if (!nxt.empty()) self(self, left - 1, depth + 1);
            }
        };
        const size_t E = edges.count;
        for (size_t i = 0; i < E;) {
            const value_t r = edges.diam[i];
            size_t jend = i;
            while (jend < E && edges.diam[jend] == r) ++jend;
            out.diam = r;
            out.accumulate = jend != i + 1;
            for (size_t k = i; k < jend; ++k) {
                kcur = int64_t(k);
                const int u = edges.verts[2 * k], v = edges.verts[2 * k + 1];
                std::vector<int32_t>& cand = lists[0];
                cand.clear();
                u64 pu = g->begin(u32(u)), pv = g->begin(u32(v));
                const u64 eu = g->end(u32(u)), ev = g->end(u32(v));
                while (pu < eu && pv < ev) {
                    const u32 a = g->col_idx[pu], b = g->col_idx[pv];
                    if (a < b) ++pu;
                    else if (b < a) ++pv;
                    else {
                        const int64_t ra = rank[pu], rb = rank[pv];
                        if (ra != 0 && ra <= kcur && rb != 0 && rb <= kcur) cand.push_back(int32_t(a));
                        ++pu;
                        ++pv;
                    }
                }
                out.base[0] = int32_t(u);
                out.base[1] = int32_t(v);
                if (!cand.empty()) rec(rec, need, 0);
            }
            if (out.accumulate) out.flush();
            i = jend;
        }
    }
};

// ---------------------------------------------------------------------------
// Active-simplex registry: the keys of R_n are the active n-simplices; for
// every (n-1)-face it lists the active simplices containing it.
// ---------------------------------------------------------------------------
struct ActiveReg {
    int m = 0;                              // vertices per top simplex
    FastMap slot_of{1024};                  // active rank -> vertex slot
    std::vector<int32_t> verts;             // m per slot
    std::vector<slot_t> freelist;
    size_t nslots = 0;
    FaceSlot face_slot;                     // face index -> list slot
    std::vector<std::vector<rank_t>> lr;    // per list: active ranks
    std::vector<std::vector<int32_t>> lw;   // per list: the extra vertex
    std::vector<std::vector<slot_t>> ls;    // per list: that active's R slot
    std::vector<slot_t> lfree;
};

// ---------------------------------------------------------------------------
// The algorithm.
// ---------------------------------------------------------------------------
template <typename G>
struct Engine {
    G& geo;
    int nv = 0, n = 0;                  // vertices, top homology dimension
    BinomialTable binom;
    std::vector<SetDict> R, Ri;
    std::vector<std::vector<std::pair<value_t, value_t>>> pairs;
    EdgeList edges;
    FaceSlot csmap;                     // face index -> R slot, dimension csdim
    bool cson = false;
    int csdim = 0;
    SSet bar, buf, touched, others, topbar, swapbuf;
    std::vector<rank_t> removed;
    ActiveReg act;
    std::vector<char> seen;
    std::vector<uint64_t> seenbits;
    std::vector<int32_t> seedverts;
    std::vector<std::vector<slot_t>> seedlists;
    size_t rptr = 1;                    // edges of rank < rptr are in the radius graph

    Engine(G& g, int maxdim) : geo(g), nv(g.n), n(maxdim) {}

    value_t lvl_diam_of(int p, rank_t r) const { return p == 1 ? edges.dm(size_t(r)) : Ri[p].side_diam(r); }
    const int32_t* lvl_verts_of(int p, rank_t r) const { return p == 1 ? edges.vp(size_t(r)) : R[p].side_vert(r); }

    void cs_clear(rank_t z, bool apply) {
        if (apply) csmap.set(comb_index(lvl_verts_of(csdim, z), csdim + 1, binom), 0);
    }
    // the reduced boundary is {y}: remove y from every R[z] containing it
    void remove_maximal(SetDict& Rd, SetDict& Rid, rank_t y, bool apply_cs) {
        removed.clear();
        touched = Rid.at(Rid.slot(y));
        Rid.erase(y);
        for (rank_t z : touched) {
            const slot_t zs = Rd.slot(z);
            Rd.remove_elem(zs, y);
            if (Rd.at(zs).empty()) {
                cs_clear(z, apply_cs);
                Rd.erase(z);
                removed.push_back(z);
            }
        }
    }
    // general case: |bar| >= 2, j = its youngest element
    void replace_pivot(SetDict& Rd, SetDict& Rid, const SSet& b, rank_t j, bool apply_cs) {
        removed.clear();
        touched = Rid.at(Rid.slot(j));
        others.clear();
        for (rank_t y : b)
            if (y != j) others.push_back(y);
        for (rank_t z : touched) {
            const slot_t zs = Rd.slot(z);
            Rd.xor_into(zs, b, swapbuf);
            if (Rd.at(zs).empty()) {
                cs_clear(z, apply_cs);
                Rd.erase(z);
                removed.push_back(z);
            }
        }
        const size_t K = touched.size();
        for (rank_t y : others) {
            SSet& sy = Rid.at(Rid.slot(y));
            if (K * 128 < sy.size())
                for (rank_t z : touched) sset_toggle(sy, z);
            else
                sset_xor(sy, touched, buf);
            if (sy.empty()) Rid.erase(y);
        }
        Rid.erase(j);
    }
    void unregister_active(const std::vector<rank_t>& doomed) {
        ActiveReg& ar = act;
        for (rank_t rk : doomed) {
            const slot_t slot = ar.slot_of.get(rk);
            if (!slot) continue;
            const int32_t* base = &ar.verts[size_t(slot - 1) * ar.m];
            comb_t fi[33];
            face_indices(base, ar.m, binom, fi);
            for (int omit = 1; omit <= ar.m; ++omit) {
                const comb_t f = fi[omit - 1];
                const slot_t lsx = ar.face_slot.get(f);
                if (!lsx) continue;
                auto& lr = ar.lr[lsx - 1];
                auto& lw = ar.lw[lsx - 1];
                auto& lsl = ar.ls[lsx - 1];
                for (size_t k = 0; k < lr.size(); ++k)
                    if (lr[k] == rk) {
                        lr[k] = lr.back(); lr.pop_back();
                        lw[k] = lw.back(); lw.pop_back();
                        lsl[k] = lsl.back(); lsl.pop_back();
                        break;
                    }
                if (lr.empty()) {
                    ar.face_slot.set(f, 0);
                    ar.lfree.push_back(lsx);
                }
            }
            ar.slot_of.erase(rk);
            ar.freelist.push_back(slot);
        }
    }
    void register_active(rank_t rk, slot_t rslot, const int32_t* verts) {
        ActiveReg& ar = act;
        slot_t slot;
        if (ar.freelist.empty()) {
            ++ar.nslots;
            ar.verts.resize(ar.nslots * size_t(ar.m));
            slot = slot_t(ar.nslots);
        } else {
            slot = ar.freelist.back();
            ar.freelist.pop_back();
        }
        int32_t* base = &ar.verts[size_t(slot - 1) * ar.m];
        for (int i = 0; i < ar.m; ++i) base[i] = verts[i];
        ar.slot_of.set(rk, slot);
        comb_t fi[33];
        face_indices(base, ar.m, binom, fi);
        for (int omit = 1; omit <= ar.m; ++omit) {
            const comb_t f = fi[omit - 1];
            slot_t lsx = ar.face_slot.get(f);
            if (!lsx) {
                if (ar.lfree.empty()) {
                    ar.lr.emplace_back();
                    ar.lw.emplace_back();
                    ar.ls.emplace_back();
                    lsx = slot_t(ar.lr.size());
                } else {
                    lsx = ar.lfree.back();
                    ar.lfree.pop_back();
                }
                ar.face_slot.set(f, lsx);
            }
            ar.lr[lsx - 1].push_back(rk);
            ar.lw[lsx - 1].push_back(base[omit - 1]);
            ar.ls[lsx - 1].push_back(rslot);
        }
    }
    // a nonzero reduced boundary of a d-simplex of diameter sdiam: a death
    void resolve_bar(int d, SSet& b, value_t sdiam) {
        SetDict& Rd = R[d - 1];
        SetDict& Rid = Ri[d - 1];
        const bool apply_cs = cson && d - 1 == csdim;
        const rank_t j = b.back();
        const value_t jdiam = lvl_diam_of(d - 1, j);
        if (b.size() == 1) remove_maximal(Rd, Rid, j, apply_cs);
        else replace_pivot(Rd, Rid, b, j, apply_cs);
        if (d == n + 1 && !removed.empty()) unregister_active(removed);
        if (sdiam != jdiam) pairs[d - 1].push_back({jdiam, sdiam});
    }

    void advance_radius(value_t r) {
        size_t p = rptr;
        while (p <= edges.count && edges.diam[p - 1] <= r) {
            geo.add_radius(edges.verts[2 * (p - 1)], edges.verts[2 * (p - 1) + 1]);
            ++p;
        }
        rptr = p;
    }

    // The simplex x whose cofacets active enumeration examines, with the
    // positions covered by every one of its edges of length >= its diameter.
    // cover is computed on first use: most candidates never need it, and its
    // distance reads are cache misses. An edge is its own maximal edge.
    struct XInfo {
        const int32_t* v;   // ascending
        int m;
        comb_t idx;
        value_t diam;
        mutable uint32_t cover;   // AND over x's maximal edges of their position masks
        mutable bool have_cover;
    };
    XInfo xinfo(const int32_t* v, int m, comb_t idx, value_t diam) const {
        return XInfo{v, m, idx, diam, 3u, m == 2};
    }
    uint32_t cover_of(const XInfo& x) const {
        if (!x.have_cover) {
            uint32_t c = ~0u;
            for (int a = 0; a < x.m; ++a)
                for (int b = a + 1; b < x.m; ++b)
                    if (!(geo.dist(x.v[a], x.v[b]) < x.diam)) c &= (1u << a) | (1u << b);
            x.cover = c;
            x.have_cover = true;
        }
        return x.cover;
    }
    // Is x u {w} ready to appear, i.e. is x its youngest face? A face omitting
    // x's vertex v_j is younger than x exactly when v_j < w (colexicographic
    // order: the two sets differ only in v_j and w), and it is then required
    // to have diameter < diam(x), i.e. every edge of x u {w} with length >=
    // diam(x) must contain v_j. With J = the positions of x below w, that is:
    // J is inside every maximal edge of x (J & ~cover == 0), and no edge w-v_i
    // of length >= diam(x) misses part of J.
    bool coface_ready(const XInfo& x, int32_t w) const {
        int pos = 0;
        while (pos < x.m && x.v[pos] < w) ++pos;
        if (pos == 0) return true;
        const uint32_t J = (pos >= 32) ? ~0u : (1u << pos) - 1;
        if (J & ~cover_of(x)) return false;
        for (int i = 0; i < x.m; ++i)
            if ((J & ~(1u << i)) && !(geo.dist(w, x.v[i]) < x.diam)) return false;
        return true;
    }
    // Step 1: extra vertices w such that x u {w} contains another active simplex
    void collect_candidates(const int32_t* xverts, const comb_t* fpre) {
        for (int32_t w : seedverts) {
            seen[w] = 0;
            seedlists[w].clear();
            seenbits[w >> 6] = 0;
        }
        seedverts.clear();
        const int m = act.m;
        comb_t fbuf[33];
        const comb_t* fi = fpre;
        if (!fi) {
            face_indices(xverts, m, binom, fbuf);
            fi = fbuf;
        }
        for (int p = 0; p < m; ++p) act.face_slot.prefetch(fi[p]);
        for (int omit = 1; omit <= m; ++omit) {
            const slot_t lsx = act.face_slot.get(fi[omit - 1]);
            if (!lsx) continue;
            const int xextra = xverts[omit - 1];
            const auto& lw = act.lw[lsx - 1];
            const auto& lsl = act.ls[lsx - 1];
            for (size_t t = 0; t < lw.size(); ++t) {
                const int w = lw[t];
                if (!geo.radius_adj(xextra, w, rptr)) continue;
                if (!seen[w]) {
                    seen[w] = 1;
                    seedverts.push_back(w);
                    seenbits[w >> 6] |= uint64_t(1) << (w & 63);
                }
                seedlists[w].push_back(lsl[t]);
            }
        }
    }
    // Step 2: does a coface not reached in step 1 exist?
    bool has_leftover_coface(const XInfo& x) {
        return geo.any_common_radius_neighbor(x.v, x.m, rptr, seenbits, seen,
                                              [&](int32_t cand) { return coface_ready(x, cand); });
    }
    void xor_many(SSet& out, SetDict& Rn, slot_t xslot, const std::vector<slot_t>& seeds) {
        out.clear();
        const size_t k = seeds.size();
        size_t start = 0;
        if (xslot) {
            if (k >= 1) { sset_xor2(out, Rn.at(xslot), Rn.at(seeds[0])); start = 1; }
            else { out = Rn.at(xslot); return; }
        } else if (k >= 2) {
            sset_xor2(out, Rn.at(seeds[0]), Rn.at(seeds[1]));
            start = 2;
        }
        for (size_t t = start; t < k; ++t) sset_xor(out, Rn.at(seeds[t]), buf);
    }
    // x is a new top-dimensional simplex with an empty reduced boundary;
    // create() inserts its singleton sets {x} into R[n] and Ri[n]. When step 1
    // finds no candidate vertex, those sets are never read before x is either
    // killed by a leftover coface or registered, so they are created only if
    // x survives (most die at once; this saves four hash operations each).
    template <class Create>
    void run_active_enumeration(rank_t xrank, comb_t xidx, const int32_t* verts, value_t sdiam,
                                const comb_t* faces, Create&& create) {
        const int m = n + 1;
        advance_radius(sdiam);
        collect_candidates(verts, faces);
        SetDict& Rn = R[n];
        const XInfo xi = xinfo(verts, m, xidx, sdiam);
        if (seedverts.empty()) {
            if (has_leftover_coface(xi)) return;
            create();
            register_active(xrank, Rn.slot(xrank), verts);
            return;
        }
        create();
        slot_t xslot = Rn.slot(xrank);
        bool dirty = false;
        for (size_t t = 0; t < seedverts.size(); ++t) {
            const int32_t w = seedverts[t];
            const std::vector<slot_t>& sdw = seedlists[w];
            if (seeded_is_noop(Rn, xslot, sdw)) continue;
            if (coface_ready(xi, w)) {
                xor_many(topbar, Rn, xslot, sdw);
                if (!topbar.empty()) {
                    resolve_bar(n + 1, topbar, sdiam);
                    dirty = true;
                }
            }
        }
        if (!dirty) {
            if (has_leftover_coface(xi)) {
                Rn.erase(xrank);
                Ri[n].erase(xrank);
                return;
            }
            register_active(xrank, xslot, verts);
            return;
        }
        xslot = Rn.slot(xrank);
        if (xslot) {
            if (has_leftover_coface(xi)) {
                topbar = Rn.at(xslot);
                resolve_bar(n + 1, topbar, sdiam);
            }
            xslot = Rn.slot(xrank);
        }
        if (xslot) register_active(xrank, xslot, verts);
    }
    void emit_infinite(int p) {
        if (p < 1) return;
        std::vector<rank_t> live;
        Ri[p].keys(live);
        std::sort(live.begin(), live.end());
        for (rank_t x : live) pairs[p].push_back({lvl_diam_of(p, x), VINF});
    }

    void run(value_t threshold) {
        binom = BinomialTable(nv, n + 2);
        build_edges(geo, threshold, binom, edges);
        geo.set_ranks(edges);
        geo.init_radius();
        const int words = (nv + 63) / 64;
        R.resize(n + 1);
        Ri.resize(n + 1);
        pairs.assign(n + 1, {});
        RootForest uf(nv);
        act.m = n + 1;
        act.face_slot.init(binom(nv, n));
        seen.assign(nv, 0);
        seenbits.assign(words, 0);
        seedlists.assign(nv, {});
        std::vector<int32_t> top(n + 1, 0);

        // dimensions 0 and 1
        for (size_t e = 1; e <= edges.count; ++e) {
            const int lo = edges.verts[2 * (e - 1)], hi = edges.verts[2 * (e - 1) + 1];
            const int32_t a = uf.find(lo), b = uf.find(hi);
            const value_t ediam = edges.diam[e - 1];
            if (a == b) {
                if (n >= 1) {
                    const rank_t er = rank_t(e);
                    auto create = [&] {
                        R[1].singleton(er, er);
                        Ri[1].singleton(er, er);
                    };
                    if (n > 1) create();
                    else {
                        top[0] = lo;
                        top[1] = hi;
                        run_active_enumeration(er, comb_index(edges.vp(e), 2, binom), top.data(), ediam, nullptr, create);
                    }
                }
            } else {
                uf.parent[std::max(a, b)] = std::min(a, b);   // the younger root dies
                if (ediam != 0) pairs[0].push_back({0, ediam});
            }
        }

        // dimensions >= 2
        for (int d = 2; d <= n; ++d) {
            csmap.init(binom(nv, d));
            {
                std::vector<rank_t> live;
                R[d - 1].keys(live);
                for (rank_t z : live)
                    csmap.set(comb_index(lvl_verts_of(d - 1, z), d, binom), R[d - 1].slot(z));
            }
            cson = true;
            csdim = d - 1;
            const int m = d + 1;
            SetDict& Rd = R[d];
            SetDict& Rid = Ri[d];
            SetDict& Rprev = R[d - 1];
            Rd.side_vpc = m;
            Rid.side_vpc = m;
            rank_t next_rank = 0;
            geo.generate(edges, d, binom, [&](value_t sdiam, const int32_t* cv) {
                const rank_t sr = ++next_rank;
                bar.clear();
                comb_t fi[33];
                face_indices(cv, m, binom, fi);
                for (int p = 0; p < m; ++p) csmap.prefetch(fi[p]);
                for (int p = 0; p < m; ++p) {
                    const slot_t fs = csmap.get(fi[p]);
                    if (fs) sset_xor(bar, Rprev.at(fs), buf);
                }
                if (bar.empty()) {
                    auto create = [&] {
                        Rd.set_side(Rd.singleton(sr, sr), sdiam, cv);
                        Rid.set_side(Rid.singleton(sr, sr), sdiam, cv);
                    };
                    if (d == n)
                        run_active_enumeration(sr, fi[m - 1] + binom.at(cv[m - 1], m), cv, sdiam, fi, create);
                    else create();
                } else {
                    resolve_bar(d, bar, sdiam);
                }
            });
            cson = false;
            csmap.init(0);
            emit_infinite(d - 1);
            R[d - 1] = SetDict();
            Ri[d - 1] = SetDict();
        }

        for (int v = 0; v < nv; ++v)
            if (uf.find(v) == v) pairs[0].push_back({0, VINF});
        emit_infinite(n);
    }
};

template <typename G>
inline rips::Barcode run(G& geo, u32 maxdim, f32 threshold, u64* edges = nullptr) {
    // Same envelope as the coboundary engine: the combinatorial-index tables
    // and 32-bit position masks below assume maxdim + 1 <= 32.
    if (maxdim > 30)
        throw std::invalid_argument("redzed: maxdim must be <= 30");
    if (geo.n > std::numeric_limits<int32_t>::max())
        throw std::overflow_error("redzed: more than 2^31 - 1 vertices");
    rips::Barcode out;
    out.ensure(maxdim);
    if (geo.n == 0) return out;
    Engine<G> eng(geo, int(maxdim));
    eng.run(threshold);
    if (edges) *edges = eng.edges.count;
    for (u32 d = 0; d <= maxdim; ++d)
        for (const auto& pr : eng.pairs[d]) out.add(d, pr.first, pr.second);
    return out;
}

}  // namespace detail

// True when every simplex index the engine needs fits in 64 bits.
inline bool fits(u32 n, u32 maxdim) {
    try {
        detail::BinomialTable b(int(std::min<u64>(n, u64(std::numeric_limits<int32_t>::max()))),
                                int(maxdim) + 2);
        return true;
    } catch (const std::overflow_error&) {
        return false;
    }
}

// Full Rips of a dense complex, cut at cx.threshold (the enclosing radius
// when the threshold is infinite, as in rips::compute_dense).
inline rips::Barcode compute(const rips::DenseComplex& cx, u32 maxdim, u64* edges = nullptr) {
    detail::DenseGeom geo(cx);
    return detail::run(geo, maxdim, cx.threshold, edges);
}

// Flag complex of a weighted graph (rows sorted by neighbor id).
// wide_rank forces 64-bit edge ranks (testing).
inline rips::Barcode compute_graph(const CsrGraph& g, u32 maxdim, f32 threshold = INFINITY,
                                   u64* edges = nullptr, bool wide_rank = false) {
    if (!wide_rank && g.m < u64(std::numeric_limits<uint32_t>::max())) {
        detail::SparseGeom<uint32_t> geo(g);
        return detail::run(geo, maxdim, threshold, edges);
    }
    detail::SparseGeom<uint64_t> geo(g);
    return detail::run(geo, maxdim, threshold, edges);
}

}  // namespace tops::redzed

#endif  // TURBOTOPS_REDZED_HPP
