// SPDX-License-Identifier: MIT
// core.hpp — basic types, a minimal thread pool, the CSR graph, union-find.
#ifndef TURBOTOPS_CORE_HPP
#define TURBOTOPS_CORE_HPP

// Release version, kept in sync with CHANGELOG.md and the git tag.
// TURBOTOPS_VERSION_STRING is always "MAJOR.MINOR.PATCH".
#define TURBOTOPS_VERSION_MAJOR 0
#define TURBOTOPS_VERSION_MINOR 1
#define TURBOTOPS_VERSION_PATCH 0
#define TURBOTOPS_VERSION_STRING "0.1.0"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

namespace tops {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using f32 = float;

// ---------------------------------------------------------------------------
// Threads. Every parallel phase writes disjoint outputs or is followed by a
// total-order sort, so results never depend on the thread count.
// ---------------------------------------------------------------------------
namespace detail {
inline std::atomic<u32>& thread_setting() {
    static std::atomic<u32> t{0};   // 0 = hardware concurrency
    return t;
}
}  // namespace detail

inline void set_num_threads(u32 t) { detail::thread_setting().store(t); }

inline u32 num_threads() {
    const u32 t = detail::thread_setting().load();
    if (t) return t;
    return std::max<u32>(1, std::thread::hardware_concurrency());
}

// fn(i) for i in [0, n), dynamically scheduled.
template <typename F>
inline void parallel_for(u64 n, F&& fn) {
    const u32 nt = u32(std::min<u64>(num_threads(), n));
    if (nt <= 1) {
        for (u64 i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<u64> next{0};
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (u32 t = 0; t < nt; ++t)
        pool.emplace_back([&] {
            for (;;) {
                const u64 i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) break;
                fn(i);
            }
        });
    for (auto& th : pool) th.join();
}

// fn(t, lo, hi) over nt static contiguous ranges of [0, n).
template <typename F>
inline void parallel_ranges(u64 n, u32 nt, F&& fn) {
    nt = u32(std::min<u64>(std::max<u32>(nt, 1), std::max<u64>(n, 1)));
    if (nt <= 1) {
        fn(0u, u64(0), n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (u32 t = 0; t < nt; ++t) {
        const u64 lo = n * t / nt, hi = n * (t + 1) / nt;
        pool.emplace_back([&, t, lo, hi] { fn(t, lo, hi); });
    }
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------
// Undirected weighted graph in CSR form. Each edge is stored in both rows;
// rows are sorted by neighbor id; no self loops.
// ---------------------------------------------------------------------------
struct CsrGraph {
    u32 n = 0;
    u64 m = 0;                   // undirected edges
    std::vector<u64> row_ptr;    // n + 1 offsets
    std::vector<u32> col_idx;    // 2m
    std::vector<f32> weights;    // 2m

    u64 begin(u32 v) const { return row_ptr[v]; }
    u64 end(u32 v) const { return row_ptr[v + 1]; }
    u64 degree(u32 v) const { return end(v) - begin(v); }

    // Weight of edge (u, v), +inf if absent.
    f32 weight(u32 u, u32 v) const {
        const u32* b = col_idx.data() + begin(u);
        const u32* e = col_idx.data() + end(u);
        const u32* it = std::lower_bound(b, e, v);
        return (it != e && *it == v) ? weights[u64(it - col_idx.data())] : INFINITY;
    }

    // Build from per-vertex neighbor lists (each edge present in both lists;
    // lists need not be sorted).
    static CsrGraph from_rows(std::vector<std::vector<std::pair<u32, f32>>>& rows) {
        CsrGraph g;
        g.n = u32(rows.size());
        g.row_ptr.assign(u64(g.n) + 1, 0);
        for (u32 v = 0; v < g.n; ++v) g.row_ptr[v + 1] = g.row_ptr[v] + rows[v].size();
        const u64 nnz = g.row_ptr[g.n];
        g.m = nnz / 2;
        g.col_idx.resize(nnz);
        g.weights.resize(nnz);
        parallel_for(g.n, [&](u64 v) {
            auto& r = rows[v];
            std::sort(r.begin(), r.end());
            u64 o = g.row_ptr[v];
            for (const auto& e : r) {
                g.col_idx[o] = e.first;
                g.weights[o] = e.second;
                ++o;
            }
            std::vector<std::pair<u32, f32>>().swap(r);
        });
        return g;
    }

    // Structural checks: offsets, sorted rows, symmetric weights, no loops.
    bool valid() const {
        if (row_ptr.size() != u64(n) + 1 || row_ptr[0] != 0 || row_ptr[n] != 2 * m)
            return false;
        for (u32 v = 0; v < n; ++v)
            for (u64 t = begin(v); t < end(v); ++t) {
                const u32 u = col_idx[t];
                if (u >= n || u == v) return false;
                if (t > begin(v) && col_idx[t - 1] >= u) return false;
                if (weight(u, v) != weights[t]) return false;
            }
        return true;
    }
};

struct UnionFind {
    std::vector<u32> parent, rank_;
    explicit UnionFind(u32 n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0u);
    }
    u32 find(u32 x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(u32 a, u32 b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

}  // namespace tops

#endif  // TURBOTOPS_CORE_HPP
