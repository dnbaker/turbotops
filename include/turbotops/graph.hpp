// SPDX-License-Identifier: MIT
// graph.hpp — neighborhood graphs over a point cloud.
//
//   radius_graph(cloud, tau)        all pairs with d <= tau (exact)
//   knn_lists_exact(cloud, k)       exact k nearest neighbors per point
//   knn_lists_hnsw(cloud, k, ...)   approximate k nearest neighbors (HNSW)
//   knn_graph(cloud, lists)         symmetric union of the lists, exact weights
//   knn_recall(cloud, lists, ...)   measured recall of approximate lists
//
// Exact search uses a k-d tree. For any Minkowski metric with p >= 1, the
// distance from a query to any point inside an axis-aligned box is at least
// the metric applied to the per-axis gaps between the query and the box, so
// subtrees are pruned with that bound. Results are exact for every input;
// pruning is effective in low and moderate dimension.
//
// Ties in kNN are broken by point index, so exact lists are deterministic.
#ifndef TURBOTOPS_GRAPH_HPP
#define TURBOTOPS_GRAPH_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core.hpp"
#include "metric.hpp"

#ifndef TURBOTOPS_NO_HNSW
#include "hnswlib/hnswlib.h"
#endif

namespace tops {

// ---------------------------------------------------------------------------
// k-d tree over a point cloud (median splits on the axis of largest spread,
// per-node bounding boxes).
// ---------------------------------------------------------------------------
struct KdTree {
    static constexpr u32 LEAF = 16;
    struct Node {
        u32 begin, end;        // range in `perm`
        u32 left = 0, right = 0;   // children (0 = leaf)
    };
    const PointCloud* c = nullptr;
    std::vector<u32> perm;
    std::vector<Node> nodes;
    std::vector<f32> box;      // per node: dim lows, then dim highs

    explicit KdTree(const PointCloud& cloud) : c(&cloud) {
        perm.resize(c->n);
        for (u32 i = 0; i < c->n; ++i) perm[i] = i;
        if (c->n) build(0, c->n);
    }

    const f32* lo(u32 node) const { return box.data() + u64(node) * 2 * c->dim; }
    const f32* hi(u32 node) const { return lo(node) + c->dim; }

    // Lower bound on d(q, x) for x in the node's box.
    f32 box_dist(u32 node, const f32* q, f32* gap) const {
        const f32 *l = lo(node), *h = hi(node);
        for (u32 a = 0; a < c->dim; ++a)
            gap[a] = std::max(0.0f, std::max(l[a] - q[a], q[a] - h[a]));
        static thread_local std::vector<f32> zero;
        if (zero.size() < c->dim) zero.assign(c->dim, 0.0f);
        return c->metric(gap, zero.data(), c->dim);
    }

  private:
    u32 build(u32 b, u32 e) {
        const u32 id = u32(nodes.size());
        nodes.push_back(Node{b, e});
        const u32 dim = c->dim;
        box.resize(box.size() + 2 * dim);
        f32* l = box.data() + u64(id) * 2 * dim;
        f32* h = l + dim;
        for (u32 a = 0; a < dim; ++a) {
            l[a] = INFINITY;
            h[a] = -INFINITY;
        }
        for (u32 t = b; t < e; ++t) {
            const f32* x = c->x + u64(perm[t]) * dim;
            for (u32 a = 0; a < dim; ++a) {
                l[a] = std::min(l[a], x[a]);
                h[a] = std::max(h[a], x[a]);
            }
        }
        if (e - b <= LEAF) return id;
        u32 axis = 0;
        f32 spread = -1.0f;
        for (u32 a = 0; a < dim; ++a)
            if (h[a] - l[a] > spread) {
                spread = h[a] - l[a];
                axis = a;
            }
        if (spread <= 0.0f) return id;   // all points identical
        const u32 mid = b + (e - b) / 2;
        std::nth_element(perm.begin() + b, perm.begin() + mid, perm.begin() + e, [&](u32 p, u32 q) {
            const f32 xp = c->x[u64(p) * dim + axis], xq = c->x[u64(q) * dim + axis];
            return xp < xq || (xp == xq && p < q);
        });
        const u32 left = build(b, mid);
        const u32 right = build(mid, e);
        nodes[id].left = left;
        nodes[id].right = right;
        return id;
    }
};

namespace detail {
using Cand = std::pair<f32, u32>;   // (distance, id): lexicographic total order

// Prune a subtree only when its lower bound clearly exceeds the cutoff: the
// bound and the point distances may round differently, and pruning less
// never changes the result.
inline bool beyond(f32 lower_bound, f32 cutoff) {
    return lower_bound > cutoff + 1e-5f * cutoff;
}

// Exact k nearest neighbors of point i (excluding i), ascending by
// (distance, id), written to out[0..k).
inline void knn_query(const KdTree& t, u32 i, u32 k, std::vector<Cand>& heap,
                      std::vector<u32>& stack, std::vector<f32>& gap, u32* out) {
    const PointCloud& c = *t.c;
    const f32* q = c.x + u64(i) * c.dim;
    gap.resize(c.dim);
    heap.clear();
    stack.clear();
    stack.push_back(0);
    while (!stack.empty()) {
        const u32 nd = stack.back();
        stack.pop_back();
        const f32 bound = heap.size() < k ? INFINITY : heap.front().first;
        if (beyond(t.box_dist(nd, q, gap.data()), bound)) continue;
        const KdTree::Node& N = t.nodes[nd];
        if (!N.left) {
            for (u32 s = N.begin; s < N.end; ++s) {
                const u32 j = t.perm[s];
                if (j == i) continue;
                const Cand cand{c(i, j), j};
                if (heap.size() < k) {
                    heap.push_back(cand);
                    std::push_heap(heap.begin(), heap.end());
                } else if (cand < heap.front()) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = cand;
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            continue;
        }
        // Visit the nearer child first (pushed last).
        const f32 dl = t.box_dist(N.left, q, gap.data());
        const f32 dr = t.box_dist(N.right, q, gap.data());
        if (dl <= dr) {
            stack.push_back(N.right);
            stack.push_back(N.left);
        } else {
            stack.push_back(N.left);
            stack.push_back(N.right);
        }
    }
    std::sort(heap.begin(), heap.end());
    for (u32 s = 0; s < k; ++s) out[s] = heap[s].second;
}
}  // namespace detail

// All pairs with d(i, j) <= tau.
inline CsrGraph radius_graph(const PointCloud& c, f32 tau) {
    const KdTree t(c);
    std::vector<std::vector<std::pair<u32, f32>>> rows(c.n);
    parallel_for(c.n, [&](u64 ii) {
        const u32 i = u32(ii);
        const f32* q = c.x + ii * c.dim;
        thread_local std::vector<u32> stack;
        thread_local std::vector<f32> gap;
        gap.resize(c.dim);
        stack.clear();
        stack.push_back(0);
        auto& row = rows[i];
        while (!stack.empty()) {
            const u32 nd = stack.back();
            stack.pop_back();
            if (detail::beyond(t.box_dist(nd, q, gap.data()), tau)) continue;
            const KdTree::Node& N = t.nodes[nd];
            if (N.left) {
                stack.push_back(N.left);
                stack.push_back(N.right);
                continue;
            }
            for (u32 s = N.begin; s < N.end; ++s) {
                const u32 j = t.perm[s];
                if (j == i) continue;
                const f32 d = c(i, j);
                if (d <= tau) row.emplace_back(j, d);
            }
        }
    });
    return CsrGraph::from_rows(rows);
}

// k nearest neighbors of every point (row-major n x k ids, ascending by
// (distance, id)). k is clamped to n - 1.
struct KnnLists {
    u32 n = 0, k = 0;
    std::vector<u32> ids;
    const u32* row(u32 i) const { return ids.data() + u64(i) * k; }
};

inline KnnLists knn_lists_exact(const PointCloud& c, u32 k) {
    KnnLists L;
    L.n = c.n;
    L.k = c.n ? std::min<u32>(k, c.n - 1) : 0;
    L.ids.resize(u64(L.n) * L.k);
    if (!L.k) return L;
    const KdTree t(c);
    parallel_for(c.n, [&](u64 i) {
        thread_local std::vector<detail::Cand> heap;
        thread_local std::vector<u32> stack;
        thread_local std::vector<f32> gap;
        detail::knn_query(t, u32(i), L.k, heap, stack, gap, L.ids.data() + i * L.k);
    });
    return L;
}

#ifndef TURBOTOPS_NO_HNSW
namespace detail {
struct MinkowskiParams {
    u32 dim;
    Metric metric;
};
inline float minkowski_hnsw(const void* a, const void* b, const void* p) {
    const auto* q = static_cast<const MinkowskiParams*>(p);
    return q->metric(static_cast<const f32*>(a), static_cast<const f32*>(b), q->dim);
}
struct MinkowskiSpace : hnswlib::SpaceInterface<float> {
    MinkowskiParams prm;
    MinkowskiSpace(u32 dim, Metric m) : prm{dim, m} {}
    size_t get_data_size() override { return size_t(prm.dim) * sizeof(f32); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return &minkowski_hnsw; }
    void* get_dist_func_param() override { return &prm; }
};
}  // namespace detail

struct HnswParams {
    u32 M = 16;
    u32 ef_construction = 200;
    u32 ef_search = 64;
    u64 seed = 100;
};

// Approximate kNN lists from an HNSW index over the cloud. Insertion runs in
// parallel, so the index (and therefore the lists) can differ between runs;
// use knn_recall to measure the result.
inline KnnLists knn_lists_hnsw(const PointCloud& c, u32 k, const HnswParams& hp = {}) {
    KnnLists L;
    L.n = c.n;
    L.k = c.n ? std::min<u32>(k, c.n - 1) : 0;
    L.ids.resize(u64(L.n) * L.k);
    if (!L.k) return L;
    detail::MinkowskiSpace space(c.dim, c.metric);
    hnswlib::HierarchicalNSW<float> index(&space, c.n, hp.M, hp.ef_construction, hp.seed);
    index.addPoint(c.x, 0);
    parallel_for(c.n - 1, [&](u64 i) { index.addPoint(c.x + (i + 1) * c.dim, i + 1); });
    index.setEf(std::max<u32>(hp.ef_search, L.k + 1));
    parallel_for(c.n, [&](u64 i) {
        auto res = index.searchKnn(c.x + i * c.dim, L.k + 1);
        std::vector<detail::Cand> got;
        got.reserve(res.size());
        while (!res.empty()) {
            const u32 j = u32(res.top().second);
            if (j != i) got.emplace_back(c(u32(i), j), j);
            res.pop();
        }
        std::sort(got.begin(), got.end());
        u32* out = L.ids.data() + i * L.k;
        for (u32 t = 0; t < L.k; ++t) out[t] = t < got.size() ? got[t].second : u32(i);
    });
    return L;
}
#endif

// Symmetric kNN graph: (i, j) is an edge if j is in i's list or i in j's.
// Weights are recomputed with the cloud's metric.
inline CsrGraph knn_graph(const PointCloud& c, const KnnLists& L) {
    std::vector<std::vector<std::pair<u32, f32>>> rows(c.n);
    for (u32 i = 0; i < c.n; ++i)
        for (u32 t = 0; t < L.k; ++t) {
            const u32 j = L.row(i)[t];
            if (j == i) continue;
            rows[i].emplace_back(j, 0.0f);
            rows[j].emplace_back(i, 0.0f);
        }
    parallel_for(c.n, [&](u64 v) {
        auto& r = rows[v];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end(),
                            [](const auto& a, const auto& b) { return a.first == b.first; }),
                r.end());
        for (auto& e : r) e.second = c(u32(v), e.first);
    });
    return CsrGraph::from_rows(rows);
}

// Recall of approximate kNN lists against exact search on `samples` points
// drawn uniformly without replacement (all points if samples >= n).
// A neighbor counts as found when its distance is within the true k-th
// distance, so ties at the k-th distance are not penalized.
struct Recall {
    u64 checked_points = 0;
    u64 expected = 0, found = 0;
    u64 points_exact = 0;   // points whose list is fully correct
    double recall() const { return expected ? double(found) / double(expected) : 1.0; }
    // One-sided 95% lower bound on the per-point exactness rate (Wilson).
    double exact_rate_lower95() const {
        if (!checked_points) return 0.0;
        const double z = 1.6448536269514722, nn = double(checked_points);
        const double ph = double(points_exact) / nn;
        const double den = 1 + z * z / nn;
        const double cen = ph + z * z / (2 * nn);
        const double rad = z * std::sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn));
        return (cen - rad) / den;
    }
};

inline Recall knn_recall(const PointCloud& c, const KnnLists& L, u64 samples, u64 seed = 1) {
    Recall R;
    if (!L.k) return R;
    std::vector<u32> pts(c.n);
    for (u32 i = 0; i < c.n; ++i) pts[i] = i;
    if (samples < c.n) {
        std::mt19937_64 rng(seed);
        for (u64 t = 0; t < samples; ++t) {
            std::uniform_int_distribution<u64> U(t, c.n - 1);
            std::swap(pts[t], pts[U(rng)]);
        }
        pts.resize(samples);
    }
    const KdTree tree(c);
    std::vector<u64> found(pts.size(), 0);
    parallel_for(pts.size(), [&](u64 s) {
        thread_local std::vector<detail::Cand> heap;
        thread_local std::vector<u32> stack, truth;
        thread_local std::vector<f32> gap;
        truth.resize(L.k);
        const u32 i = pts[s];
        detail::knn_query(tree, i, L.k, heap, stack, gap, truth.data());
        const f32 kth = c(i, truth[L.k - 1]);
        u64 f = 0;
        for (u32 t = 0; t < L.k; ++t) {
            const u32 j = L.row(i)[t];
            if (j != i && c(i, j) <= kth) ++f;
        }
        found[s] = f;
    });
    R.checked_points = pts.size();
    R.expected = u64(pts.size()) * L.k;
    for (u64 f : found) {
        R.found += f;
        R.points_exact += f == L.k;
    }
    return R;
}

}  // namespace tops

#endif  // TURBOTOPS_GRAPH_HPP
