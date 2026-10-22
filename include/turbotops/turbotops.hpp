// SPDX-License-Identifier: MIT
// turbotops.hpp — persistent homology of Vietoris–Rips filtrations.
//
// Single entry point for point clouds and for weighted graphs:
//
//   tops::Options opt;            // maxdim, threshold, metric, graph, ...
//   tops::Result r = tops::persistence(points, n, dim, opt);
//   r.barcode.birth[d], r.barcode.death[d]   // bars of dimension d
//
// Dispatch:
//   graph = full, threshold = inf   dense distance matrix, filtration cut at
//                                   the enclosing radius (beyond which the
//                                   complex is a cone)
//   graph = full, threshold < inf   sparse radius graph (exact)
//   graph = knn                     flag complex of the symmetric kNN graph
//                                   (exact kNN, or HNSW when approximate)
//   maxdim = 0, graph = full        minimum spanning tree in O(n) memory
//
// Mode (dimensions >= 1): the coboundary engine is the cohomology reducer
// (rips.hpp, multithreaded, 64- or 128-bit indices, maxdim <= 30, any input)
// and is ON BY DEFAULT; the boundary engine is RedZeD (redzed.hpp,
// single-threaded, 64-bit simplex indices only, n < 2^31). Both give
// byte-identical diagrams wherever both can run; detail::select_mode resolves
// Mode::{Auto, Boundary, Coboundary} to an engine plus a human-readable
// reason.
//
// All filtrations are inclusive: simplices with diameter <= threshold are
// kept, and a simplex enters at its diameter.
#ifndef TURBOTOPS_TURBOTOPS_HPP
#define TURBOTOPS_TURBOTOPS_HPP

#include <chrono>
#include <stdexcept>
#include <string>

#include "core.hpp"
#include "graph.hpp"
#include "memory.hpp"
#include "metric.hpp"
#include "mst.hpp"
#include "redzed.hpp"
#include "rips.hpp"

namespace tops {

using rips::Barcode;

enum class Engine { Auto, Reducer, RedZeD };

// User-facing reduction mode. Coboundary = cohomology reducer (general path,
// default on); Boundary = RedZeD (faster where it applies, capability-gated);
// Auto = measured rule, default-on coboundary within noise (see select_mode).
enum class Mode { Auto, Boundary, Coboundary };

inline Mode parse_mode(const std::string& s) {
    if (s == "auto") return Mode::Auto;
    if (s == "boundary") return Mode::Boundary;
    if (s == "coboundary") return Mode::Coboundary;
    throw std::invalid_argument("mode must be auto, boundary or coboundary");
}

inline Engine parse_engine(const std::string& s) {
    if (s == "auto") return Engine::Auto;
    if (s == "reducer") return Engine::Reducer;
    if (s == "redzed") return Engine::RedZeD;
    throw std::invalid_argument("engine must be auto, reducer or redzed");
}

struct Options {
    u32 maxdim = 1;
    Mode mode = Mode::Auto;      // boundary | coboundary | auto (default)
    f32 threshold = INFINITY;
    std::string metric = "l2";   // l1 | l2 | linf | lp:<p>
    enum class Graph { Full, Knn } graph = Graph::Full;
    u32 k = 15;                  // neighbors for Graph::Knn
    bool approximate = false;    // Graph::Knn via HNSW instead of exact search
#ifndef TURBOTOPS_NO_HNSW
    HnswParams hnsw;
#endif
    u64 recall_samples = 0;      // > 0: measure HNSW recall on this many points
    u64 memory_budget = 0;       // bytes; 0 = half of physical memory
    u32 threads = 0;             // 0 = hardware concurrency
};

struct Stats {
    std::string method;          // "dense", "sparse" or "mst"
    std::string engine;          // "coboundary" or "boundary"
    std::string mode_reason;     // why select_mode chose this engine
    f32 threshold = INFINITY;    // effective filtration cutoff
    u64 edges = 0;
    bool wide_index = false;
    double seconds_graph = 0, seconds_total = 0;
    double peak_rss_mb = 0;
    Recall recall;               // filled when recall_samples > 0
    std::vector<u64> columns, apparent, emergent;  // coboundary-engine telemetry
};

struct Result {
    Barcode barcode;
    Stats stats;
};

namespace detail {
// Auto-selector rule, measured with bench/mode_grid.py (29 cells x 3 modes x
// 3 interleaved reps, medians). Boundary (RedZeD) wins only for H1 on dense
// full-Rips input and on graphs of mean degree >= kBoundaryMinMeanDegree;
// for H >= 2 and on sparser graphs the coboundary reducer wins, and where the
// two are within timing noise the default-on philosophy prefers coboundary.
//
// The crossover bracket from the fill cells (uniform kNN H1, n = 10k/30k):
// coboundary wins at mean degree 13.8/13.7, boundary wins at 18.2/18.1 and
// stays ahead through degree 31 and beyond; dense H1 is boundary at every n.
// 16 is the bracket midpoint: margins near the crossover are 1.2-1.6x on
// either side, so anything in [14, 18] is defensible and a mis-pick there
// costs at most ~1.3x.
inline constexpr double kBoundaryMinMeanDegree = 16.0;

// Resolves the requested mode to a concrete engine. mean_degree is 2m/n for
// graph input and INFINITY for dense full-Rips input. A forced mode that the
// problem does not support is an error, never a silent switch.
inline Engine select_mode(Mode mode, u32 n, u32 maxdim, double mean_degree, std::string& why) {
    if (mode == Mode::Coboundary) {
        why = "coboundary (--mode coboundary): cohomology reducer, maxdim <= 30, "
              "64/128-bit indices, any input";
        return Engine::Reducer;
    }
    if (mode == Mode::Boundary) {
        if (maxdim == 0) {
            why = "boundary (--mode boundary) requested, but H0 runs as a minimum "
                  "spanning tree and uses no reduction engine";
            return Engine::Reducer;
        }
        if (!redzed::fits(n, maxdim))
            throw std::invalid_argument(
                "boundary mode (RedZeD) needs every simplex index in 64 bits "
                "(C(n, maxdim + 2) < 2^64, implied n < 2^31); this problem does "
                "not fit — use the default coboundary mode instead");
        why = "boundary (--mode boundary): RedZeD, reduction to zero differentials";
        return Engine::RedZeD;
    }
    if (maxdim == 0) {
        why = "auto: H0 only, minimum spanning tree (no reduction engine)";
        return Engine::Reducer;
    }
    const bool fits = redzed::fits(n, maxdim);
    if (fits && maxdim == 1 &&
        (std::isinf(mean_degree) || mean_degree >= kBoundaryMinMeanDegree)) {
        why = std::isinf(mean_degree)
                  ? "auto: boundary (RedZeD) — measured fastest for H1 on dense "
                    "full-Rips input"
                  : "auto: boundary (RedZeD) — measured fastest for H1 on graphs "
                    "of mean degree >= " +
                        std::to_string(int(kBoundaryMinMeanDegree));
        return Engine::RedZeD;
    }
    why = fits ? "auto: coboundary (reducer) — default-on; boundary is picked "
                 "only for H1 on dense / mean degree >= " +
                     std::to_string(int(kBoundaryMinMeanDegree)) + " inputs"
               : "auto: coboundary (reducer) — boundary (RedZeD) cannot index "
                 "this problem within 64-bit simplex indices";
    return Engine::Reducer;
}
inline double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace detail

namespace detail {
// Validates the filtration parameters shared by both entry points.
inline void validate_options(f32 threshold, u32 k) {
    if (std::isnan(threshold) || threshold < 0.0f)
        throw std::invalid_argument(
            "persistence: threshold must be non-negative or +inf");
    if (k == 0) throw std::invalid_argument("persistence: k must be >= 1");
}
}  // namespace detail

// Flag complex of a weighted graph (CsrGraph::valid() must hold).
inline Result persistence(const CsrGraph& g, u32 maxdim, f32 threshold = INFINITY,
                          u32 threads = 0, Mode mode = Mode::Auto) {
    const auto t0 = std::chrono::steady_clock::now();
    detail::validate_options(threshold, 1);
    if (!g.valid())
        throw std::invalid_argument(
            "persistence: CsrGraph invariant violated (see CsrGraph::valid)");
    if (threads) set_num_threads(threads);
    Result r;
    rips::RipsStats rs;
    const Engine e = detail::select_mode(mode, g.n, maxdim,
                                         g.n ? 2.0 * double(g.m) / g.n : 0.0,
                                         r.stats.mode_reason);
    r.stats.engine = e == Engine::RedZeD ? "boundary" : "coboundary";
    if (e == Engine::RedZeD) {
        r.barcode = redzed::compute_graph(g, maxdim, threshold, &rs.edges);
    } else {
        r.barcode = rips::compute_graph(g, maxdim, threshold, &rs);
    }
    r.barcode.ensure(maxdim);
    r.stats.method = e == Engine::RedZeD ? "sparse-redzed" : "sparse";
    r.stats.threshold = threshold;
    r.stats.edges = rs.edges;
    r.stats.wide_index = rs.wide_index;
    r.stats.columns = rs.columns;
    r.stats.apparent = rs.apparent;
    r.stats.emergent = rs.emergent;
    r.stats.seconds_total = detail::since(t0);
    r.stats.peak_rss_mb = peak_rss_mb();
    return r;
}

inline Result persistence(const f32* points, u32 n, u32 dim, const Options& o) {
    const auto t0 = std::chrono::steady_clock::now();
    detail::validate_options(o.threshold, o.graph == Options::Graph::Knn ? o.k : 1);
    if (o.threads) set_num_threads(o.threads);
    const u64 budget = o.memory_budget ? o.memory_budget : default_memory_budget();
    const PointCloud cloud{points, n, dim, Metric::parse(o.metric)};
    for (u64 i = 0; i < u64(n) * dim; ++i)
        if (!std::isfinite(points[i]))
            throw std::invalid_argument("persistence: non-finite coordinate");
    Result r;
    Stats& st = r.stats;

    if (o.graph == Options::Graph::Full && o.maxdim == 0) {
        r.barcode = mst::h0(cloud, o.threshold);
        st.method = "mst";
        st.engine = "mst";
        st.mode_reason = "H0 only: minimum spanning tree (no reduction engine)";
        st.threshold = o.threshold;
        st.edges = n ? n - 1 : 0;
    } else if (o.graph == Options::Graph::Full && std::isinf(o.threshold)) {
        rips::RipsStats rs;
        const Engine e = detail::select_mode(o.mode, n, o.maxdim, INFINITY, st.mode_reason);
        st.engine = e == Engine::RedZeD ? "boundary" : "coboundary";
        if (e == Engine::RedZeD) {
            // Relabel the points in k-d tree order (the diagram does not depend
            // on labels): neighbours get nearby ids, so RedZeD's bitset rows
            // and distance reads stay local.
            std::vector<f32> y(u64(n) * dim);
            {
                const KdTree t(cloud);
                for (u32 i = 0; i < n; ++i)
                    std::copy_n(points + u64(t.perm[i]) * dim, dim, y.data() + u64(i) * dim);
            }
            const PointCloud local{y.data(), n, dim, cloud.metric};
            rips::DenseComplex cx = rips::make_dense(n, local, budget);
            cx.threshold = cx.enclosing_radius();
            rs.threshold = cx.threshold;
            r.barcode = redzed::compute(cx, o.maxdim, &rs.edges);
            st.method = "dense-redzed";
        } else {
            r.barcode = rips::compute_dense(n, cloud, INFINITY, o.maxdim, &rs, false, budget);
            st.method = "dense";
        }
        st.threshold = rs.threshold;
        st.edges = rs.edges;
        st.wide_index = rs.wide_index;
        st.columns = rs.columns;
        st.apparent = rs.apparent;
        st.emergent = rs.emergent;
    } else {
        const auto tg = std::chrono::steady_clock::now();
        CsrGraph g;
        if (o.graph == Options::Graph::Full) {
            g = radius_graph(cloud, o.threshold);
        } else {
            KnnLists L;
            if (o.approximate) {
#ifndef TURBOTOPS_NO_HNSW
                L = knn_lists_hnsw(cloud, o.k, o.hnsw);
                if (o.recall_samples) st.recall = knn_recall(cloud, L, o.recall_samples);
#else
                throw std::invalid_argument("persistence: built without HNSW support");
#endif
            } else {
                L = knn_lists_exact(cloud, o.k);
            }
            g = knn_graph(cloud, L);
        }
        st.seconds_graph = detail::since(tg);
        rips::RipsStats rs;
        const Engine e = detail::select_mode(o.mode, n, o.maxdim,
                                             n ? 2.0 * double(g.m) / n : 0.0,
                                             st.mode_reason);
        st.engine = e == Engine::RedZeD ? "boundary" : "coboundary";
        if (e == Engine::RedZeD) {
            r.barcode = redzed::compute_graph(g, o.maxdim, o.threshold, &rs.edges);
            st.method = "sparse-redzed";
        } else {
            r.barcode = rips::compute_graph(g, o.maxdim, o.threshold, &rs);
            st.method = "sparse";
        }
        st.threshold = o.threshold;
        st.edges = rs.edges;
        st.wide_index = rs.wide_index;
        st.columns = rs.columns;
        st.apparent = rs.apparent;
        st.emergent = rs.emergent;
    }
    r.barcode.ensure(o.maxdim);
    st.seconds_total = detail::since(t0);
    st.peak_rss_mb = peak_rss_mb();
    return r;
}

}  // namespace tops

#endif  // TURBOTOPS_TURBOTOPS_HPP
