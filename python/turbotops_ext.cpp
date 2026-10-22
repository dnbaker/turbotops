// SPDX-License-Identifier: MIT
// Python bindings (nanobind). Exposed through the turbo_tops package.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "turbotops/turbotops.hpp"

namespace nb = nanobind;
using namespace tops;

namespace {

using Points = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Array2 = nb::ndarray<nb::numpy, float, nb::ndim<2>>;

// Bars of each dimension as an (m, 2) float32 array; essential classes have
// death = inf.
nb::list to_python(const Barcode& bc) {
    nb::list out;
    for (u32 d = 0; d < bc.ndims(); ++d) {
        const u64 m = bc.birth[d].size();
        auto* buf = new float[2 * m];
        for (u64 i = 0; i < m; ++i) {
            buf[2 * i] = bc.birth[d][i];
            buf[2 * i + 1] = bc.death[d][i];
        }
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        out.append(Array2(buf, {size_t(m), size_t(2)}, owner));
    }
    return out;
}

nb::dict stats_dict(const Stats& s) {
    nb::dict d;
    d["method"] = s.method;
    d["threshold"] = s.threshold;
    d["edges"] = s.edges;
    d["index_bits"] = s.wide_index ? 128 : 64;
    d["seconds_graph"] = s.seconds_graph;
    d["seconds_total"] = s.seconds_total;
    d["peak_rss_mb"] = s.peak_rss_mb;
    if (s.recall.checked_points) {
        d["recall"] = s.recall.recall();
        d["recall_points"] = s.recall.checked_points;
        d["recall_exact_points"] = s.recall.points_exact;
        d["recall_exact_rate_lower95"] = s.recall.exact_rate_lower95();
    }
    return d;
}

nb::tuple persistence_points(Points x, u32 maxdim, float threshold, const std::string& metric,
                             const std::string& graph, u32 k, bool approximate,
                             u64 recall_samples, u32 threads, u64 memory_budget) {
    Options o;
    o.maxdim = maxdim;
    o.threshold = threshold;
    o.metric = metric;
    if (graph == "full") o.graph = Options::Graph::Full;
    else if (graph == "knn") o.graph = Options::Graph::Knn;
    else throw nb::value_error("graph must be 'full' or 'knn'");
    o.k = k;
    o.approximate = approximate;
    o.recall_samples = recall_samples;
    o.threads = threads;
    o.memory_budget = memory_budget;
    Result r;
    {
        nb::gil_scoped_release release;
        r = persistence(x.data(), u32(x.shape(0)), u32(x.shape(1)), o);
    }
    return nb::make_tuple(to_python(r.barcode), stats_dict(r.stats));
}

nb::tuple persistence_csr(nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> indptr,
                          nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig> indices,
                          nb::ndarray<const float, nb::ndim<1>, nb::c_contig> data,
                          u32 n, u32 maxdim, float threshold, u32 threads) {
    if (indptr.shape(0) != size_t(n) + 1 || indices.shape(0) != data.shape(0))
        throw nb::value_error("inconsistent CSR arrays");
    std::vector<std::vector<std::pair<u32, f32>>> rows(n);
    for (u32 i = 0; i < n; ++i)
        for (int64_t t = indptr.data()[i]; t < indptr.data()[i + 1]; ++t) {
            const int64_t j = indices.data()[t];
            const float w = data.data()[t];
            if (j < 0 || j >= int64_t(n)) throw nb::value_error("column index out of range");
            if (!(w >= 0) || !std::isfinite(w)) throw nb::value_error("weights must be finite and >= 0");
            if (j == i) continue;
            rows[i].emplace_back(u32(j), w);
            rows[u32(j)].emplace_back(i, w);
        }
    for (auto& r : rows) {   // keep the smallest weight of repeated entries
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end(),
                            [](const auto& a, const auto& b) { return a.first == b.first; }),
                r.end());
    }
    const CsrGraph g = CsrGraph::from_rows(rows);
    Result r;
    {
        nb::gil_scoped_release release;
        r = persistence(g, maxdim, threshold, threads);
    }
    return nb::make_tuple(to_python(r.barcode), stats_dict(r.stats));
}

}  // namespace

NB_MODULE(_turbotops, m) {
    m.doc() = "Vietoris-Rips persistent homology";
    m.attr("__version__") = TURBOTOPS_VERSION_STRING;
    m.def("persistence_points", &persistence_points, nb::arg("x"), nb::arg("maxdim"),
          nb::arg("threshold"), nb::arg("metric"), nb::arg("graph"), nb::arg("k"),
          nb::arg("approximate"), nb::arg("recall_samples"), nb::arg("threads"),
          nb::arg("memory_budget"));
    m.def("persistence_csr", &persistence_csr, nb::arg("indptr"), nb::arg("indices"),
          nb::arg("data"), nb::arg("n"), nb::arg("maxdim"), nb::arg("threshold"),
          nb::arg("threads"));
}
