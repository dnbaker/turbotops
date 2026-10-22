// SPDX-License-Identifier: MIT
// mst.hpp — degree-0 persistence of the Rips filtration of a point cloud in
// O(n) memory.
//
// The finite H0 deaths are the edge weights of a minimum spanning tree of the
// complete graph (Prim's algorithm: O(n^2) distance evaluations, O(n)
// memory). Each step scans the remaining vertices in parallel with a fixed
// pool of threads. MST edges longer than the threshold are cut, each cut
// adding one essential class.
#ifndef TURBOTOPS_MST_HPP
#define TURBOTOPS_MST_HPP

#include <algorithm>
#include <barrier>
#include <cmath>
#include <thread>
#include <vector>

#include "core.hpp"
#include "metric.hpp"
#include "rips.hpp"

namespace tops::mst {

inline rips::Barcode h0(const PointCloud& c, f32 threshold = INFINITY) {
    for (u64 i = 0; i < u64(c.n) * c.dim; ++i)
        if (!std::isfinite(c.x[i]))
            throw std::invalid_argument("mst: non-finite coordinate");
    rips::Barcode out;
    out.ensure(0);
    if (c.n == 0) return out;
    const u32 n = c.n;
    std::vector<f32> best(n, INFINITY);
    std::vector<u8> in(n, 0);
    std::vector<f32> deaths;
    deaths.reserve(n - 1);
    const u32 nt = u32(std::min<u64>(num_threads(), n));
    using Cand = std::pair<f32, u32>;
    std::vector<Cand> local(nt, Cand{INFINITY, ~0u});
    u32 cur = 0, step = 1;
    in[0] = 1;

    auto on_step = [&]() noexcept {
        Cand m{INFINITY, ~0u};
        for (const auto& l : local) m = std::min(m, l);
        deaths.push_back(m.first);
        in[m.second] = 1;
        cur = m.second;
        ++step;
    };
    std::barrier sync(nt, on_step);
    auto worker = [&](u32 t) {
        const u64 lo = u64(n) * t / nt, hi = u64(n) * (t + 1) / nt;
        // 3-D Euclidean clouds (the common spatial case) get their distance
        // row in vector batches; the values are bit-identical to c(cur, j).
        const bool batched = c.dim == 3 && c.metric.kind == Metric::L2;
        alignas(32) f32 db[256];
        while (step < n) {
            Cand m{INFINITY, ~0u};
            if (batched) {
                for (u64 j0 = lo; j0 < hi; j0 += 256) {
                    const u64 len = std::min<u64>(256, hi - j0);
                    kern::l2_row3(c.x + u64(cur) * 3, c.x + j0 * 3, len, db);
                    for (u64 t2 = 0; t2 < len; ++t2) {
                        const u64 j = j0 + t2;
                        if (in[j]) continue;
                        const f32 d = db[t2];
                        if (d < best[j]) best[j] = d;
                        if (Cand{best[j], u32(j)} < m) m = {best[j], u32(j)};
                    }
                }
            } else {
                for (u64 j = lo; j < hi; ++j) {
                    if (in[j]) continue;
                    const f32 d = c(cur, u32(j));
                    if (d < best[j]) best[j] = d;
                    if (Cand{best[j], u32(j)} < m) m = {best[j], u32(j)};
                }
            }
            local[t] = m;
            sync.arrive_and_wait();
        }
    };
    if (n > 1) {
        std::vector<std::thread> pool;
        for (u32 t = 1; t < nt; ++t) pool.emplace_back(worker, t);
        worker(0);
        for (auto& th : pool) th.join();
    }

    std::sort(deaths.begin(), deaths.end());
    u64 essential = 1;
    for (f32 d : deaths) {
        if (d > threshold) ++essential;
        else if (d > 0.0f) out.add(0, 0.0f, d);
    }
    for (u64 e = 0; e < essential; ++e) out.add(0, 0.0f, INFINITY);
    return out;
}

}  // namespace tops::mst

#endif  // TURBOTOPS_MST_HPP
