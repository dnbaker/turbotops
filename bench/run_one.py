#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one Python engine on one input; print 'seconds bars_per_dim' as JSON.

    run_one.py <engine> <input.npz> <maxdim> <threads>

engine: ripser-py | giotto-ph | giotto-ph-collapse | redzed
input.npz holds either 'points' (+ 'metric') for a full Rips complex, or a
sparse distance matrix ('data', 'row', 'col', 'n') for a flag complex. The
timed region is the library call only; reading the input is excluded.
"""
import json
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix


def main():
    engine, path, maxdim, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    z = np.load(path, allow_pickle=False)
    sparse = "data" in z.files
    if sparse:
        n = int(z["n"])
        X = coo_matrix((z["data"], (z["row"], z["col"])), shape=(n, n)).tocsr()
        kw = dict(metric="precomputed")
    else:
        X = z["points"].astype(np.float64)
        metric = str(z["metric"])
        kw = dict(metric=metric) if metric != "minkowski" else dict(metric="minkowski", metric_params={"p": float(z["p"])})
    if engine == "redzed":
        # RedZeD takes a dense float64 distance matrix; building it is not timed.
        import redzed_tda
        from scipy.spatial.distance import pdist
        thr = float(z["threshold"]) if "threshold" in z.files else float("inf")
        if sparse:
            raise SystemExit("redzed: sparse input unsupported")
        p = {"euclidean": 2.0, "manhattan": 1.0, "chebyshev": float("inf")}.get(kw["metric"])
        p = p if p is not None else kw["metric_params"]["p"]
        D = pdist(X, "chebyshev") if p == float("inf") else pdist(X, "minkowski", p=p)
        t0 = time.perf_counter()
        dgms = redzed_tda.rips(D, maxdim=maxdim, threshold=thr)
        secs = time.perf_counter() - t0
        bars = [int(np.sum(d[:, 1] > d[:, 0])) for d in dgms]
        print(json.dumps({"seconds": secs, "bars": bars}))
        return
    t0 = time.perf_counter()
    if engine == "ripser-py":
        import ripser
        if sparse:
            res = ripser.ripser(X, maxdim=maxdim, distance_matrix=True)
        elif kw["metric"] == "minkowski":
            res = ripser.ripser(X, maxdim=maxdim, metric="minkowski", p=kw["metric_params"]["p"])
        else:
            res = ripser.ripser(X, maxdim=maxdim, metric=kw["metric"])
        dgms = res["dgms"]
    else:
        from gph import ripser_parallel
        res = ripser_parallel(X, maxdim=maxdim, n_threads=threads,
                              collapse_edges=(engine == "giotto-ph-collapse"), **kw)
        dgms = res["dgms"]
    secs = time.perf_counter() - t0
    bars = [int(np.sum(d[:, 1] > d[:, 0])) for d in dgms]
    print(json.dumps({"seconds": secs, "bars": bars}))


if __name__ == "__main__":
    main()
