# SPDX-License-Identifier: MIT
"""turbo_tops — Vietoris–Rips persistent homology.

    >>> import numpy as np, turbo_tops as tt
    >>> x = np.random.default_rng(0).normal(size=(2000, 3)).astype(np.float32)
    >>> dgms, info = tt.persistence(x, maxdim=2)
    >>> dgms[1]          # (m, 2) array of (birth, death); death = inf if essential

Inputs: a point cloud (n, d) with a Minkowski metric, or a sparse weighted
graph (any scipy.sparse matrix; the flag complex of its edges is used).
"""
import math

import numpy as np

from . import _turbotops

__all__ = ["persistence", "__version__"]

__version__ = _turbotops.__version__


def persistence(x, maxdim=1, threshold=math.inf, metric="l2", graph="full", k=15,
                approximate=False, recall_samples=0, threads=0, memory_budget=0):
    """Persistence diagrams of the Vietoris–Rips filtration.

    Parameters
    ----------
    x : array (n, d) or scipy.sparse matrix (n, n)
        Point cloud, or symmetric edge weights of a graph.
    maxdim : int
        Highest homology dimension computed.
    threshold : float
        Filtration cutoff (inclusive). With graph="full" and threshold=inf the
        filtration stops at the enclosing radius, which changes no bars.
    metric : str
        "l1", "l2", "linf" or "lp:<p>" with p >= 1 (point clouds only).
    graph : {"full", "knn"}
        All pairs, or the flag complex of the symmetric k-nearest-neighbor graph.
    k : int
        Neighbors for graph="knn".
    approximate : bool
        Use HNSW for the k-NN graph instead of exact search.
    recall_samples : int
        With approximate=True, measure k-NN recall against exact search on
        this many points (reported in the returned info).
    threads : int
        Worker threads (0 = all cores). Results do not depend on it.
    memory_budget : int
        Bytes allowed for the dense distance matrix (0 = half of RAM).

    Returns
    -------
    dgms : list of (m, 2) float32 arrays, one per dimension 0..maxdim
    info : dict with method, effective threshold, edge count, timings,
           peak RSS and, if requested, k-NN recall
    """
    if hasattr(x, "tocsr"):
        g = x.tocsr()
        g.sort_indices()
        return _turbotops.persistence_csr(
            np.ascontiguousarray(g.indptr, dtype=np.int64),
            np.ascontiguousarray(g.indices, dtype=np.int64),
            np.ascontiguousarray(g.data, dtype=np.float32),
            int(g.shape[0]), int(maxdim), float(threshold), int(threads))
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("x must be a 2-d array (n, d)")
    return _turbotops.persistence_points(
        x, int(maxdim), float(threshold), str(metric), str(graph), int(k),
        bool(approximate), int(recall_samples), int(threads), int(memory_budget))
