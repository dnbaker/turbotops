#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Python binding checks (no test framework needed).

    PYTHONPATH=python python3 tests/test_python.py

Compares turbo_tops against the turbotops CLI (built with `make`) on the
same input in every mode, and checks graph input and error handling.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

import turbo_tops as tt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "turbotops")


def cli_bars(path, args):
    out = subprocess.run([CLI, "run", path, "--quiet"] + args, check=True,
                         capture_output=True, text=True).stdout
    f = lambda v: float(np.float32(v))
    return sorted((int(l.split()[0]), f(l.split()[1]), f(l.split()[2])) for l in out.splitlines())


def py_bars(dgms):
    f = lambda v: float(np.float32(v))
    return sorted((d, f(b), f(e)) for d, a in enumerate(dgms) for b, e in a if e > b)


def main():
    fails = 0
    x = np.random.default_rng(0).normal(size=(600, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.txt")
        np.savetxt(path, x, fmt="%.9g", delimiter=",")
        for kw, args in [
            (dict(maxdim=2), ["--maxdim", "2"]),
            (dict(maxdim=1, metric="l1"), ["--maxdim", "1", "--metric", "l1"]),
            (dict(maxdim=1, metric="linf"), ["--maxdim", "1", "--metric", "linf"]),
            (dict(maxdim=1, metric="lp:1.5"), ["--maxdim", "1", "--metric", "lp:1.5"]),
            (dict(maxdim=2, graph="knn", k=10), ["--maxdim", "2", "--graph", "knn", "-k", "10"]),
            (dict(maxdim=1, threshold=0.7), ["--maxdim", "1", "--threshold", "0.7"]),
            (dict(maxdim=0), ["--maxdim", "0"]),
        ]:
            dgms, info = tt.persistence(x, **kw)
            ok = py_bars(dgms) == cli_bars(path, args) and len(dgms) == kw["maxdim"] + 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {kw} ({info['method']})")
            fails += not ok
    try:
        import scipy.sparse as sp
        from scipy.spatial import cKDTree
        t = cKDTree(x)
        g = t.sparse_distance_matrix(t, 0.7, output_type="coo_matrix").tocsr()
        a, _ = tt.persistence(g, maxdim=1)
        b, _ = tt.persistence(x, maxdim=1, threshold=0.7)
        # scipy computes distances in float64; compare bar counts and values to float32 tolerance
        ok = [len(d) for d in a] == [len(d) for d in b] and all(
            np.allclose(np.sort(p, axis=0), np.sort(q, axis=0), rtol=1e-5) for p, q in zip(a, b))
        print(f"  [{'PASS' if ok else 'FAIL'}] sparse graph input matches thresholded point input")
        fails += not ok
    except ImportError:
        print("  [SKIP] scipy not installed")
    for bad in (dict(metric="lp:0.5"), dict(graph="ring")):
        try:
            tt.persistence(x, **bad)
            print(f"  [FAIL] {bad} accepted")
            fails += 1
        except (ValueError, RuntimeError):
            print(f"  [PASS] {bad} rejected")
    print("FAILED" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
