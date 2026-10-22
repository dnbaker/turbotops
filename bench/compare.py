#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark turbotops against ripser (CLI and Python) and giotto-ph.

    python bench/compare.py [--quick] [--reps R] [--out bench/results.csv]

Requires numpy and scipy; ripser.py and giotto-ph are optional (missing
engines are skipped). Every engine receives the same complex:

  full     point cloud; each engine computes its own distances (full Rips)
  tau      tau-ball flag complex; Python engines and the ripser CLI get the
           sparse distance matrix of pairs with d <= tau, built with
           scipy's cKDTree (its build time is reported in prep_s and added
           to their total); turbotops builds its own radius graph from the
           points, inside its timing
  knn      flag complex of the symmetric exact k-NN graph, given to the other
           engines the same way

Timings are wall-clock seconds of one engine call (turbotops and the ripser
CLI: whole process, including reading the input). Peak RSS is measured per
process with getrusage. Bars of zero length are excluded from the counts;
every row must agree on the bar counts, which the script checks.
Run on an otherwise idle machine and report the load average.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TURBOTOPS = os.path.join(ROOT, "turbotops")
RIPSER_CLI = os.path.join(ROOT, "third_party", "ripser", "ripser")
NCPU = os.cpu_count() or 1


def cloud(kind, n, dim, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(size=(n, dim)).astype(np.float32)
    if kind == "torus":
        a, b = rng.uniform(0, 2 * np.pi, (2, n))
        return np.c_[(2 + np.cos(a)) * np.cos(b), (2 + np.cos(a)) * np.sin(b), np.sin(a)].astype(np.float32)
    if kind == "sphere":
        x = rng.normal(size=(n, dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    raise ValueError(kind)


def run_proc(argv):
    """Run a subprocess; return (wall seconds, stdout, peak RSS MB of that child)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.perf_counter()
        proc = subprocess.Popen(argv, stdout=out, stderr=err)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
        proc.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        if proc.returncode != 0:
            raise RuntimeError(f"{os.path.basename(argv[0])} exit {proc.returncode}: {err.read()[-300:]}")
        scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 1024
        return wall, out.read(), ru.ru_maxrss * scale


def bars_from_turbotops(text, maxdim):
    counts = [0] * (maxdim + 1)
    for line in text.splitlines():
        d, b, e = line.split()
        if e == "inf" or float(e) > float(b):
            counts[int(d)] += 1
    return counts


def bars_from_ripser_cli(text, maxdim):
    counts, d = [0] * (maxdim + 1), None
    for line in text.splitlines():
        if line.startswith("persistence intervals in dim"):
            d = int(line.split()[-1].rstrip(":"))
        elif d is not None and line.strip().startswith("["):
            b, e = line.strip()[1:-1].split(",")
            if e.strip() in ("", " ") or float(e) > float(b):
                counts[d] += 1
    return counts


def sparse_pairs(X, metric_p, mode, tau=None, k=None):
    """Upper-triangular (i, j, d) arrays of the complex's edges."""
    tree = cKDTree(X.astype(np.float64))
    if mode == "tau":
        M = tree.sparse_distance_matrix(tree, tau, p=metric_p, output_type="coo_matrix")
        keep = M.row < M.col
        return M.row[keep], M.col[keep], M.data[keep].astype(np.float32)
    d, idx = tree.query(X.astype(np.float64), k=k + 1, p=metric_p)
    rows = np.repeat(np.arange(len(X)), k)
    cols = idx[:, 1:].ravel()
    i, j = np.minimum(rows, cols), np.maximum(rows, cols)
    key = np.unique(i.astype(np.int64) * len(X) + j)
    i, j = key // len(X), key % len(X)
    diff = X[i].astype(np.float64) - X[j]
    dist = np.linalg.norm(diff, ord=metric_p, axis=1)
    return i, j, dist.astype(np.float32)


def p_of(metric):
    return {"l1": 1, "l2": 2, "linf": np.inf}.get(metric) or float(metric.split(":")[1])


def py_metric(metric):
    return {"l1": "manhattan", "l2": "euclidean", "linf": "chebyshev"}.get(metric, "minkowski")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes only")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    ap.add_argument("--python", default=sys.executable, help="interpreter with ripser/giotto-ph")
    ap.add_argument("--engines", default="", help="comma-separated subset of engine names to run")
    ap.add_argument("--cases", default="", help="comma-separated case indices to run")
    args = ap.parse_args()

    have = {}
    for mod, name in (("ripser", "ripser-py"), ("gph", "giotto-ph"), ("redzed_tda", "redzed")):
        have[name] = subprocess.run([args.python, "-c", f"import {mod}"], capture_output=True).returncode == 0
    have["ripser-cli"] = os.access(RIPSER_CLI, os.X_OK)

    q = args.quick
    cases = [
        # (regime, cloud, n, dim, maxdim, metric, tau, k)
        ("full", "uniform", 2000 if q else 5000, 3, 1, "l2", None, None),
        ("full", "uniform", 3000 if q else 10000, 3, 1, "l2", None, None),
        ("full", "torus", 400 if q else 1200, 3, 2, "l2", None, None),
        ("full", "uniform", 2000 if q else 5000, 3, 1, "l1", None, None),
        ("full", "uniform", 2000 if q else 5000, 3, 1, "linf", None, None),
        ("full", "uniform", 2000 if q else 5000, 3, 1, "lp:3", None, None),
        ("tau", "uniform", 10000, 3, 1, "l2", 0.164, None),
        ("tau", "uniform", 25000, 3, 1, "l2", 0.128, None),
        ("tau", "uniform", 50000, 3, 1, "l2", 0.105, None),
        ("tau", "uniform", 10000, 3, 2, "l2", 0.12, None),
        ("knn", "uniform", 25000 if q else 100000, 3, 1, "l2", None, 15),
        ("knn", "uniform", 25000 if q else 100000, 3, 2, "l2", None, 15),
    ]
    if q:
        cases = [c for c in cases if c[0] != "tau" or c[2] <= 10000]

    if args.cases:
        keep = {int(c) for c in args.cases.split(",")}
        cases = [c for i, c in enumerate(cases) if i in keep]
    wanted = set(args.engines.split(",")) if args.engines else None
    tmp = tempfile.mkdtemp(prefix="ttops_bench_")
    rows = []
    load0 = os.getloadavg()[0]
    for (regime, kind, n, dim, maxdim, metric, tau, k) in cases:
        X = cloud(kind, n, dim, seed=1)
        txt = os.path.join(tmp, "x.txt")
        np.savetxt(txt, X, fmt="%.9g", delimiter=",")
        p = p_of(metric)
        prep_s = 0.0
        npz = os.path.join(tmp, "in.npz")
        sparse_txt = os.path.join(tmp, "edges.txt")
        if regime == "full":
            extra = dict(metric=py_metric(metric), p=np.float64(p if np.isfinite(p) else 0))
            np.savez(npz, points=X, **extra)
        else:
            t0 = time.perf_counter()
            i, j, d = sparse_pairs(X, p, regime, tau=tau, k=k)
            prep_s = time.perf_counter() - t0
            np.savez(npz, data=np.r_[d, d], row=np.r_[i, j], col=np.r_[j, i], n=np.int64(n))
            np.savetxt(sparse_txt, np.c_[i, j, d], fmt=["%d", "%d", "%.9g"], delimiter=" ")
        label = f"{regime} {kind} n={n} H{maxdim} {metric}" + (f" tau={tau}" if tau else "") + (f" k={k}" if k else "")

        engines = []
        targs = [TURBOTOPS, "run", txt, "--maxdim", str(maxdim), "--metric", metric, "--quiet"]
        if regime == "tau":
            targs += ["--threshold", str(tau)]
        if regime == "knn":
            targs += ["--graph", "knn", "-k", str(k)]
        for th in (1, NCPU):
            engines.append((f"turbotops t{th}", 0.0,
                            lambda th=th: (lambda w, o, r: (w, bars_from_turbotops(o, maxdim), r))(
                                *run_proc(targs + ["--threads", str(th)]))))
        if have["ripser-cli"] and (regime != "full" or metric == "l2"):
            if regime == "full":
                rargs = [RIPSER_CLI, "--format", "point-cloud", "--dim", str(maxdim), txt]
            else:
                rargs = [RIPSER_CLI, "--format", "sparse", "--dim", str(maxdim), sparse_txt]
            engines.append(("ripser-cli", prep_s,
                            lambda rargs=rargs: (lambda w, o, r: (w, bars_from_ripser_cli(o, maxdim), r))(*run_proc(rargs))))
        for name, th in (("ripser-py", 1), ("giotto-ph", 1), ("giotto-ph", NCPU), ("giotto-ph-collapse", NCPU)):
            base = "giotto-ph" if name.startswith("giotto") else name
            if not have[base]:
                continue
            # Edge collapse targets sparse inputs; on complete graphs it is
            # orders of magnitude slower, so it is only run on tau/knn.
            if name == "giotto-ph-collapse" and regime == "full":
                continue
            # ripser.py exposes no Minkowski exponent.
            if name == "ripser-py" and metric.startswith("lp:"):
                continue
            def go(name=name, th=th):
                w, o, r = run_proc([args.python, os.path.join(HERE, "run_one.py"), name, npz, str(maxdim), str(th)])
                res = json.loads(o)
                return res["seconds"], res["bars"], r
            engines.append((f"{name} t{th}" if name != "ripser-py" else name, prep_s, go))

        # RedZeD reads a dense distance matrix (float64, n^2): full Rips, and
        # thresholded Rips while the matrix stays small.
        if have["redzed"] and (regime == "full" or (regime == "tau" and n <= 10000)):
            rz_npz = os.path.join(tmp, "rz.npz")
            np.savez(rz_npz, points=X, metric=py_metric(metric), p=np.float64(p if np.isfinite(p) else 0),
                     threshold=np.float64(tau if tau else np.inf))
            def go_rz():
                w, o, r = run_proc([args.python, os.path.join(HERE, "run_one.py"), "redzed", rz_npz, str(maxdim), "1"])
                res = json.loads(o)
                return res["seconds"], res["bars"], r
            engines.append(("redzed", 0.0, go_rz))
        if wanted:
            engines = [e for e in engines if e[0] in wanted]
        results = {e[0]: [] for e in engines}
        info = {}
        failed = {}
        for rep in range(args.reps):
            for name, prep, fn in engines:
                if name in failed:
                    continue
                try:
                    secs, bars, rss = fn()
                except Exception as exc:   # record and keep going
                    failed[name] = str(exc).replace("\n", " ")[:200]
                    continue
                results[name].append(secs + prep)
                info[name] = (bars, rss, prep)
        ref = info[engines[0][0]][0]
        for name, _, _ in engines:
            if name in failed:
                rows.append(dict(case=label, engine=name, seconds="", prep_s="", peak_rss_mb="",
                                 bars="", parity="ERROR: " + failed[name]))
                print(f"{label:40s} {name:24s} ERROR {failed[name]}", flush=True)
                continue
            bars, rss, prep = info[name]
            # ripser.py reports the essential H0 class; all engines count it.
            ok = list(bars[: maxdim + 1]) == list(ref[: maxdim + 1])
            row = dict(case=label, engine=name, seconds=round(float(np.median(results[name])), 3),
                       prep_s=round(prep, 3), peak_rss_mb=round(rss, 1) if rss == rss else "",
                       bars=" ".join(map(str, bars[: maxdim + 1])), parity="ok" if ok else "MISMATCH")
            rows.append(row)
            print(f"{label:40s} {name:24s} {row['seconds']:9.3f}s  rss={row['peak_rss_mb']}  bars={row['bars']}  {row['parity']}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"load average at start {load0:.2f}, end {os.getloadavg()[0]:.2f}; {NCPU} cpus -> {args.out}")


if __name__ == "__main__":
    main()
