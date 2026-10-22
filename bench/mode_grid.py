#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Mode-comparison grid for the boundary|coboundary auto-selector.

    python bench/mode_grid.py [--quick] [--reps N] [--threads N]
                              [--out bench/results_mode_grid.csv]

Runs `./turbotops run` in BOTH reduction modes (--mode coboundary, --mode
boundary) plus the default (--mode auto), REPS reps per cell, interleaved
(rep-major: each rep runs coboundary, boundary, auto), and reports median
total time with [min, max] spread. Grid coverage:

  dense   full Rips on uniform clouds (mean degree is effectively n - 1)
  knn     flag complex of the symmetric exact k-NN graph, mean degree ~ 2k
  radius  tau-ball flag complex, tau calibrated to a target mean degree
  family  uniform | torus (S^1 x S^1 in R^3) | gaussmix (8 tight clusters)
  dim     3 except the H3 sphere cells (S^3-sampled 4-sphere... unit S^3 in R^4)

Full grid: dense uniform n in {2k, 5k, 10k} at H1 and n = 2k at H2; kNN
(uniform) n in {5k (k 8/32/128), 20k (k 8/32), 50k (k 8/32)} at H1 and
n = 5k (k 8/32) at H2; crossover fill (uniform kNN, H1) at mean degree
~12/16/24/32 for n in {10k, 30k} (k 12/16/22/28, landing at measured mean
degree ~14/18/25/32); radius n = 10k at mean
degree ~8/~128, H1; families at moderate n: torus n = 2k H1+H2, gaussmix
(8 components, sd 0.08) n = 5k H1 and n = 1.5k H2; H3 spot checks: sphere
n = 120 (dim 4, dense) and kNN n = 3k k = 8 (sparse).

Per run the harness records wall time, the binary's own total, peak RSS,
edge count, mean degree (2m/n), and the coboundary engine's column/apparent/
emergent telemetry. THE PARITY INVARIANTS: (1) every stdout diagram in a
cell must be byte-identical across modes and reps -- mode selection changes
speed, never output; (2) auto's engine must be the one with the better
median on that cell. Both are asserted loudly. Run on an otherwise idle
machine; --threads defaults to 8 to leave headroom.
"""
import argparse
import csv
import hashlib
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TURBOTOPS = os.path.join(ROOT, "turbotops")

SUMMARY = re.compile(
    r"method=\S+ threshold=\S+ edges=(\d+) index_bits=\d+ graph=([\d.]+)s "
    r"total=([\d.]+)s peak_rss=([\d.]+)MB")
TELEM = re.compile(r"H(\d+) columns=(\d+) skipped=(\d+)")
MODE_LINE = re.compile(r"mode: (.*)")
MODES = ("coboundary", "boundary", "auto")


def gen_cloud(path, family, n, dim, seed=7):
    if family == "gaussmix":
        import numpy as np
        rng = np.random.default_rng(seed)
        k = 8
        centers = rng.uniform(0.15, 0.85, (k, dim))
        assign = rng.integers(0, k, n)
        x = centers[assign] + 0.08 * rng.normal(size=(n, dim))
        np.savetxt(path, x, fmt="%.9g", delimiter=",")
        return
    subprocess.run([TURBOTOPS, "gen", family, str(n), str(dim), str(seed), path],
                   check=True, capture_output=True)


def calibrate_tau(path, n, target_degree):
    """Radius giving ~target_degree mean degree, from a subsample quantile.

    The actual mean degree is recorded from the run's edge count anyway; this
    only needs to land in the right neighborhood.
    """
    import numpy as np
    x = np.loadtxt(path, delimiter=",", dtype=np.float64)
    m = min(400, len(x))
    sub = x[np.random.default_rng(0).choice(len(x), m, replace=False)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    iu = np.triu_indices(m, 1)
    q = target_degree / (n - 1.0)
    return float(np.quantile(d[iu], min(q, 1.0)))


def run_mode(fixture, extra, maxdim, mode, threads):
    argv = [TURBOTOPS, "run", fixture, "--maxdim", str(maxdim),
            "--threads", str(threads), "--mode", mode] + extra
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.perf_counter()
        proc = subprocess.Popen(argv, stdout=out, stderr=err)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
        rc = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        stdout = out.read()
        stderr = err.read()
    if rc != 0:
        raise RuntimeError(f"{' '.join(argv)} exit {rc}: {stderr[-400:]}")
    scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 1024
    s = SUMMARY.search(stderr)
    mode_m = MODE_LINE.search(stderr)
    telem = {int(d): (int(c), int(k)) for d, c, k in TELEM.findall(stderr)}
    n_bars = len([ln for ln in stdout.splitlines() if ln.strip()])
    return {
        "diagram": stdout,
        "sha": hashlib.sha1(stdout.encode()).hexdigest()[:12],
        "wall_s": round(wall, 3),
        "total_s": float(s.group(3)),
        "graph_s": float(s.group(2)),
        "peak_mb": round(ru.ru_maxrss * scale, 1),
        "edges": int(s.group(1)),
        "reason": mode_m.group(1) if mode_m else "",
        "telemetry": telem,
        "bars": n_bars,
    }


def cells(quick):
    """Grid cells: (family, input, n, dim, maxdim, label, extra, target_deg)."""
    out = []
    for n in ([2000] if quick else [2000, 5000, 10000]):
        out.append(("uniform", "dense", n, 3, 1, "dense", [], None))
    out.append(("uniform", "dense", 2000, 3, 2, "dense", [], None))
    knn_md1 = [(5000, 32)] if quick else [
        (5000, 8), (5000, 32), (5000, 128),
        (20000, 8), (20000, 32),
        (50000, 8), (50000, 32),
    ]
    for n, k in knn_md1:
        out.append(("uniform", "knn", n, 3, 1, f"knn-k{k}",
                    ["--graph", "knn", "-k", str(k)], None))
    if not quick:
        for n, k in ((5000, 8), (5000, 32)):
            out.append(("uniform", "knn", n, 3, 2, f"knn-k{k}",
                        ["--graph", "knn", "-k", str(k)], None))
        # Crossover fill: pin the H1-sparse mean-degree threshold (measured
        # degree is ~1.15 x k on the symmetric graph: k 12/16/22/28 lands at
        # mean degree ~14/18/25/32, bracketing the v1 constant 32).
        for n in (10000, 30000):
            for k in (12, 16, 22, 28):
                out.append(("uniform", "knn", n, 3, 1, f"fill-k{k}",
                            ["--graph", "knn", "-k", str(k)], None))
        # Input families: the rule must not be a uniform-distribution artifact.
        out.append(("torus", "dense", 2000, 3, 1, "torus", [], None))
        out.append(("torus", "dense", 2000, 3, 2, "torus", [], None))
        out.append(("gaussmix", "dense", 5000, 3, 1, "gaussmix", [], None))
        out.append(("gaussmix", "dense", 1500, 3, 2, "gaussmix", [], None))
        # H >= 3 spot checks: dense (sphere S^3 in R^4) and sparse (kNN-8).
        out.append(("sphere", "dense", 120, 4, 3, "sphere-d4", [], None))
        out.append(("uniform", "knn", 3000, 3, 3, "knn-k8",
                    ["--graph", "knn", "-k", "8"], None))
        for deg in (8, 128):
            out.append(("uniform", "radius", 10000, 3, 1, f"radius-deg~{deg}", [], deg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results_mode_grid.csv"))
    ap.add_argument("--filter", default="",
                    help="run only cells whose label contains this substring")
    args = ap.parse_args()

    if not os.path.exists(TURBOTOPS):
        sys.exit("build the CLI first (make)")

    wanted = cells(args.quick)
    if args.filter:
        wanted = [c for c in wanted if args.filter in c[5]]

    rows = []
    medians = {}
    tmpdir = tempfile.mkdtemp(prefix="mode_grid_")
    for family, input_kind, n, dim, maxdim, label, extra, target_deg in wanted:
        fixture = os.path.join(tmpdir, f"{family}{n}.txt")
        if not os.path.exists(fixture):
            gen_cloud(fixture, family, n, dim)
        cell_extra = list(extra)
        if input_kind == "radius":
            tau = calibrate_tau(fixture, n, target_deg)
            cell_extra = ["--threshold", f"{tau:.6f}"]

        runs = {}
        for rep in range(args.reps):        # interleaved: rep-major
            for mode in MODES:
                r = run_mode(fixture, cell_extra, maxdim, mode, args.threads)
                runs[(rep, mode)] = r
                print(f"{family:9s} {input_kind:7s} n={n:<6d} H{maxdim} {label:14s} "
                      f"rep{rep} {mode:10s} total={r['total_s']:8.2f}s "
                      f"wall={r['wall_s']:8.2f}s peak={r['peak_mb']:7.1f}MB "
                      f"edges={r['edges']:8d} deg={2.0 * r['edges'] / n:7.1f} "
                      f"bars={r['bars']:6d}", flush=True)

        shas = {r["sha"] for r in runs.values()}
        assert len(shas) == 1, (
            f"PARITY FAILURE {family}/{input_kind} n={n} H{maxdim} {label}: "
            "diagrams differ across modes/reps: " +
            ", ".join(f"{m}={runs[(0, m)]['sha']}" for m in MODES))
        mean_deg = 2.0 * runs[(0, "coboundary")]["edges"] / n
        key = (family, input_kind, n, maxdim, label)
        med, spread = {}, {}
        for mode in MODES:
            ts = sorted(runs[(rep, mode)]["total_s"] for rep in range(args.reps))
            med[mode] = statistics.median(ts)
            spread[mode] = ts
            rep0 = runs[(0, mode)]
            t1 = rep0["telemetry"].get(1, (0, 0))
            for rep in range(args.reps):
                r = runs[(rep, mode)]
                rows.append({
                    "family": family, "input": input_kind, "n": n, "dim": dim,
                    "maxdim": maxdim, "label": label, "mean_degree": round(mean_deg, 1),
                    "mode": mode, "rep": rep, "total_s": r["total_s"],
                    "wall_s": r["wall_s"], "peak_mb": r["peak_mb"],
                    "edges": r["edges"], "bars": r["bars"],
                    "h1_columns": t1[0], "h1_skipped": t1[1],
                    "diagram_sha": r["sha"], "auto_reason": r["reason"] if mode == "auto" else "",
                })
        medians[key] = (mean_deg, med, spread, runs[(0, "auto")]["reason"])

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nmedians (total s [min, max]) and auto-check:")
    n_bad = 0
    for key, (deg, med, spread, reason) in medians.items():
        winner = min(("coboundary", "boundary"), key=lambda m: med[m])
        auto_engine = "boundary" if reason.startswith("auto: boundary") else "coboundary"
        ok = auto_engine == winner
        n_bad += not ok
        fmt = lambda m: f"{med[m]:8.2f} [{spread[m][0]:.2f},{spread[m][-1]:.2f}]"
        print(f"  {key[0]:9s} {key[1]:7s} n={key[2]:<6d} H{key[3]} {key[4]:14s} deg={deg:7.1f} "
              f"cob={fmt('coboundary')} bnd={fmt('boundary')} auto={fmt('auto')} "
              f"winner={winner:10s} auto->{auto_engine:10s} {'OK' if ok else 'MISMATCH'}",
              flush=True)
    assert not n_bad, f"auto picked the slower engine in {n_bad} cell(s)"
    print(f"wrote {args.out} ({len(rows)} rows); all cells byte-identical; "
          f"auto matched the median winner everywhere")


if __name__ == "__main__":
    main()
