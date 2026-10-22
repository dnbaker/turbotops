# turbotops

Persistent homology of Vietoris–Rips filtrations: point clouds under any
Minkowski metric (L1, L2, L∞, Lp for p ≥ 1) or weighted graphs, in any
homology dimension. C++20 header library, command-line tool and Python
module.

- **Any Minkowski norm.** L1, L2, L∞ and general Lp (p ≥ 1) for full,
  thresholded and k-NN complexes, with exact k-d tree search under each
  metric. Most Rips libraries support only Euclidean distance or a
  precomputed matrix.
- **Exact.** Diagrams are verified bar-for-bar against the Ripser CLI on
  identical distances (`turbotops test`), including tie-heavy inputs.
- **Two engines, one answer.** A parallel coboundary (cohomology) reducer and
  a port of RedZeD (boundary reduction to zero differentials with active
  enumeration), chosen by `--mode`; both give byte-identical diagrams.
- **Any dimension, any size.** Simplices are indexed with 64-bit integers, or
  128-bit when the complex needs it, so neither the number of points nor the
  homology dimension is capped by the index width.
- **Low memory.** Full Rips keeps one distance matrix (or half of one under
  a tight budget); thresholded and k-NN complexes keep only their edges.
- **Deterministic.** Results do not depend on the number of threads.

## Build

```bash
make CXX=/usr/local/bin/g++-15   # optimized for this machine (-march=native)
make PORTABLE=1                  # generic binary for the target architecture
make NO_SIMD=1                   # scalar kernels only
scripts/fetch_ripser.sh          # reference Ripser CLI, used by the tests
make test
```

Requirements: a C++20 compiler (GCC ≥ 11 or Clang ≥ 14) and POSIX threads.
Verified on Apple Silicon (macOS) with
`make CXX=/usr/local/bin/g++-15` (Homebrew GCC 15.3) and
`make CXX=/usr/bin/clang++` (Apple clang 21); both pass the full self-test
(`make test`: 117 unit checks plus Ripser bar-for-bar parity when the
reference CLI is present, a CLI smoke suite, and header checks). Plain
`make` uses `c++`, and an environment `CXX` overrides the default — pick
the toolchain explicitly. Two toolchain notes: use the `g++-15` driver, not
`gcc-15` — the C driver fails to link libstdc++ on that machine; and the
Homebrew LLVM clang 16 (`/usr/local/opt/llvm/bin/clang++`) cannot find the
macOS SDK headers there (`sys/stat.h`), which is an environment limitation,
not a source issue.
SIMD kernels are written once against [SIMDe](https://github.com/simd-everywhere/simde),
which maps them to AVX2, SSE, NEON or scalar code. Within a build, results
are bit-identical for any thread count. Across builds, L∞ distances are
bit-identical; for L1, L2 and Lp in dimension ≥ 16 the vectorized sums use a
different summation order than `NO_SIMD=1`, so distances (and bar endpoints)
can differ in the last bits. General Lp uses a vectorized power function with
relative error below 1e-5.

Using the headers from your own project: compile with the same arch flags
as the Makefile — `-march=native`, or `-Wno-psabi` with a baseline ISA for
portable builds. The SIMDe helpers in `metric.hpp` return 256-bit vectors;
without one of those flags GCC emits `-Wpsabi` diagnostics (an error under
`-Werror`).

The core build is self-contained: SIMDe, hnswlib and the RedZeD reference
are vendored under `third_party/` and no network access is needed. The two
fetch scripts — `scripts/fetch_ripser.sh` (reference CLI for `make test`
parity checks) and the nanobind clone inside `scripts/build_python.sh` —
download their targets on first use and are otherwise unnecessary.

Python (needs `numpy`; fetches nanobind on first build):

```bash
PYTHON=python3 scripts/build_python.sh
PYTHONPATH=python python3 -c "import turbo_tops"
```

## Command line

```bash
turbotops gen torus 1000 3 1 torus.txt                 # sample cloud
turbotops run torus.txt --maxdim 2                     # full Rips, L2 (~5 s)
turbotops run torus.txt --maxdim 1 --metric l1         # any Minkowski metric
turbotops run torus.txt --maxdim 1 --threshold 0.3     # thresholded (sparse)
turbotops run torus.txt --graph knn -k 15 --maxdim 2   # flag complex of the k-NN graph
printf '0 1 1\n1 2 1\n2 0 1\n' > graph.txt             # weighted edge list 'i j w'
turbotops run graph.txt --edges --maxdim 2             # flag complex of the graph
```

Output is one bar per line, `dim birth death` (`inf` for essential classes).
A summary (method, effective threshold, edges, timings, peak memory) goes to
stderr.

Inputs: text with one point per line (comma or whitespace separated), or raw
float32 with `--dim`. Run `turbotops --help` for all options.

## Python

```python
import numpy as np, turbo_tops as tt

x = np.random.default_rng(0).normal(size=(5000, 3)).astype(np.float32)
dgms, info = tt.persistence(x, maxdim=2)                  # full Rips, L2
dgms, info = tt.persistence(x, maxdim=1, metric="lp:1.5")
dgms, info = tt.persistence(x, maxdim=2, graph="knn", k=15)
dgms, info = tt.persistence(scipy_sparse_matrix, maxdim=2)  # graph input
```

`dgms[d]` is an `(m, 2)` float32 array of `(birth, death)` pairs.

## How it works

| Input | Complex | Method |
|---|---|---|
| points, no threshold | full Rips | dense distance matrix, filtration cut at the enclosing radius (beyond it the complex is a cone, so no bar changes) |
| points, threshold τ | Rips up to τ | exact radius graph, sparse reduction |
| points, `--graph knn` | flag complex of the symmetric k-NN graph | exact k-NN (or HNSW with `--approximate`), sparse reduction |
| weighted graph | its flag complex | sparse reduction |
| points, `--maxdim 0` | full Rips, H0 | minimum spanning tree, O(n) memory |

The reduction is the cohomology algorithm of Ripser (Bauer, 2021): implicit
coboundary columns over the combinatorial number system, apparent and
emergent pairs, clearing between dimensions. Implementation choices that
differ from Ripser:

- **Row-streamed cofacet scans.** With enough memory the dense matrix is
  stored in full rows, so the scan for the cofacets of a simplex streams one
  contiguous row per vertex; candidates are tested eight at a time
  (row maxima against the cutoff) and whole blocks are skipped.
- **Parallel apparent-pair tests.** Deciding which edges are apparent pairs
  is independent per edge and runs on all cores before the (sequential)
  reduction; column assembly for each dimension is parallel as well.
- **Wide indices.** The reducer is instantiated with 64- or 128-bit simplex
  indices, chosen from C(n, maxdim + 2).
- **Exact neighbor search for every Minkowski metric.** For any p ≥ 1,
  |x_c − y_c| ≤ ‖x − y‖_p for each coordinate c. Points are sorted along the
  coordinate of largest spread; radius queries scan only the window of
  projected gap ≤ τ and k-NN queries grow a window until the gap exceeds the
  current k-th distance. Exact for any input, fast in low dimension.
- **Overflow-safe Lp.** Distances are computed as
  M · (Σ (|d_k|/M)^p)^(1/p) with M = max |d_k|, so no term overflows for any
  p or coordinate scale.

## Engines and modes

`--mode boundary|coboundary|auto` (default `auto`; `--coboundary`,
`--no-coboundary`/`--boundary` force a side; `--engine reducer|redzed` remains
as a legacy alias). The coboundary engine is ON BY DEFAULT: it is the general
path, and wherever the measurements below do not separate the two engines,
the default-on philosophy keeps coboundary.

- **coboundary** (`rips.hpp`): implicit cohomology reduction in the style of
  Ripser with apparent and emergent pairs and clearing, multithreaded,
  64- or 128-bit simplex indices, any `maxdim <= 30`, any input. This is the
  capability-universal path.
- **boundary** (`redzed.hpp`): RedZeD (Kapulkin and Kershaw, arXiv 2606.06310;
  port of the MIT reference code). Boundary-matrix reduction to zero
  differentials with active enumeration. Faster than the upstream
  implementation on identical inputs, with exactness-preserving additions:
  - an O(m) coface-readiness test (colexicographic face order plus a
    maximal-edge mask) in place of the O(m^3) one;
  - deferred creation of sets for simplices that die immediately;
  - a parallel, order-preserving edge build;
  - k-d tree vertex relabelling for locality;
  - SIMD bitset and distance kernels.

  Capability gate: single-threaded, 64-bit simplex indices only
  (`C(n, maxdim + 2) < 2^64`, implied `n < 2^31`). A forced
  `--mode boundary` on a problem past that limit is an error, not a silent
  switch; under `--mode auto` the selector falls back to coboundary and says
  so.
- **auto** (default) picks boundary only where the measured selector says it
  wins — H1 on dense full-Rips input, or H1 on graphs of mean degree ≥ 16,
  when the complex fits the 64-bit index gate — and coboundary everywhere
  else. The rule was measured with `bench/mode_grid.py`: 29
  cells × 3 modes × 3 interleaved reps, medians reported (dense uniform
  n = 2k–10k; exact kNN n = 5k–50k; a crossover fill at n = 10k/30k pinning
  the sparse-H1 degree threshold inside its measured bracket (13.8, 18.2),
  midpoint 16; radius graphs; torus and Gaussian-mixture families at H1/H2;
  H3 spot checks). Both engines give byte-identical
  diagrams on every grid cell — mode selection changes speed, never output —
  and `turbotops test` re-checks this explicitly, running both forced modes
  plus auto on dense and sparse cases. The chosen mode and why are printed on
  stderr (`turbotops: mode: ...`).

Reduction internals (apparent/emergent pairs, row-streamed cofacet scans,
parallel apparent-pair tests, wide indices, exact neighbor search under
every Minkowski metric, overflow-safe Lp) are documented in "How it works"
below.

## Approximate k-NN and measured recall

`--approximate` builds the k-NN graph with HNSW. The flag complex of an
approximate k-NN graph is a different complex from the exact one, so
turbotops measures it: `--verify-recall S` compares S random points against
exact search and reports recall and a 95% lower bound on the fraction of
points whose neighbor list is exact.

## Benchmarks

### Current engines (this machine)

Cells are seconds / peak resident memory in MB. "turbotops" columns are the
coboundary engine (multithreaded); "RedZeD" is the boundary engine — the port
inside turbotops, i.e. what `--mode auto` picks for full-Rips H1. Both give
byte-identical diagrams on every row. Provenance: committed measurements in
`bench/results_redzed.csv` and `bench/results_turbotops_kdtree.csv` (whole
process timing; 8-core, 16-thread laptop CPU; the boundary engine is
single-threaded by design).

| complex | turbotops coboundary (1 thr) | turbotops coboundary (16 thr) | boundary (RedZeD mode) |
|---|---|---|---|
| full uniform n=5000 H1 l2 | 7.70 / 458 | — | 1.84 / 318 |
| full uniform n=10000 H1 l2 | 36.22 / 1359 | — | 8.61 / 1117 |
| full torus n=1200 H2 l2 | 45.71 / 392 | — | 33.31 / 222 |
| full uniform n=5000 H1 l1 | 10.83 / 514 | — | 2.16 / 334 |
| full uniform n=5000 H1 linf | 7.73 / 457 | — | 1.25 / 274 |
| full uniform n=5000 H1 lp:3 | 7.96 / 457 | — | 1.74 / 305 |
| tau uniform n=10000 H1 l2 tau=0.164 | 1.08 / 46 | 0.69 / 55 | 1.30 / 696 |
| tau uniform n=10000 H2 l2 tau=0.12 | 3.40 / 30 | 1.10 / 30 | 3.37 / 732 |
| tau uniform n=25000 H1 l2 tau=0.128 | 4.94 / 140 | 3.50 / 172 | — |
| tau uniform n=50000 H1 l2 tau=0.105 | 14.99 / 318 | 11.06 / 390 | — |
| knn uniform n=100000 H1 l2 k=15 | 2.98 / 84 | 2.05 / 90 | — |
| knn uniform n=100000 H2 l2 k=15 | 4.88 / 85 | 2.41 / 91 | — |

Summary: on full Rips the boundary engine wins H1 by 4–5× and H2 by ~1.4×,
at lower peak memory. On thresholded (tau) and k-NN complexes the two
engines are within noise of each other in time (tau H2: 3.37 s vs 3.40 s),
but the boundary engine keeps the dense distance matrix and uses ~10–20×
more peak memory there (696–732 MB vs 30–55 MB), which is exactly why
`--mode auto` routes sparse input to coboundary. A forced `--mode boundary`
past the 64-bit index gate is an error; under `auto` the selector falls
back to coboundary and says so on stderr.

### Cross-library comparison (early build, before the boundary engine)

Measured with an early internal build in which only the coboundary reducer
existed, so its "turbotops" columns are the coboundary engine alone. Current
per-mode numbers are in the table above.

`bench/compare.py` runs turbotops, the Ripser CLI, ripser.py, giotto-ph and RedZeD on
the same complexes (see the script header for how inputs are matched) and
checks that all engines report the same number of bars.

| complex | turbotops (1 thr) | turbotops (16 thr) | Ripser CLI | ripser.py | giotto-ph (1 thr) | giotto-ph (16 thr) | giotto-ph + collapse (16 thr) | RedZeD |
|---|---|---|---|---|---|---|---|---|
| full uniform n=5000 H1 l2 | 6.29 / 458 | 5.37 / 458 | 18.31 / 656 | 15.77 / 2269 | 17.93 / 1189 | 10.67 / 1264 | — | 1.84 / 318 |
| full uniform n=10000 H1 l2 | 31.21 / 1359 | 27.08 / 1358 | 97.03 / 2052 | 74.99 / 7405 | 92.32 / 3703 | 55.05 / 3903 | — | 8.61 / 1117 |
| full torus n=1200 H2 l2 | 46.84 / 393 | 10.51 / 388 | 53.62 / 518 | 137.33 / 8373 | 48.40 / 641 | 10.91 / 640 | — | 33.31 / 222 |
| full uniform n=5000 H1 l1 | 11.14 / 514 | 9.94 / 513 | — | 23.06 / 2398 | 25.11 / 1280 | 14.46 / 1410 | — | 2.16 / 334 |
| full uniform n=5000 H1 linf | 6.77 / 457 | 6.16 / 457 | — | 12.83 / 1826 | 19.75 / 1043 | 12.77 / 1215 | — | 1.25 / 274 |
| full uniform n=5000 H1 lp:3 | 7.40 / 457 | 6.03 / 457 | — | — | 19.06 / 1105 | 12.80 / 1122 | — | 1.74 / 305 |
| tau uniform n=10000 H1 l2 tau=0.164 | 1.08 / 46 | 0.69 / 55 | 2.50 / 72 | 2.47 / 247 | 3.31 / 364 | 2.81 / 367 | 8.17 / 358 | 1.30 / 696 |
| tau uniform n=25000 H1 l2 tau=0.128 | 4.94 / 140 | 3.50 / 172 | 9.49 / 254 | 7.75 / 654 | 10.46 / 923 | 8.00 / 879 | 28.35 / 787 | — |
| tau uniform n=50000 H1 l2 tau=0.105 | 14.99 / 318 | 11.06 / 390 | 25.23 / 592 | 20.52 / 1333 | 26.68 / 1927 | 20.35 / 1950 † | 71.37 / 1613 † | — |
| tau uniform n=10000 H2 l2 tau=0.12 | 3.40 / 30 | 1.10 / 30 | 4.50 / 38 | 4.82 / 438 | 5.12 / 224 | 2.23 / 222 | 3.14 / 234 † | 3.37 / 732 |
| knn uniform n=100000 H1 l2 k=15 | 2.98 / 84 | 2.05 / 90 | 6.34 / 92 | 5.41 / 278 | 5.97 / 409 | 4.67 / 413 | 6.55 / 748 | — |
| knn uniform n=100000 H2 l2 k=15 | 4.88 / 85 | 2.41 / 91 | 10.18 / 95 | failed | 9.12 / 410 | 5.06 / 413 † | 6.71 / 748 | — |
Cells (early-build table) are seconds / peak resident memory in MB (median of 2
runs; 8-core, 16-thread laptop CPU, idle). turbotops and the Ripser CLI are
timed as whole
processes, including reading the text input; ripser.py, giotto-ph and RedZeD
are timed around the library call. For thresholded (tau) and k-NN complexes,
the edge list given to the other engines is built with scipy and its time is
added to theirs; turbotops builds its own graph inside its timing. RedZeD
accepts only a dense distance matrix (built beforehand, not timed), so it
appears only where that matrix is practical. ripser.py has no Minkowski
exponent (lp:3 row) and failed on the k-NN H2 case at n = 100,000 (simplex
index overflow). giotto-ph's edge collapse is not run on full complexes: in
a pilot it took 750 s where plain giotto-ph took 2 s (n = 2,000, L1).
† One bar more or fewer than every other engine in this run; not reproduced
in isolated reruns of the same inputs.

Summary (early-build table): on full Rips complexes RedZeD is the fastest engine (4–6× faster
than turbotops in H1, 1.4× in H2) and turbotops is 2.5–3× faster than the
Ripser CLI on one thread. On thresholded and k-NN complexes turbotops is
the fastest engine at one thread and ties RedZeD where RedZeD can run; its
peak memory there is 1.1–2× lower than the Ripser CLI, 3–15× lower than
ripser.py and giotto-ph, and 13–25× lower than RedZeD.

## Testing

`turbotops test` runs unit tests (metric axioms, exact neighbor search
against brute force, 64- vs 128-bit indices, thread-count independence,
MST vs Rips H0, option-validation gates, the dense lower layout, and an
HNSW/recall smoke check) and, when the Ripser CLI is available, compares
diagrams bar-for-bar with Ripser for dense, k-NN and thresholded complexes in
dimensions up to 4 under L1, L2, L∞ and Lp. `make test` additionally runs
`tests/test_cli.sh` (every documented flag, golden diagrams on
hand-computable inputs, and the error paths, each checked for its message and
exit code) and `tests/check_headers.sh` (every public header compiled
standalone under `-Wall -Wextra -Werror` in both arch profiles, plus a
two-TU ODR/linkage check).
`PYTHONPATH=python python3 tests/test_python.py` checks that the Python
module returns exactly the command-line results in every mode.

## License

MIT (see `LICENSE`). Third-party components: `THIRD_PARTY_NOTICES.md`.
