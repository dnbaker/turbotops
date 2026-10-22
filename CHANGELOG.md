# Changelog

All notable changes to turbotops. Releases are tagged `v<version>`.

## v0.1.0 — 2026-10-22

Initial public release: two reduction engines behind one exactness gate —
`turbotops test` compares diagrams bar-for-bar against the Ripser CLI
(117 in-binary checks plus Ripser parity), and both engines produce
byte-identical diagrams on the auto-selector measurement grid
(29 cells × 3 modes × 3 reps; `bench/results_mode_grid.csv`).

### Engines

- **Coboundary (cohomology) reducer** in the style of Ripser (Bauer, 2021):
  apparent/emergent pairs, clearing, row-streamed cofacet scans, parallel
  apparent-pair tests, 64-/128-bit simplex indices, `maxdim <= 30`, any
  input. Simplices are addressed by the combinatorial number system and
  never stored as vertex lists.
- **Boundary engine (RedZeD port)**: boundary-matrix reduction to zero
  differentials with active enumeration (Kapulkin & Kershaw, arXiv
  2606.06310; port of the MIT reference code). Additions over upstream:
  exact O(1) coface-readiness test, O(m) face indices with prefetch,
  deferred singleton-set creation, parallel order-preserving edge build,
  k-d tree vertex relabelling, SIMDe bitset/distance kernels, CSR edge
  rank width templated on edge count (u32/u64). Single-threaded and
  64-bit simplex indices only (`C(n, maxdim + 2) < 2^64`, implied
  `n < 2^31`).
- **`--mode boundary|coboundary|auto` selector** (default `auto`) with a
  measured rule — boundary only where it wins: H1 on dense full-Rips input,
  or H1 on graphs of mean degree ≥ 16, subject to the 64-bit simplex-index
  gate; coboundary everywhere else. The sparse-H1 crossover was pinned with
  a 3-rep grid (measured bracket (13.8, 18.2), midpoint 16). Mode selection
  changes speed, never output; a forced `--mode boundary` past the index
  gate is an error, under `auto` the selector falls back to coboundary and
  says so.
- **Exact neighbor search under every Minkowski metric** (exact k-d tree:
  windowed coordinate scans with per-axis lower bounds); overflow-safe Lp
  distances (no term exceeds 1 after normalization by the max gap).
- **H0 via minimum spanning tree** in O(n) memory.
- **Optional HNSW approximate k-NN** with measured recall reporting
  (`--verify-recall`: recall plus a 95% lower bound on fully-exact lists).

### Input

- Point clouds under any Minkowski metric (L1, L2, L∞, Lp for p ≥ 1); text
  input (comma/whitespace separated, `#` comments, UTF-8 BOM tolerated) or
  raw little-endian float32 `.bin` with `--dim`.
- Thresholded (radius) complexes and k-NN flag complexes.
- Weighted graphs as edge lists (`--edges`), one `i j w` per line.

### CLI

- `turbotops --version` prints `turbotops 0.1.0`.
- Integer options (`--maxdim`, `-k`, `--dim`, `--threads`, `--hnsw-m`,
  `--ef-construction`, `--ef-search`, `gen` sizes) are range-checked:
  values past 2^32 - 1 or negative are rejected with "out of range" rather
  than wrapping.
- Empty point files, empty edge lists and directory inputs are errors
  ("no points" / "no vertices"), not silent empty diagrams.
- `--approximate` and `--verify-recall` require `--graph knn`.
- `--output` is validated before the computation starts.
- Diagram output is canonically ordered (dimension, birth, death), so both
  engines produce byte-identical output; zero-length bars are omitted.

### Library

- `TURBOTOPS_VERSION_{MAJOR,MINOR,PATCH}` and `TURBOTOPS_VERSION_STRING`
  in `core.hpp`; `turbo_tops.__version__` mirrors the CLI.
- `tops::persistence()` (both overloads) rejects NaN/negative thresholds
  and `k = 0`, and checks the `CsrGraph` invariant, so the CLI, the Python
  module and direct C++ users all fail loudly on degenerate parameters.
- Both engines gate `maxdim <= 30`; the RedZeD face-index tables and
  position masks never see an out-of-envelope dimension.
- `mst::h0` rejects non-finite coordinates at entry.

### Tests

- 117 in-binary checks (run by `turbotops test`): metric axioms, exact
  neighbor search against brute force, 64- vs 128-bit indices, thread-count
  independence, option validation, invalid CSR graph,
  `parse_mode`/`parse_engine` rejections, both engine maxdim gates, MST
  threshold behavior, dense lower layout vs full layout, HNSW/recall smoke —
  plus Ripser bar-for-bar parity on dense, k-NN and thresholded complexes
  up to dimension 4 under L1, L2, L∞ and Lp.
- `tests/test_cli.sh`: CLI smoke suite — every documented flag, exact
  golden diagrams on hand-computable inputs (unit square under
  L1/L2/L∞/Lp, thresholds, k-NN, edge lists, all mode flags), `.bin`
  roundtrip, and ~40 error paths each checked for message and exit code.
- `tests/check_headers.sh`: every public header compiled standalone under
  `-Wall -Wextra -Werror` in the native and portable arch profiles, plus a
  two-TU umbrella-header ODR/linkage check.
- `tests/test_python.py`: the Python module returns exactly the
  command-line results in every mode.

### Known limitations (honest negatives)

- The boundary (RedZeD) engine is single-threaded and 64-bit-index only;
  `--mode auto` routes everything else to the coboundary engine.
- On thresholded (tau) and k-NN complexes the two engines are within noise
  of each other in speed (tau H2: redzed 3.37 s vs coboundary 3.43 s,
  n = 10000); the boundary engine's peak memory is much higher there
  (~700 MB vs ~30–90 MB) because it keeps the dense distance matrix.
