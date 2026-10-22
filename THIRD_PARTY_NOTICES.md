# Third-party components

| Component | Location | License |
|---|---|---|
| SIMDe 0.8.4 (amalgamated `x86/avx2.h`, `x86/fma.h`) | `third_party/simde/` | MIT — `third_party/simde/COPYING` |
| hnswlib 0.8.0 | `third_party/hnswlib/` | Apache-2.0 — `third_party/hnswlib/LICENSE` |
| nanobind (Python bindings; fetched at build time) | — | BSD-3-Clause |
| RedZeD @22ef1bf (algorithm port in `include/turbotops/redzed.hpp`) | — | MIT — `third_party/redzed/LICENSE` (Copyright (c) 2026 Nathan Kershaw) |

The persistence algorithm in `include/turbotops/rips.hpp` follows
U. Bauer, *Ripser: efficient computation of Vietoris–Rips persistence
barcodes*, J. Appl. Comput. Topol. 5 (2021); the implementation is
independent.
