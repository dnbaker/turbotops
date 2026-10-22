#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render benchmark CSVs (from compare.py) as a Markdown table.

    python bench/table.py results.csv [more.csv ...]

Later files override earlier ones for the same (case, engine). Cells show
seconds / peak RSS in MB; '—' = engine not applicable or not run.
"""
import csv
import sys

ORDER = ["turbotops t1", "turbotops t16", "ripser-cli", "ripser-py", "giotto-ph t1",
         "giotto-ph t16", "giotto-ph-collapse t16", "redzed"]
HEAD = {"turbotops t1": "turbotops (1 thr)", "turbotops t16": "turbotops (16 thr)",
        "ripser-cli": "Ripser CLI", "ripser-py": "ripser.py", "giotto-ph t1": "giotto-ph (1 thr)",
        "giotto-ph t16": "giotto-ph (16 thr)", "giotto-ph-collapse t16": "giotto-ph + collapse (16 thr)",
        "redzed": "RedZeD"}


def main():
    cells, cases = {}, []
    for path in sys.argv[1:]:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["case"] not in cases:
                    cases.append(r["case"])
                cells[(r["case"], r["engine"])] = r
    engines = [e for e in ORDER if any((c, e) in cells for c in cases)]
    print("| complex | " + " | ".join(HEAD[e] for e in engines) + " |")
    print("|---|" + "---|" * len(engines))
    for c in cases:
        row = []
        for e in engines:
            r = cells.get((c, e))
            if r is None:
                row.append("—")
            elif r["parity"].startswith("ERROR"):
                row.append("failed")
            else:
                mark = "" if r["parity"] == "ok" else " †"
                rss = f" / {float(r['peak_rss_mb']):.0f}" if r["peak_rss_mb"] else ""
                row.append(f"{float(r['seconds']):.2f}{rss}{mark}")
        print(f"| {c} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
