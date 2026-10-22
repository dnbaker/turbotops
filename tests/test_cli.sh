#!/usr/bin/env sh
# SPDX-License-Identifier: MIT
# CLI smoke suite for turbotops: every documented flag, golden diagrams on
# hand-computable inputs, and the error paths (each must exit 2 with a clear
# message). Run from the repo root:  sh tests/test_cli.sh
#
# Golden diagrams are exact-text. They are stable because every golden uses
# low-dimensional inputs (scalar distance kernels — no SIMD summation-order
# variance) and the CLI output is canonically sorted.
set -u
cd "$(dirname "$0")/.."
TT=./turbotops
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0

ok() { pass=$((pass + 1)); }
bad() { echo "  [FAIL] $1" >&2; fail=$((fail + 1)); }

# expect_err <name> <expected-exit> <stderr-substring> <cmd...>
expect_err() {
    name=$1; want=$2; sub=$3; shift 3
    "$@" > "$TMP/o" 2> "$TMP/e"
    rc=$?
    if [ "$rc" -ne "$want" ]; then bad "$name: exit $rc (want $want)"; return; fi
    if ! grep -qF -- "$sub" "$TMP/e"; then bad "$name: stderr lacks '$sub' (got: $(cat "$TMP/e"))"; return; fi
    ok
}

# expect_golden <name> <golden-file> <cmd...>
expect_golden() {
    name=$1; gold=$2; shift 2
    if ! "$@" > "$TMP/o" 2> "$TMP/e"; then bad "$name: exit $? ($(cat "$TMP/e"))"; return; fi
    if ! diff -q "$gold" "$TMP/o" > /dev/null; then bad "$name: diagram mismatch"; diff "$gold" "$TMP/o" >&2 | head -6; return; fi
    ok
}

# ---- inputs ----------------------------------------------------------------
# Unit square (0,0) (1,0) (1,1) (0,1): edges 1,1,1,1, diagonals sqrt(2).
printf '0 0\n1 0\n1 1\n0 1\n' > "$TMP/sq.txt"
# Right triangle: edges 1, 1, sqrt(2) — its H1 class has zero length and is
# omitted, so the diagram is H0 only.
printf '0 0\n1 0\n0 1\n' > "$TMP/tri.txt"
printf '' > "$TMP/empty.txt"
printf '0 0\n1 0 2\n' > "$TMP/badrows.txt"
printf '0 0\nfoo 1\n' > "$TMP/badcoord.txt"
printf '\xEF\xBB\xBF0 0\n1 0\n1 1\n0 1\n' > "$TMP/bom.txt"   # UTF-8 BOM + square
printf '0 1 1.0\n1 2 1.0\n2 3 1.0\n3 0 1.0\n0 2 1.4142135623730951\n1 3 1.4142135623730951\n' > "$TMP/sq.edg"
printf '# square with a duplicate edge (keep the smaller weight)\n0 1 9.0\n' >> "$TMP/sq.edg"
printf '0 1\n' > "$TMP/badedge.txt"
printf '0 1 -2.0\n' > "$TMP/negw.txt"
printf '99999999999 0 1.0\n' > "$TMP/vbig.txt"
head -c 14 /dev/zero > "$TMP/trunc.bin"

cat > "$TMP/sq.l2.gold" <<'EOF'
0 0 1
0 0 1
0 0 1
0 0 inf
1 1 1.41421354
EOF
cat > "$TMP/h0sq.gold" <<'EOF'
0 0 1
0 0 1
0 0 1
0 0 inf
EOF
cat > "$TMP/tri.gold" <<'EOF'
0 0 1
0 0 1
0 0 inf
EOF
cat > "$TMP/sq.l1.gold" <<'EOF'
0 0 1
0 0 1
0 0 1
0 0 inf
1 1 2
EOF
cat > "$TMP/sq.lp3.gold" <<'EOF'
0 0 1
0 0 1
0 0 1
0 0 inf
1 1 1.25992107
EOF
cat > "$TMP/sq.thr.gold" <<'EOF'
0 0 1
0 0 1
0 0 1
0 0 inf
1 1 inf
EOF

# ---- version / help ---------------------------------------------------------
[ "$($TT --version)" = "turbotops 0.1.0" ] && ok || bad "--version output"
$TT --help > /dev/null 2>&1 && ok || bad "--help exit"
$TT help > /dev/null 2>&1 && ok || bad "help exit"
expect_err "bare turbotops" 2 "usage" $TT

# ---- golden diagrams --------------------------------------------------------
expect_golden "square full l2" "$TMP/sq.l2.gold" $TT run "$TMP/sq.txt" --maxdim 1 --quiet
expect_golden "square l1" "$TMP/sq.l1.gold" $TT run "$TMP/sq.txt" --maxdim 1 --metric l1 --quiet
expect_golden "square linf (zero-length H1 omitted)" "$TMP/h0sq.gold" \
    $TT run "$TMP/sq.txt" --maxdim 1 --metric linf --quiet
expect_golden "square lp:3" "$TMP/sq.lp3.gold" $TT run "$TMP/sq.txt" --maxdim 1 --metric lp:3 --quiet
expect_golden "square threshold cuts the diagonal" "$TMP/sq.thr.gold" \
    $TT run "$TMP/sq.txt" --maxdim 1 --threshold 1 --quiet
expect_golden "triangle (H0 only)" "$TMP/tri.gold" $TT run "$TMP/tri.txt" --quiet
expect_golden "BOM stripped" "$TMP/sq.l2.gold" $TT run "$TMP/bom.txt" --maxdim 1 --quiet
expect_golden "edge list incl. duplicate edge" "$TMP/sq.l2.gold" \
    $TT run "$TMP/sq.edg" --edges --maxdim 1 --quiet
expect_golden "kNN flag complex (diagonal-free cycle)" "$TMP/sq.thr.gold" \
    $TT run "$TMP/sq.txt" --graph knn -k 2 --maxdim 1 --quiet
# maxdim 2 with H2 must equal the maxdim-1 diagram here (no 2-class).
expect_golden "maxdim 2 stable" "$TMP/sq.l2.gold" $TT run "$TMP/sq.txt" --maxdim 2 --quiet

# ---- every documented flag, smoke -------------------------------------------
$TT gen gauss 60 3 7 "$TMP/c.txt" && ok || bad "gen gauss"
$TT gen uniform 30 2 3 "$TMP/u.txt" && ok || bad "gen uniform"
$TT gen sphere 30 4 3 "$TMP/s.txt" && ok || bad "gen sphere"
$TT gen torus 30 3 3 "$TMP/t.txt" && ok || bad "gen torus"
$TT gen gauss 60 3 7 "$TMP/c2.txt" && cmp -s "$TMP/c.txt" "$TMP/c2.txt" && ok || bad "gen determinism"

# Mode flags: identical output, never a different diagram.
for m in "--mode auto" "--mode coboundary" "--mode boundary" "--coboundary" \
         "--no-coboundary" "--boundary" "--engine auto" "--engine reducer" "--engine redzed"; do
    # shellcheck disable=SC2086
    expect_golden "mode flag $m" "$TMP/sq.l2.gold" $TT run "$TMP/sq.txt" --maxdim 1 $m --quiet
done
# --threads must not change output.
$TT run "$TMP/c.txt" --maxdim 1 --threads 1 --quiet > "$TMP/t1" 2>/dev/null
$TT run "$TMP/c.txt" --maxdim 1 --threads 4 --quiet > "$TMP/t4" 2>/dev/null
cmp -s "$TMP/t1" "$TMP/t4" && ok || bad "--threads changes output"
# --output: diagram goes to the file, stdout stays empty.
$TT run "$TMP/sq.txt" --maxdim 1 --quiet --output "$TMP/diag.txt" > "$TMP/stdout.txt" 2>/dev/null \
    && [ ! -s "$TMP/stdout.txt" ] && cmp -s "$TMP/diag.txt" "$TMP/sq.l2.gold" && ok \
    || bad "--output plumbing"
# --quiet: nothing on stderr; without it, the summary appears.
$TT run "$TMP/sq.txt" --quiet 2> "$TMP/e" >/dev/null && [ ! -s "$TMP/e" ] && ok || bad "--quiet"
$TT run "$TMP/sq.txt" 2> "$TMP/e" >/dev/null && grep -q "turbotops: mode:" "$TMP/e" && ok || bad "summary on stderr"
# --memory-budget forces the lower dense layout; diagram is unchanged.
# (n=30 torus: full matrix 3600 B, lower 1740 B; a 3000 B budget picks lower.)
$TT run "$TMP/t.txt" --maxdim 2 --quiet > "$TMP/full.out" 2>/dev/null
$TT run "$TMP/t.txt" --maxdim 2 --quiet --memory-budget 3000 > "$TMP/low.out" 2>/dev/null
cmp -s "$TMP/full.out" "$TMP/low.out" && ok || bad "--memory-budget lower layout differs"
$TT run "$TMP/t.txt" --maxdim 2 --quiet --memory-budget 500 2>/dev/null && bad "tiny budget accepted" || ok
# .bin roundtrip: same data, same diagram.
$TT gen sphere 40 3 5 "$TMP/s40.bin" && $TT gen sphere 40 3 5 "$TMP/s40.txt"
$TT run "$TMP/s40.bin" --dim 3 --maxdim 1 --quiet > "$TMP/bin.out" 2>/dev/null
$TT run "$TMP/s40.txt" --maxdim 1 --quiet > "$TMP/txt.out" 2>/dev/null
cmp -s "$TMP/bin.out" "$TMP/txt.out" && ok || bad ".bin roundtrip"
$TT run "$TMP/s40.bin" --dim 3 --maxdim 1 --quiet --metric l1 > /dev/null 2>&1 && ok || bad ".bin with metric"
# --dim on a text input is accepted (only used for .bin).
expect_golden "--dim ignored for text" "$TMP/tri.gold" $TT run "$TMP/tri.txt" --dim 3 --quiet
# HNSW path + recall reporting + HNSW tunables.
$TT run "$TMP/c.txt" --graph knn -k 5 --approximate --verify-recall 10 --maxdim 1 \
    > /dev/null 2> "$TMP/e" && grep -q "kNN recall" "$TMP/e" && ok || bad "--approximate/--verify-recall"
$TT run "$TMP/c.txt" --graph knn -k 5 --approximate --verify-recall 0 --maxdim 1 \
    --hnsw-m 8 --ef-construction 60 --ef-search 24 > /dev/null 2> "$TMP/e" \
    && grep -q "kNN recall" "$TMP/e" && ok || bad "HNSW tunables / verify-recall 0"

# ---- error paths: loud, specific, exit 2 -------------------------------------
expect_err "missing file" 2 "cannot open" $TT run "$TMP/nope.txt"
expect_err "bad coordinate" 2 "bad coordinate 'foo'" $TT run "$TMP/badcoord.txt"
expect_err "inconsistent rows" 2 "inconsistent row length" $TT run "$TMP/badrows.txt"
expect_err "empty input" 2 "no points" $TT run "$TMP/empty.txt"
expect_err "directory as input" 2 "no points" $TT run "$TMP"
expect_err "bad edge row" 2 "expected 'i j w'" $TT run "$TMP/badedge.txt" --edges
expect_err "negative edge weight" 2 "bad weight" $TT run "$TMP/negw.txt" --edges
expect_err "empty edge list" 2 "no vertices" $TT run "$TMP/empty.txt" --edges
expect_err "vertex id too large" 2 "vertex id too large" $TT run "$TMP/vbig.txt" --edges
expect_err ".bin without --dim" 2 "--dim is required" $TT run "$TMP/trunc.bin"
expect_err ".bin bad size" 2 "not a multiple of 4 * dim" $TT run "$TMP/trunc.bin" --dim 3
expect_err "--maxdim overflow" 2 "out of range" $TT run "$TMP/sq.txt" --maxdim 4294967297
expect_err "--maxdim negative" 2 "out of range" $TT run "$TMP/sq.txt" --maxdim -1
expect_err "--maxdim 31" 2 "maxdim must be <= 30" $TT run "$TMP/sq.txt" --maxdim 31
expect_err "--maxdim 31 forced boundary" 2 "redzed: maxdim must be <= 30" \
    $TT run "$TMP/sq.txt" --maxdim 31 --mode boundary
expect_err "-k 0" 2 "k must be >= 1" $TT run "$TMP/sq.txt" --graph knn -k 0
expect_err "-k overflow" 2 "out of range" $TT run "$TMP/sq.txt" --graph knn -k 4294967296
expect_err "negative threshold" 2 "threshold must be non-negative" $TT run "$TMP/sq.txt" --threshold -1
expect_err "NaN threshold" 2 "threshold must be non-negative" $TT run "$TMP/sq.txt" --threshold nan
expect_err "unknown option" 2 "unknown option '--wat'" $TT run "$TMP/sq.txt" --wat
expect_err "missing value" 2 "missing value for --maxdim" $TT run "$TMP/sq.txt" --maxdim
expect_err "extra positional" 2 "unexpected argument" $TT run "$TMP/sq.txt" "$TMP/tri.txt"
expect_err "no input" 2 "missing input file" $TT run --maxdim 1
expect_err "unknown command" 2 "unknown command" $TT frobnicate
expect_err "bad --graph" 2 "--graph must be full or knn" $TT run "$TMP/sq.txt" --graph ring
expect_err "bad --metric" 2 "metric: unknown" $TT run "$TMP/sq.txt" --metric cosine
expect_err "bad lp" 2 "p must be in" $TT run "$TMP/sq.txt" --metric lp:0.5
expect_err "bad --mode" 2 "mode must be" $TT run "$TMP/sq.txt" --mode diagonal
expect_err "bad --engine" 2 "engine must be" $TT run "$TMP/sq.txt" --engine warp
expect_err "mode conflict" 2 "conflicts with an earlier" \
    $TT run "$TMP/sq.txt" --mode boundary --coboundary
expect_err "verify-recall w/o approximate" 2 "--verify-recall requires --approximate" \
    $TT run "$TMP/c.txt" --verify-recall 3
expect_err "approximate w/o knn" 2 "--approximate requires --graph knn" \
    $TT run "$TMP/c.txt" --approximate
expect_err "verify-recall w/o knn" 2 "--verify-recall requires --graph knn" \
    $TT run "$TMP/c.txt" --approximate --verify-recall 3
expect_err "test --ripser value" 2 "missing value for --ripser" $TT test --ripser
expect_err "gen bad kind" 2 "unknown cloud kind" $TT gen mobius 10 3 1 "$TMP/x.txt"
expect_err "gen torus dim" 2 "torus requires dim 3" $TT gen torus 10 4 1 "$TMP/x.txt"
expect_err "gen argc" 2 "gen <kind>" $TT gen torus 10 3
expect_err "unwritable output" 2 "cannot write" \
    $TT run "$TMP/sq.txt" --output "$TMP/no/such/dir/out.txt"
expect_err "bad byte budget" 2 "bad byte count" $TT run "$TMP/sq.txt" --memory-budget 12x

echo "cli smoke: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
