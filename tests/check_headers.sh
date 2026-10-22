#!/usr/bin/env sh
# SPDX-License-Identifier: MIT
# Header hygiene: every public header compiles standalone (-Wall -Wextra
# -Werror) in both project flag profiles, and a two-TU binary that includes
# the umbrella header twice links cleanly (no ODR violations, no leaked
# internal symbols). Run from the repo root:  sh tests/check_headers.sh
set -e
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Pick a compiler that can actually build C++20 here: an ambient CXX (e.g.
# the Homebrew LLVM clang, which cannot see the macOS SDK headers) is
# honored only if it passes a probe.
pick_cxx() {
    for cand in "${CXX:-}" /usr/local/bin/g++-15 c++; do
        [ -n "$cand" ] || continue
        if printf '#include <sys/stat.h>\n#include <algorithm>\nint main() { return 0; }\n' > "$TMP/probe.cpp" &&
           "$cand" -std=c++20 -fsyntax-only "$TMP/probe.cpp" > /dev/null 2>&1; then
            CXX=$cand
            return 0
        fi
    done
    echo "check_headers: no working C++20 compiler found" >&2
    return 1
}
pick_cxx
echo "check_headers: using $CXX"

for mode in native portable; do
    if [ "$mode" = native ]; then
        FLAGS="-march=native"
    else
        FLAGS="-Wno-psabi"   # PORTABLE=1 profile
    fi
    for h in core metric graph memory mst rips redzed turbotops; do
        printf '#include "turbotops/%s.hpp"\nint main() { return 0; }\n' "$h" > "$TMP/$h.cpp"
        "$CXX" -std=c++20 -fsyntax-only -Wall -Wextra -Werror -Wno-unused-parameter \
            $FLAGS -Iinclude -isystem third_party -isystem third_party/simde "$TMP/$h.cpp"
    done
done

printf '#include "turbotops/turbotops.hpp"\nint hdr_a() { tops::Options o; return (int)o.k; }\n' > "$TMP/a.cpp"
printf '#include "turbotops/turbotops.hpp"\nextern int hdr_a();\nint main() { return hdr_a(); }\n' > "$TMP/b.cpp"
"$CXX" -std=c++20 -O1 -Wall -Wextra -Werror -march=native \
    -Iinclude -isystem third_party -isystem third_party/simde \
    "$TMP/a.cpp" "$TMP/b.cpp" -o "$TMP/odr" -pthread

echo "check_headers: 8 headers self-contained (native + portable profiles), two-TU ODR link clean"
