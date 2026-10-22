#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Fetch and build the ripser CLI (reference implementation used by
# `turbotops test` and the benchmarks) into third_party/ripser/.
#   CXX=g++ scripts/fetch_ripser.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=01add51ff64aaf40889483260cc5c3b7d0f2a1e7   # Ripser/ripser master, 2023
DIR=third_party/ripser
CXX=${CXX:-c++}
if [ ! -d "$DIR/.git" ]; then
    git clone --quiet https://github.com/Ripser/ripser.git "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$SHA" 2>/dev/null || true
git -C "$DIR" checkout --quiet --detach "$SHA"
"$CXX" -std=c++11 -O3 -D NDEBUG -o "$DIR/ripser" "$DIR/ripser.cpp"
echo "built $DIR/ripser ($SHA)"
