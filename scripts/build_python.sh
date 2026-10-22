#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build the turbo_tops Python extension in place (python/turbo_tops/).
#   PYTHON=python3 CXX=g++ PORTABLE=0 scripts/build_python.sh
# Fetches nanobind (pinned) into third_party/nanobind on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
CXX=${CXX:-c++}
NB_TAG=v2.2.0
NB=third_party/nanobind
if [ ! -d "$NB/.git" ]; then
    git clone --quiet --depth 1 --branch "$NB_TAG" --recurse-submodules --shallow-submodules \
        https://github.com/wjakob/nanobind.git "$NB"
fi
PY_INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
SUFFIX=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ARCH="-march=native"
[ "${PORTABLE:-0}" = "1" ] && ARCH=""
LINK=(-shared)
[ "$(uname)" = "Darwin" ] && LINK+=(-Wl,-undefined,dynamic_lookup)
mkdir -p build
NBI=(-isystem "$PY_INC" -isystem "$NB/include" -isystem "$NB/ext/robin_map/include")
"$CXX" -std=c++17 -O2 -fPIC -DNDEBUG -DNB_COMPACT_ASSERTIONS -fno-strict-aliasing \
    "${NBI[@]}" -c "$NB/src/nb_combined.cpp" -o build/nb_combined.o
"$CXX" -std=c++20 -O3 $ARCH -fPIC -fvisibility=hidden -DNDEBUG -pthread \
    -Iinclude -isystem third_party -isystem third_party/simde "${NBI[@]}" \
    -c python/turbotops_ext.cpp -o build/turbotops_ext.o
"$CXX" "${LINK[@]}" -pthread build/turbotops_ext.o build/nb_combined.o \
    -o "python/turbo_tops/_turbotops$SUFFIX"
echo "built python/turbo_tops/_turbotops$SUFFIX"
