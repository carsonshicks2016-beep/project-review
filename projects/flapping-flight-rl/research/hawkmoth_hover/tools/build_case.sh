#!/usr/bin/env bash
set -euo pipefail

IBAMR_PREFIX="${IBAMR_PREFIX:-${HOME}/Applications/ibamr-research/install}"
BUILD_DIR="${BUILD_DIR:-$(cd "$(dirname "$0")/.." && pwd)/build}"
CASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmake -S "${CASE_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${IBAMR_PREFIX}/packages" \
    -DIBAMR_DIR="${IBAMR_PREFIX}/packages/IBAMR-0.19.0/lib/cmake/ibamr" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel 4
