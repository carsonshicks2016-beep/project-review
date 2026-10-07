#!/usr/bin/env bash
set -euo pipefail

IBAMR_PREFIX="${IBAMR_PREFIX:-${HOME}/Applications/ibamr-research/install}"
CASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CASE_ROOT}/build"

cmake -S "${CASE_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${IBAMR_PREFIX}/packages" \
    -DIBAMR_DIR="${IBAMR_PREFIX}/packages/IBAMR-0.19.0/lib/cmake/ibamr" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target ibamr_navier_stokes_01_3d --parallel 4
