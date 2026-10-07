#!/usr/bin/env bash
set -euo pipefail

# Reproducible local macOS install for this benchmark. Build products stay
# outside the repository. Override IBAMR_PREFIX/AUTOIBAMR_DIR to relocate them.
AUTOIBAMR_DIR="${AUTOIBAMR_DIR:-${HOME}/Applications/ibamr-research/autoibamr}"
IBAMR_PREFIX="${IBAMR_PREFIX:-${HOME}/Applications/ibamr-research/install}"
AUTOIBAMR_TAG="v0.19.0"
AUTOIBAMR_COMMIT="ea833cb1dc1d1c7d6a1842048db62d12b5e2cb05"
IBAMR_COMMIT="a0a8d0a4dc5deb6f576a5837b211baf0d99f5df6"

if [[ ! -d "${AUTOIBAMR_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${AUTOIBAMR_DIR}")"
    git clone --depth 1 --branch "${AUTOIBAMR_TAG}" https://github.com/IBAMR/autoibamr.git "${AUTOIBAMR_DIR}"
fi

actual_commit="$(git -C "${AUTOIBAMR_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${AUTOIBAMR_COMMIT}" ]]; then
    echo "autoibamr commit mismatch: expected ${AUTOIBAMR_COMMIT}, found ${actual_commit}" >&2
    exit 2
fi

# CMake's own test configuration may auto-detect an incomplete Homebrew Qt5
# package on macOS. Disable the optional GUI and Qt test probes in that package.
python3 - "${AUTOIBAMR_DIR}/IBAMR-toolchain/packages/cmake.package" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'CONFOPTS="-DCMAKE_USE_OPENSSL=OFF"'
new = 'CONFOPTS="-DCMAKE_USE_OPENSSL=OFF -DBUILD_QtDialog=OFF -DCMake_TEST_Qt5=OFF"'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit(f"unrecognized cmake.package configure line: {path}")
PY

export CC="$(command -v mpicc)"
export CXX="$(command -v mpicxx)"
export FC="$(command -v mpifort)"
"${AUTOIBAMR_DIR}/autoibamr.sh" \
    --ibamr-version 0.19.0 \
    --disable-libmesh \
    --prefix="${IBAMR_PREFIX}" \
    --jobs=4 \
    --yes

mkdir -p "${IBAMR_PREFIX}/share/hawkmoth-hover"
{
    printf 'Captured UTC: '
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    printf '\n--- host ---\n'
    sw_vers
    uname -a
    printf '\n--- compilers and MPI ---\n'
    "$(command -v cc)" --version | head -n 1
    "$(command -v c++)" --version | head -n 1
    "$(command -v gfortran)" --version | head -n 1
    "$(command -v mpirun)" --version | head -n 3
    printf '\n--- dependency manager versions ---\n'
    brew list --versions open-mpi gcc cmake 2>/dev/null || true
    printf '\n--- pinned sources ---\n'
    printf 'IBAMR tag: v0.19.0\nIBAMR source commit: %s\n' "${IBAMR_COMMIT}"
    printf 'autoibamr commit: %s\n' "${AUTOIBAMR_COMMIT}"
} > "${IBAMR_PREFIX}/share/hawkmoth-hover/build-environment.txt"
printf 'Saved build environment to %s\n' "${IBAMR_PREFIX}/share/hawkmoth-hover/build-environment.txt"
