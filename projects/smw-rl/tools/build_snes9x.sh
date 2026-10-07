#!/usr/bin/env bash
# Rebuild the snes9x libretro core for Apple Silicon.
#
# stable-retro's prebuilt arm64 core is compiled big-endian (see
# smwrl/setup_core.py for the full explanation) and renders a black screen
# forever. This rebuilds the same sources from the stable-retro sdist with
# -DARM, which selects the LSB_FIRST / FAST_LSB_WORD_ACCESS paths in
# cores/snes/port.h, and drops the result in vendor/.
#
#   bash tools/build_snes9x.sh
#
# Requires: Xcode Command Line Tools, curl, python3.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/smwrl-snes9x-build"
VERSION="${STABLE_RETRO_VERSION:-1.0.1}"

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d "sr" ]; then
  echo "==> fetching stable-retro $VERSION sources"
  URL=$(python3 - "$VERSION" <<'PY'
import json, sys, urllib.request
version = sys.argv[1]
with urllib.request.urlopen("https://pypi.org/pypi/stable-retro/json") as fh:
    data = json.load(fh)
for f in data["releases"][version]:
    if f["filename"].endswith(".tar.gz"):
        print(f["url"]); break
else:
    raise SystemExit(f"no sdist for stable-retro {version}")
PY
)
  curl -sL "$URL" -o sr.tar.gz
  mkdir -p sr && tar xzf sr.tar.gz -C sr --strip-components=1
fi

echo "==> building snes9x for arm64 (little-endian)"
cd "$WORK/sr/cores/snes/libretro"
make clean >/dev/null 2>&1 || true
export MACOSX_DEPLOYMENT_TARGET=11.0
export CFLAGS="-DARM -O2 -fno-strict-aliasing"
export CXXFLAGS="-DARM -O2 -fno-strict-aliasing"
make platform=osx -j"$(sysctl -n hw.ncpu)" >/dev/null

mkdir -p "$ROOT/vendor"
cp snes9x_libretro.dylib "$ROOT/vendor/snes9x_libretro.dylib"
echo "==> wrote $ROOT/vendor/snes9x_libretro.dylib"

echo "==> installing into the active venv and verifying"
cd "$ROOT"
python -m smwrl.setup_core
