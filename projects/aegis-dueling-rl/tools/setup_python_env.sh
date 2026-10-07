#!/usr/bin/env bash
# Aegis — reproducible Python trainer environment (Apple Silicon / macOS).
# Creates CPython 3.10.12 venv at ~/venvs/aegis with ML-Agents 1.1.0. Idempotent.
#
# Why this isn't just `pip install mlagents`:
#   1. Homebrew's python@3.10 is 3.10.20, but mlagents needs Python <=3.10.12.
#      -> use uv to fetch an exact, prebuilt CPython 3.10.12 (no source compile).
#   2. mlagents 1.1.0 pins grpcio<=1.48.2, which has NO arm64 wheel and won't compile
#      on modern clang. mlagents runs fine on newer grpcio, so we OVERRIDE it to a
#      wheel-backed grpcio==1.62.2.
#
# Verified working 2026-06-15: torch 2.12.0, numpy 1.23.5, grpcio 1.62.2.
set -euo pipefail

VENV="$HOME/venvs/aegis"

command -v uv >/dev/null 2>&1 || brew install uv

echo "==> creating venv on CPython 3.10.12 at $VENV"
uv venv --seed --python 3.10.12 "$VENV"

OVERRIDES="$(mktemp)"
printf 'grpcio==1.62.2\n' > "$OVERRIDES"

echo "==> installing mlagents 1.1.0 (grpcio override -> wheel) + ML-Agents-native stack"
# Pin the stack ML-Agents 1.1.0 was tested against:
#   torch 2.2.1     -> ONNX export works WITHOUT onnxscript (newer torch needs it + protobuf 7)
#   protobuf 3.20.3 -> matches ML-Agents' generated _pb2 code (protobuf >=4 breaks it)
#   onnx 1.15.0     -> compatible with protobuf 3.20
# Do NOT install onnxscript — it pulls onnx 1.22/protobuf 7 and breaks ML-Agents comms.
uv pip install --python "$VENV/bin/python" --override "$OVERRIDES" \
    "numpy==1.23.5" "mlagents==1.1.0" "torch==2.2.1" "protobuf==3.20.3" "onnx==1.15.0"

rm -f "$OVERRIDES"

echo "==> verifying"
"$VENV/bin/mlagents-learn" --help >/dev/null
echo "OK: trainer installed. Activate with: source $VENV/bin/activate"
