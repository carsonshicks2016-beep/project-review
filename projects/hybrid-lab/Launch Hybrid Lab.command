#!/bin/zsh
cd "$(dirname "$0")" || exit 1
exec python3 -m hybrid_lab --open
