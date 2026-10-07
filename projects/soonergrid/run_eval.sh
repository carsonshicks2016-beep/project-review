#!/bin/bash
set -e
echo "Running benchmark with rl_marl policy included..."
PYTHONPATH=. python3 soonergrid/research/benchmark.py
echo "Building workspace..."
PYTHONPATH=. python3 build_workspace.py
echo "All done!"
