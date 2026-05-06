#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm --gpus all \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  hpc-omp-offload \
  bash -lc "set -euo pipefail; \
    NVPTX_BC=\"\"; \
    for bc in /opt/llvm/lib/libomptarget-nvptx-sm_*.bc; do \
      if [[ -f \"$bc\" ]]; then NVPTX_BC=$bc; break; fi; \
    done; \
    if [[ -n \"$NVPTX_BC\" ]]; then export OPENMP_NVPTX_BC_PATH=\"$NVPTX_BC\"; fi; \
    USE_CLANG=1 ./scripts/build_all; \
    python3 benchmarks/scripts/run_benchmarks.py"
