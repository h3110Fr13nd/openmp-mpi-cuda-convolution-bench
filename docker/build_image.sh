#!/usr/bin/env bash
set -euo pipefail

docker build -t hpc-omp-offload -f docker/Dockerfile.openmp-offload docker
