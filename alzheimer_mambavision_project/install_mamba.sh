#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-build-isolation --no-deps --only-binary=:all: \
  --extra-index-url https://wheels.astral.sh/simple/cu130/ \
  "mamba-ssm==2.3.2.post1+cu.13.0.torch.2.11" \
  "causal-conv1d==1.6.2.post1+cu.13.0.torch.2.11"

python -m pip install --no-deps "mambavision==1.2.0"
