#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run.sh CONFIG [train|test] [path.key=value ...]
config="${1:-configs/mvtec3d.py}"
mode="${2:-train}"
if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" -eq 1 ]; then shift; fi
python3 -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node:-1}" run.py -c "$config" -m "$mode" "$@"
