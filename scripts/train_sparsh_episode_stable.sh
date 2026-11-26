#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -gt 1 || ($# -eq 1 && $1 != --dry-run) ]]; then
  echo "Usage: bash scripts/train_sparsh_episode_stable.sh [--dry-run]" >&2
  exit 2
fi
gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
manifest_id=$(uv run python -c 'import pyarrow.parquet as pq; print(pq.ParquetFile("metadata/episode_split/manifest.parquet").read_row_group(0, columns=["manifest_id"])["manifest_id"][0].as_py()[:8])')
output="outputs/comparison/sparsh_episode_stable_runs/${manifest_id}_N5_seed42"
command=(uv run python -m script.terminal_monitor --metrics "$output/metrics.jsonl" --total-steps 10000 --
  uv run trainctl run --name SparshDINO-EpisodeSplit-Stable-N5 --checkpoint-dir "$output" --total-steps 10000 --
  uv run python -m script.train_adapter --config-name=taf_adapter_sparsh_episode_stable)
if [[ ${1:-} == --dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_uuid"
  printf '%q ' "${command[@]}"
  printf '\n'
else
  "${command[@]}"
fi
