#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
  dry_run=true
  shift
fi
if [[ $# == 0 ]]; then set -- 1 3 5 7 10 20; fi
gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
manifest_id=$(uv run python -c 'import pyarrow.parquet as pq; print(pq.ParquetFile("metadata/manifest.parquet").read_row_group(0,columns=["manifest_id"])["manifest_id"][0].as_py()[:8])')
for n in "$@"; do
  case "$n" in 1|3|5|7|10|20) ;; *) echo "Unsupported N: $n" >&2; exit 2;; esac
  output="outputs/comparison/sparsh_runs/${manifest_id}_N${n}_seed42"
  command=(uv run python -m script.terminal_monitor --metrics "$output/metrics.jsonl" --total-steps 10000 --
    uv run trainctl run --name "SparshDINO-N${n}" --checkpoint-dir "$output" --total-steps 10000 --
    uv run python -m script.train_adapter --config-name=taf_adapter_sparsh "data.window_size=$n")
  if "$dry_run"; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_uuid"
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done
