#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

CASE="${1:-compare}"
DATA_ROOT="${TAF_DATA_ROOT:-/ssdg/spl_yeongyoo/taf_dataset}"
MANIFEST="${TAF_MANIFEST:-metadata/manifest.parquet}"
OUT="outputs/dataloader_bench/$(date +%Y%m%d_%H%M%S)_${CASE}_$$"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
COMMON=(--data-root "$DATA_ROOT" --manifest "$MANIFEST" --output-dir "$OUT"
        --window-sizes 1 --batch-size 256 --steps 10 --warmup-steps 2
        --prefetch-factor 1 --sensor-cache-mb 256 --seed 42)
case "$CASE" in
  baseline) EXTRA=(--mode baseline --cache off --workers 2) ;;
  optimized) EXTRA=(--mode optimized --cache off --workers 2 --group-reads) ;;
  compare) EXTRA=(--mode both --cache off --workers 2 --group-reads) ;;
  cache) EXTRA=(--mode optimized --cache both --workers 2 --group-reads) ;;
  workers) EXTRA=(--mode both --cache off --workers 2 4 8 --group-reads) ;;
  estimate) EXTRA=(--estimate-only) ;;
  sensor-compare) EXTRA=(--mode optimized --cache off --workers 4 --group-reads --sensor-mmap compare) ;;
  sensor-warm) EXTRA=(--mode optimized --cache off --workers 4 --group-reads --sensor-mmap on) ;;
  sensor-build) EXTRA=(--sensor-mmap-build-only) ;;
  *) printf 'Usage: %s {baseline|optimized|compare|cache|workers|estimate|sensor-compare|sensor-warm|sensor-build}\n' "$0" >&2; exit 2 ;;
esac
shift "$(( $# > 0 ? 1 : 0 ))"
printf 'Benchmark output: %s/metrics.jsonl\n' "$OUT"
exec uv run --no-sync python -m script.profile_throughput "${COMMON[@]}" "${EXTRA[@]}" "$@"
