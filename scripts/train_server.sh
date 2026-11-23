#!/usr/bin/env bash
set -e

DATA_ROOT="/ssdg/spl_yeongyoo/taf_dataset"
MANIFEST="metadata/manifest.parquet"

MID=$(uv run python -c "import pyarrow.parquet as pq; print(pq.read_table('$MANIFEST', columns=['manifest_id'])['manifest_id'][0].as_py()[:8])")

for N in 1 3 5 7 10 20; do
  OUT="outputs/comparison/runs/${MID}_N${N}_seed42"

  uv run trainctl run \
    --name "TaF-Adapter-N${N}-40k" \
    --checkpoint-dir "$OUT" \
    --total-steps 40000 \
    -- python -m script.train_adapter \
      comparison.enabled=true \
      comparison.manifest="$MANIFEST" \
      data.root_dir="$DATA_ROOT" \
      data.window_size="$N" \
      training.max_steps=40000 \
      training.seed=42 \
      training.batch_size=256 \
      training.gradient_accumulation_steps=1 \
      training.min_same_episode_window_gap=20 \
      training.num_workers=2 \
      training.prefetch_factor=1 \
      training.output_dir=outputs/comparison/runs \
      training.save_every=5000 \
      training.validate_every=2000 \
      wandb.group=window_40k
done