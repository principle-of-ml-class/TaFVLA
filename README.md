# TaF-Adapter pre-training

This repository implements only the tactile-force alignment adapter from
[TaF-VLA](https://arxiv.org/abs/2601.20321), Section IV-A and Eqs. (1)--(11).
It does not implement the VLA policy, action prediction, or downstream robot
training.

## Architecture

- `model/force_encoder.py`: independent pressure and 6-axis wrench MLP
  encoder/VQ/decoder branches. A complete temporal window is flattened before
  each MLP, and the two quantized vectors are concatenated.
- `model/vector_quantizer.py`: Euclidean nearest-neighbor lookup, the standard
  VQ-VAE codebook and commitment losses, and a straight-through estimator.
- `model/tactile_encoder.py`: framewise pretrained ViT-Small followed by a
  causal temporal Transformer. The learnable summary token is last, so its
  causal-attention row can attend to all tactile frames.
- `model/taf_adapter.py`: normalization, both encoders, Eq. (11), metrics, and
  the tactile-only `encode_tactile()` inference API.
- `model/losses.py`: cosine InfoNCE and retrieval recall.
- `utils/config.py`: Hydra/OmegaConf structured schema. Unknown keys and type
  mismatches fail during config composition.
- `utils/common.py`: seed setup, semantic config validation, and Hydra-safe
  path resolution.
- `utils/model_factory.py`: the single model-construction path shared by
  training and validation.
- `utils/data.py`: leakage-free grouped split, temporal audit, train-only
  normalization, and the unchanged temporal-diverse batch sampler.
- `utils/checkpoint.py`: atomic checkpoint save/load and RNG restoration.
- `utils/logging.py`: the shared console, JSONL, and optional W&B logger.

For the default `T=5`, the main shapes are:

```text
pressure (B,5,12,12) -> z_pressure (B,128) -> VQ -> reconstruction (B,5,12,12)
force       (B,5,6)  -> z_force    (B,128) -> VQ -> reconstruction (B,5,6)
concat(two VQ vectors)                                      -> c_force (B,256)
tactile (B,5,3,H,W) -> ViT frames (B,5,384) -> projection/causal TF -> z_tactile (B,256)
```

The default backbone is
`vit_small_patch16_224.augreg_in21k_ft_in1k`, verified against the installed
`timm==1.0.30` pretrained registry. It uses AugReg ImageNet-21k pretraining and
ImageNet-1k fine-tuning, a 384-dimensional output, 224x224 inputs, and mean/std
of 0.5 per RGB channel. Pretrained-load errors are propagated; the code never
silently falls back to random initialization. The backbone is frozen by
default, while projection and temporal Transformer layers remain trainable.

## Objectives and gradient flow

The forward value of each straight-through VQ vector is the nearest codebook
entry. Its backward derivative is routed to the force encoder. Therefore,
reconstruction and InfoNCE gradients train the force encoders; reconstruction
also trains each decoder; InfoNCE trains the tactile projection/Transformer;
and the explicit codebook term trains each independent codebook. The
commitment term trains the force encoders. A frozen ViT receives no gradients.

`loss_total = loss_nce + lambda_quant * loss_quant + lambda_recon * loss_recon`.
The default weights are all 1.0 and commitment beta is 0.25. These are initial
experimental choices, not values reported by the paper, and are kept explicit
so loss scales can be inspected before changing them. Reconstruction uses an
element-wise mean MSE independently for normalized pressure and force, another
documented implementation choice because the paper does not publish its
reduction convention.

The paper calls InfoNCE symmetric in prose, but Eq. (10) contains only the
tactile-to-force direction. `nce_mode: paper` implements that equation;
`nce_mode: symmetric` averages tactile-to-force and force-to-tactile losses.

## Hydra configuration

Both training and model validation compose `config/taf_adapter.yaml` through
Hydra and receive an OmegaConf `DictConfig`. The YAML is merged with a small
structured schema, so misspelled keys and incompatible types produce an error.
Semantic checks additionally reject invalid window sizes, learning rates, loss
weights, temperatures, split fractions, and logging intervals.

Inspect the fully composed job config without starting training:

```bash
uv run python -m script.train_adapter --cfg job
```

Any value can be overridden without editing YAML:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  data.window_size=10 \
  training.batch_size=16 \
  training.learning_rate=1e-4 \
  loss.temperature=0.07
```

Hydra is configured with `hydra.job.chdir: false`. Relative dataset,
checkpoint, and training output paths are resolved against the directory from
which the command was launched. Hydra's own run logs go below
`outputs/hydra/`, while model checkpoints and `metrics.jsonl` retain the
independent `training.output_dir` setting (default:
`outputs/taf_adapter`). The VQ commitment coefficient has one source of truth:
`vq.commitment_beta`.

## Validation

Install/synchronize dependencies and run the model tests with:

```bash
uv sync
uv run python -m script.check_model
```

The check covers T=1, 3, 5, and 10; dimensions and reconstructions; finite
losses; backward flow through both force encoders, both codebooks, both
decoders, and the temporal Transformer; absence of frozen-ViT gradients;
codebook index bounds; causal behavior and summary-token access; and a fixed
eight-pair synthetic overfit. The former `--no-pretrained` and
`--skip-overfit` flags are now `check.no_pretrained=true` and
`check.skip_overfit=true`. The overfit duration can be changed with
`check.overfit_steps=20`.

Fast refactor/config/checkpoint/logger tests run with:

```bash
uv run python -m unittest discover -s tests -v
```

## Dataset gate and training

Before training, inspect a downloaded dataset directly:

```bash
uv run python -m script.check_dataset \
  --root-dir /path/to/TaF-Dataset \
  --test-window-sizes
```

Then start pre-training:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset
```

`TaFDataset` validates episode index ranges and every Parquet-to-MP4 timestamp
match during construction. The trainer additionally decodes representative
windows from every episode and refuses to proceed if an invalid sequence was
skipped. Train/validation splitting happens at sequence or episode granularity,
never at window granularity.

### Persistent video timestamp cache

Dataset construction still obtains every presentation timestamp with the exact
`ffprobe -show_frames` strategy used for temporal alignment; timestamps are
never estimated from FPS or frame number. The first request for each
metadata-referenced MP4 stores the ordered `float64` result under
`${XDG_CACHE_HOME:-~/.cache}/taf-vla/video_timestamps/`. A later run validates
the canonical absolute path, file size, nanosecond modification time, cache
format version, extraction-strategy version, array shape/dtype/finiteness,
strict ordering, and an array checksum before accepting a hit.

Each video is committed immediately as one `.npz` via a temporary file and
atomic rename. A per-video file lock makes concurrent initializers re-check the
finished cache instead of reading partial data. Consequently, Ctrl+C preserves
all videos completed before the interrupted probe. Stale or corrupt entries are
regenerated independently and do not require clearing the whole cache. The
source dataset is never modified.

The initialization UI counts only MP4s referenced by the selected LeRobot
metadata. On an interactive TTY it uses fixed-position tqdm rows for sequences,
videos, elapsed time/ETA, and cache hit/miss/failure counts. Non-TTY jobs emit
sparse plain-text summaries without ANSI cursor control. Configure this through
Hydra:

```yaml
data:
  timestamp_cache_enabled: true
  timestamp_cache_dir: null
  show_initialization_progress: true
```

`script.check_dataset` reuses the same default cache. Its corresponding flags
are `--[no-]timestamp-cache-enabled`, `--timestamp-cache-dir`, and
`--[no-]show-initialization-progress`. Disabling the cache runs the same
ffprobe extraction every time.

### Training throughput controls

Sensor Parquet values use a process-local LRU cache bounded by
`data.sensor_cache_max_mb` per DataLoader worker. `TaFDataset.sensor_cache_stats()`
reports reads, hits, misses, hit rate, and resident bytes for the calling
process. Pass `sensor_cache_shared_stats=True` when constructing a dataset to
collect worker totals with `sensor_cache_total_stats()`; this adds a small
counter-lock cost and is used by the throughput profiler.

Training DataLoaders keep images as uint8 until they reach the model, reducing
their host-memory footprint by four times. `training.prefetch_factor` controls
queued batches when `training.num_workers > 0`; for N=20, start with 1 and
measure before raising workers or prefetch. The optional raw-image
`tactile_encoder.backbone_chunk_frames` setting limits the number of frames
converted to float32 and passed to the frozen ViT together. Keep the batch
size required by the contrastive protocol; changing it changes the negative
set.

`data.frame_cache_enabled=true` precomputes frozen ViT outputs for each unique
video frame. The trainable projection and temporal encoder still run on every
step. Cache keys include backbone weights, image preprocessing, video identity,
and aligned frame timestamps. The frame set is independent of N, so runs for
N=1, 3, 5, 7, 10, and 20 can reuse it. Video or backbone changes invalidate
the relevant entries. `data.frame_cache_rebuild=true` forces regeneration;
`data.frame_cache_dir` selects the storage location. The default raw-image path
remains available by leaving the feature cache disabled. Alignment retrieval
continues to decode raw images and uses the same validation selection.

To measure raw and cached training on the same sampled windows, with N=5 and
N=20 by default:

```bash
uv run python -m script.profile_throughput \
  --data-root /path/to/taf_dataset --batch-size 8 --steps 5
```

The profiler reports data loading, video decode and frozen ViT sample times,
training step and validation time, GPU peak memory (when CUDA is available),
process RSS, steps/s, frames/s, cache build time, and raw/cached feature error.
Its measured speedup excludes the one-time cache build; use the reported build
time when assessing short runs. `--no-pretrained` measures the same architecture
with random backbone weights when pretrained weights are unavailable.

Some TaF-Dataset distributions omit `meta/episodes` for otherwise complete
sequences. The loader uses official episode metadata whenever it exists. If it
is missing, a deliberately narrow fallback supports only one data Parquet and
one `observation.image` MP4. It reconstructs boundaries in memory only after
checking contiguous global indices, episode IDs, per-episode frame indices,
timestamps, FPS, `total_frames`, `total_episodes`, and `info.json` path
templates. The normal ffprobe timestamp tolerance and one-to-one frame mapping
checks still run afterward. A warning identifies every reconstructed sequence;
no file is added to or changed in the source dataset.

Pressure mean/std is stored per 12x12 taxel and wrench mean/std per ordered 6D
component. Statistics use unique frames from the training groups only, not
overlapping windows and never validation data. They are stored both in model
buffers and checkpoint metadata. Training uses the shuffled
`TemporalDiverseBatchSampler`. Validation reuses its compatibility rule through
a fixed-seed evaluation sampler: every emitted batch is full sized, and windows
from the same episode have starts separated by at least
`max(data.window_size, training.min_same_episode_window_gap)`. The fixed batch
plan and order are reused at every validation call. Candidate, used, and
dropped sample counts plus the number of violating batches are written to
`metrics.jsonl` and printed at startup; failure to form one valid full batch is
an error rather than a silent batch-size change. This reduces adjacent false
negatives, but cannot eliminate temporally distant samples with the same
semantic or physical state. Gradient accumulation also does not enlarge the
InfoNCE negative set: only pairs in one forward pass are negatives.

Checkpoints contain model/optimizer/scaler state, normalization, split groups,
config, step/epoch, alignment-evaluation state, and RNG state. Older checkpoints
without alignment metadata remain loadable. Resume with:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  training.resume=outputs/taf_adapter/last.pt
```

Metrics are appended to `outputs/taf_adapter/metrics.jsonl` by default. An
interactive terminal shows optimizer progress and a fixed training/validation
dashboard instead of printing each JSON dictionary.

## Weights & Biases

W&B is disabled by default and therefore requires neither authentication nor a
network connection. Authenticate separately with `wandb login` or the
`WANDB_API_KEY` environment variable, then opt in:

```bash
uv run wandb login
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  training.max_steps=1000 \
  wandb.enabled=true
```

For a network-free local W&B run:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  wandb.enabled=true \
  wandb.mode=offline
```

The final resolved Hydra config and the W&B run ID/name are recorded. Training
and validation log total, InfoNCE, quantization, codebook, commitment, pressure
and force reconstruction losses; positive/negative cosine; recall@1/5; and
pressure/force codebook perplexity. Detailed codebook used count, usage
fraction, and dead-code count are intentionally excluded from W&B, but the
model still computes them and the unfiltered values remain in `metrics.jsonl`.
W&B also receives learning rate, gradient norm, epoch, elapsed time, and the
actual optimizer step as its global step. No additional model forward pass is
made for logging, and datasets/checkpoints are not uploaded as artifacts.

W&B's Python SDK can define a common step metric, but does not reliably impose
a particular multi-run workspace panel layout. The logger therefore uploads
each native scalar once and never re-uploads the complete history as a custom
chart. In a run workspace, use **Add panel → Line plot**, choose
`global_step` for X, and select these Y series:

- **Training Loss Components**: `train/loss_total`, `train/loss_nce`,
  `train/loss_quant`, `train/loss_recon`
- **Validation Loss Components**: `val/loss_total`, `val/loss_nce`,
  `val/loss_quant`, `val/loss_recon`
- **Training Codebook Perplexity**: `train/pressure_codebook_perplexity`,
  `train/force_codebook_perplexity`
- **Validation Codebook Perplexity**: `val/pressure_codebook_perplexity`,
  `val/force_codebook_perplexity`
- **Cross-Modal Retrieval**: `alignment/t2f/recall_at_1`,
  `alignment/t2f/recall_at_5`, `alignment/t2f/mrr`,
  `alignment/t2f/hard_negative_margin` with `global_step` as X

Alignment images are logged under `alignment/visualization/*` with both
`completed_epoch` and the actual optimizer `global_step` in their record and
caption.

Each panel overlays the listed series on one chart. The `define_metric` calls
bind both `train/*` and `val/*` histories to the same monotonic `global_step`.

`training.log_every` controls terminal dashboard and JSONL aggregation,
`training.validate_every` controls validation, and `wandb.log_every` controls
training metric uploads to W&B. Validation uses its current optimizer step and
is merged into the same W&B log call when the intervals coincide. A checkpoint
stores the W&B run ID; resume continues that run when present, while older
checkpoints without the field start a new run. Online initialization errors are
reported rather than silently changing mode. Normal and exceptional shutdowns
call `wandb.finish()` after successful initialization.

On a TTY, tqdm reserves one progress row and eight fixed metric rows for loss,
validation, and cross-modal retrieval. Progress
counts completed optimizer updates, so gradient-accumulation micro-steps do not
advance it; resume starts at the checkpoint step. Elapsed time, ETA, completion
percentage, and steps/second are shown by the progress row. In non-TTY contexts
(redirected output, CI, or a terminal multiplexer without usable TTY support),
ANSI cursor movement is disabled and a plain-text summary is emitted only at
the configured logging/validation intervals.

## trainctl Discord training notifications

The project pins trainctl `0.1.0` to the verified upstream commit
`61d3d61d975086ee1bff6a6bdc79cc4d568cebc5` with its `discord` extra. A normal
locked install is therefore sufficient:

```bash
uv sync --locked
```

Create a Discord incoming webhook and expose it only through the environment.
Do not put the URL in YAML, shell history shared with others, checkpoints, W&B,
or committed `.env` files:

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL="<YOUR_WEBHOOK_URL>"
uv run trainctl config show
```

`config show` reports only whether the webhook is configured. It does not print
the credential. The verified trainctl CLI invocation is:

```bash
uv run trainctl run \
  --name taf-adapter-T5 \
  --checkpoint-dir outputs/taf_adapter \
  --total-steps 25000 \
  -- python -m script.train_adapter \
    data.window_size=5 \
    training.max_steps=25000 \
    training.output_dir=outputs/taf_adapter
```

Add `data.root_dir=/path/to/TaF-Dataset` when the configured dataset path is not
valid on the current machine. Hydra overrides are resolved before the start
annotation, so `data.window_size`, batch/accumulation, learning rate, schedule,
and output directory shown in Discord are the values actually used.

trainctl emits its automatic process started/completed/failed events. The
training SDK additionally emits one detailed start annotation, structured
metrics at `training.log_every`, validation and alignment metrics when those
evaluations actually run, a concise event after each successfully written
checkpoint, and one detailed successful-finish annotation. Metric logging does
not create a Discord message for every step. Every numbered checkpoint is
validated at its own optimizer step; an already completed validation at that
step is reused. Alignment values are attached only when fixed-set retrieval ran
at that same step. `last.pt` is also validated at its final step and registered
directly, since `step_00001000.pt` is intentionally not renamed to match the
default filename watcher.

The SDK is non-strict, defaults to rank 0 under `RANK`/`LOCAL_RANK`, and becomes
a no-op when the command is not launched by `trainctl run`. The existing direct
workflow remains valid and produces the same training artifacts:

```bash
uv run python -m script.train_adapter \
  data.window_size=5 \
  training.output_dir=outputs/taf_adapter
```

On Python exceptions or `Ctrl+C`, the SDK adds a short failure annotation and
the original exception is re-raised so the supervisor records a failed run.
Abrupt termination such as `SIGTERM` cannot produce the success annotation;
the supervisor still reports the nonzero/terminated process. Exception text is
credential-redacted, and finite numeric filtering prevents NaN/Inf values from
entering trainctl. Never print or pass the webhook URL as a Hydra override.

### Optional training performance telemetry

`trainctl.performance.enabled=true` records training throughput, DataLoader
wait, validation/retrieval/checkpoint duration, and process memory every
`trainctl.performance.log_every` optimizer steps. The values go to
`metrics.jsonl`, the existing trainctl `tracker.log()` path, and W&B under the
same `perf/*` names when W&B is enabled. `performance_summary.jsonl` receives
one structured row when a run finishes. No separate notifier is used.

Default telemetry uses CPU wall-clock time for waiting, throughput, and
evaluation. It does not synchronize CUDA every step. Enable
`trainctl.performance.detailed_profiling=true` to time transfer, ViT, model
forward, backward, and optimizer stages. On CUDA these use events and one
synchronization at each reporting boundary; on CPU their names end in
`_cpu_ms` to identify wall-clock measurements. Absent or unreliable measures
are omitted. `monitor_cache=true` enables shared worker counters for Parquet
read time and cache hit rate; the feature-cache rate describes reuse of
precomputed video files when the cache was built. `monitor_gpu_utilization=true`
tries the local PyTorch utilization API and omits the metric if unavailable.

Warnings are off by default. Enable `trainctl.performance.warnings.enabled`
to check data-wait share, configured CPU/GPU memory thresholds, cache hit rate,
and optimizer-step stalls. Set `send_to_trainctl=true` to send rate-limited
warnings through the existing trainctl notification API. Validation and
checkpoint phases pause the stall timer.

For N comparisons, collect each finished run's `performance_summary.jsonl`:

```bash
uv run python -m script.compare_performance \
  --summaries outputs/N1/performance_summary.jsonl outputs/N3/performance_summary.jsonl \
              outputs/N5/performance_summary.jsonl outputs/N7/performance_summary.jsonl \
              outputs/N10/performance_summary.jsonl outputs/N20/performance_summary.jsonl \
  --output-dir outputs/performance_comparison
```

The comparison refuses mixed hardware or training protocols, marks missing N values, and draws only
graphs supported by measured data. Offline dynamics retrieval and force-probe
commands record their measured durations in their JSON reports and append
matching events to the training summary when it exists. They also log those
`perf/*` durations to the resumed W&B run when W&B is enabled. Measure the
passive monitor itself with `python -m script.benchmark_monitoring`.

## Cross-modal alignment evaluation

Training uses mini-batch InfoNCE diagnostics, while the alignment evaluator uses
a separate, fixed validation candidate set. It extracts the model's
`z_tactile` and `c_force` under `model.eval()` and `torch.inference_mode()`, L2
normalizes both, and computes the complete cosine matrix
`S = z_tactile @ c_force.T`. Metrics are reported for both tactile→force and
force→tactile:

- Recall@1 and Recall@5 over the complete fixed candidate set
- mean reciprocal rank (MRR)
- mean matched-pair and non-matched-pair cosine similarity
- mean matched similarity minus the hardest negative similarity

Only validation indices produced by the existing group split are eligible.
Within one episode, selected window starts are separated by at least
`max(alignment_eval.min_window_start_gap, data.window_size)`. Episode queues are
sampled round-robin with a local fixed seed, and the exact global indices,
episode identifiers, starts, effective gap, and seed are saved to
`outputs/taf_adapter/alignment/validation_selection.json`. This removes direct
frame overlap, but temporally distant windows can still represent the same
physical force state. Retrieval scores must therefore not be interpreted as a
perfect semantic-duplicate-free benchmark.

The default configuration is:

```yaml
alignment_eval:
  enabled: true
  retrieval_every_n_steps: 1000
  retrieval_every_n_epochs: 1
  visualize_every_n_epochs: 5
  retrieval_max_pairs: 256
  visualization_max_pairs: 64
  min_window_start_gap: 5
  random_seed: 42
  run_on_training_end: true
  save_local_figures: true
```

Options are regular Hydra overrides:

```fish
uv run python -m script.train_adapter \
    data.root_dir=/path/to/TaF-Dataset \
    alignment_eval.visualize_every_n_epochs=10

uv run python -m script.train_adapter \
    data.root_dir=/path/to/TaF-Dataset \
    alignment_eval.enabled=false
```

An epoch metric is emitted only after every training batch for that epoch was
consumed. Step retrieval is based on completed optimizer updates, not gradient
accumulation micro-steps; `retrieval_every_n_steps=0` disables only this step
schedule. Step evaluations reuse the same saved validation pairs and skip UMAP
and Matplotlib. `completed_epoch` is 1-based and is stored in checkpoints. If a
step trigger lands on an epoch boundary, the single evaluation is epoch-labeled
so the epoch visualization schedule is preserved. If `max_steps` stops inside
an epoch, `run_on_training_end=true` performs a retrieval-only record labeled
`final` unless that exact model/global step was already evaluated. Step, epoch,
and final triggers therefore never encode the same model twice.

The ordinary `val/*` metrics and fixed 256-pair `alignment/*` metrics are
different protocols. Ordinary validation computes the unchanged training loss
over deterministic, full-size temporally diverse mini-batches (up to
`training.validation_batches`, still 50 by default). Alignment evaluation
embeds its separate fixed subset and performs complete-candidate bidirectional
retrieval. Neither protocol replaces the other.

To compare legacy contiguous validation batches with the new sampler at an
existing checkpoint without an optimizer update, run:

```bash
uv run python -m script.evaluate_validation_sampling \
  data.root_dir=/path/to/TaF-Dataset \
  training.resume=outputs/taf_adapter/step_00020000.pt
```

The command prints and saves
`outputs/taf_adapter/validation_sampling_comparison.json`, including both
protocol descriptions, metrics, evaluated sample counts, dropped samples, and
`"training_performed": false`.

Every visualization interval produces:

- `epoch_XXXX_similarity.png`: tactile rows × force/pressure columns, fixed
  cosine range `[-1, 1]`; matched pairs lie on the diagonal.
- `epoch_XXXX_umap.png`: one joint UMAP fit over both modalities, with matching
  indices connected. Each epoch refits UMAP, so coordinates across epochs are
  not a quantitative distance measure.
- `epoch_XXXX_distribution.png`: density-normalized positive, negative, and
  hard-negative cosine distributions with positive/negative means.

PNG files are saved under `outputs/taf_adapter/alignment/` and the same rendered
images are uploaded once at their evaluation step when W&B is enabled. UMAP is
CPU-side and requires `umap-learn`; install a missing dependency with
`uv add umap-learn`. Runtime is dominated by encoding at most
`retrieval_max_pairs` windows, an `N×N` cosine matrix, and periodic UMAP over at
most twice `visualization_max_pairs` points. Embeddings are moved to CPU and all
Matplotlib figures are closed after rendering.

## Reproduction boundary

Directly reproduced from the paper: two independent force modalities and
codebooks, nearest-neighbor VQ, concatenation, VQ-VAE loss, reconstruction,
framewise ViT features, temporal causal Transformer with summary token, cosine
InfoNCE, Eq. (11), and the reported `T=5`.

Project choices because the paper does not specify them: exact ViT checkpoint
and freeze policy, MLP depth/width, latent/codebook sizes, Transformer details,
normalization and reconstruction reduction, optimizer/schedule settings, loss
weights/temperature, and whether to use the prose-symmetric rather than
equation-directional InfoNCE. All are visible in `config/taf_adapter.yaml`.
# Interaction dynamics × temporal window 비교 실험

이 실험은 명시적으로 `comparison.enabled=true`일 때만 켜집니다. 기존 학습은 그대로 실행할 수 있습니다. 모델 입력 N은 `[1, 3, 5, 7, 10, 20]`이고 매 N마다 하나의 joint TaF-Adapter를 학습합니다. Force/pressure VQ-VAE와 tactile encoder는 기존 loss로 함께 학습하며, force target과 force reconstruction은 N프레임 전체입니다. ViT freeze 설정도 유지합니다. Force prediction은 학습이 끝난 체크포인트의 tactile embedding만 사용하는 별도 ridge probe입니다.

## Label 준비와 공통 manifest

`dataset_visualization`의 **기존 로컬 사본**에서 centered MTCC 결과를 export합니다. 이 저장소가 분류기를 학습 중에 실행하거나 다른 저장소를 clone하지 않습니다. 다음 명령의 경로는 자신의 실제 경로로 바꿉니다.

```bash
uv run --project /path/to/dataset_visualization taf-explorer classify \
  --dataset-root /path/to/TaF-Dataset --cache-root /path/to/label-cache \
  --config /path/to/dataset_visualization/configs/classifier.yaml --mode centered-mtcc

uv run python -m script.dynamics_experiment manifest \
  --data-root /path/to/TaF-Dataset \
  --labels /path/to/label-cache/centered-mtcc-phase-v2/centered_labels.parquet \
  --label-metadata /path/to/label-cache/centered-mtcc-phase-v2/centered_metadata.json \
  --output outputs/comparison/manifest.parquet \
  --split-seed 42 --fractions 0.8 0.1 0.1 \
  --timestamp-tolerance 0.001 --gallery-seed 42 --gallery-max-pairs 256
```

분류 명령이 출력한 실제 `centered_labels.parquet`, `centered_metadata.json` 경로를 사용합니다. Importer는 export의 `frame_index`를 **episode 내부 0부터 시작하는 위치**로 해석하고 `episode_index`와 분리합니다. 원본 loader가 검증한 timestamp와 label의 `timestamp_anchor`를 허용 오차 내에서 대조합니다. 중복·잘못 연결된 episode·범위 밖 위치는 오류로 중단하고, 누락·timestamp 불일치·No Contact·history 부족은 manifest 옆 JSON에 사유와 수를 저장합니다. 원본 F/T와 pressure를 분류용 전처리 값으로 대체하지 않습니다.

`sample_id`는 sensor/sequence/episode/anchor로 구성되어 N과 무관합니다. 공통 manifest에는 N=20의 과거 구간을 구성할 수 있는 anchor만 들어갑니다. 각 입력은 anchor `t`에 대해 `[t-N+1, ..., t]`이며 episode 경계·미래 프레임을 넘지 않습니다. Split은 sequence 단위로 한 번 저장하며 모든 N·training seed가 공유합니다. 정규화는 train에 포함된 공통 20프레임 구간의 **중복 없는 원본 프레임**으로 계산해 manifest 옆 `normalization_<manifest-id>.pt`에 공유합니다.

최종 5분류 우선순위는 sudden_event → Loading → Unloading → Holding → Transition / Unknown → No Contact 제외입니다. 원본 phase와 sudden 값은 manifest에 보존합니다. `transition`은 현재 classifier의 **Transition / Unknown**입니다. 현재 sudden 진단은 고정된 **trailing 20프레임** 안의 급변 여부이며 모델 입력 N과 독립적입니다. Centered label은 미래 관측을 이용하는 **offline heuristic annotation**입니다. 평가 그룹과 metadata에만 사용하고 모델 입력에 넣지 않습니다. `centered_metadata.json`을 제공하면 분류 해시·버전·trailing 20 설정을 검증합니다.

## 학습

```bash
# N·training seed 하나
uv run python -m script.train_adapter comparison.enabled=true \
  comparison.manifest=outputs/comparison/manifest.parquet \
  data.root_dir=/path/to/TaF-Dataset data.window_size=5 data.split_seed=42 \
  training.seed=42 training.min_same_episode_window_gap=20 \
  training.batch_size=16 training.gradient_accumulation_steps=1 \
  training.output_dir=outputs/comparison/runs wandb.enabled=false

# 여섯 N × seed 하나
uv run python -m script.train_adapter -m comparison.enabled=true \
  comparison.manifest=outputs/comparison/manifest.parquet \
  data.root_dir=/path/to/TaF-Dataset data.window_size=1,3,5,7,10,20 \
  training.seed=42 training.min_same_episode_window_gap=20 \
  training.batch_size=16 training.gradient_accumulation_steps=1 \
  training.output_dir=outputs/comparison/runs wandb.enabled=false

# 여섯 N × 세 seed (18개 학습을 실제로 시작하므로 자원을 확인한 뒤 실행)
uv run python -m script.train_adapter -m comparison.enabled=true \
  comparison.manifest=outputs/comparison/manifest.parquet \
  data.root_dir=/path/to/TaF-Dataset data.window_size=1,3,5,7,10,20 \
  training.seed=42,43,44 training.min_same_episode_window_gap=20 \
  training.batch_size=16 training.gradient_accumulation_steps=1 \
  training.output_dir=outputs/comparison/runs wandb.enabled=false
```

비교 모드는 실제 forward batch size를 모든 N에서 같게 유지합니다. Gradient accumulation은 InfoNCE negative 수를 늘리지 않습니다. 고정 sampling seed 42와 같은 anchor 목록으로 완전한 batch만 사용하며, 같은 episode의 anchor 간격을 기본 20프레임 이상으로 제한합니다. 구성하지 못한 표본 수와 실제 batch 크기는 `metrics.jsonl`에 기록합니다. Force 모듈의 N별 파라미터 수는 구조상 달라집니다. Tactile 모듈 초기화 난수는 force 모듈과 분리해 같은 seed에서 같은 shape의 파라미터가 동일하게 시작합니다. 출력 폴더·W&B 이름에는 manifest ID, N, seed가 들어갑니다. 각 run은 독립 학습하고 `last.pt`가 final-step 비교 체크포인트입니다. Resume은 같은 비교 실험 식별자만 허용합니다.

## 체크포인트별 평가와 결과

```bash
uv run python -m script.dynamics_experiment retrieval \
  --checkpoint outputs/comparison/runs/XXXXXXXX_N5_seed42/last.pt \
  --manifest outputs/comparison/manifest.parquet --split test \
  --output outputs/comparison/results/retrieval_N5_seed42.parquet

uv run python -m script.dynamics_experiment probe \
  --checkpoint outputs/comparison/runs/XXXXXXXX_N5_seed42/last.pt \
  --manifest outputs/comparison/manifest.parquet \
  --output outputs/comparison/results/probe_N5_seed42.parquet

uv run python -m script.dynamics_experiment aggregate \
  --results outputs/comparison/results/*.parquet \
  --output outputs/comparison/summary
```

위 체크포인트 경로의 XXXXXXXX은 manifest_id 앞 8자로 바꿉니다.

`retrieval --split validation`도 지원합니다. Manifest 옆 `gallery_plan.json`의 sample ID와 순서는 모든 N·seed에서 같습니다. Force embedding은 **각 N의 원본 force/pressure window와 해당 모델의 force encoder**로 다시 계산합니다. 전체 gallery에서 exact-pair rank를 구한 뒤 query만 dynamics별로 집계합니다. 빈 그룹은 0점이 아니라 unavailable입니다. Query별 rank·Recall, gallery 크기, VQ code pair 중복률, tie 수가 저장됩니다. VQ 중복을 이유로 positive나 loss 정의를 바꾸지 않습니다.

Probe는 tactile encoder를 freeze하고 train embedding으로만 bias 포함 ridge를 fitting합니다. Tactile embedding은 train 평균·표준편차로 표준화하고 target은 공통 train frame의 F/T 평균·표준편차로 정규화합니다. 고정 alpha 후보 `[0.01, 0.1, 1, 10, 100]` 중 validation MSE가 가장 작은 값을 선택합니다. Test는 선택 후 한 번 평가합니다. 출력 `(B,N,6)`은 원래 기록 단위로 역변환됩니다. Whole-window MAE/RMSE는 N마다 다른 과거 구간, endpoint MAE/RMSE는 공통 anchor 시점을 뜻합니다. Whole-window embedding으로 과거 force를 예측하므로 과거 각 시점의 온라인 causal 추정이 아닙니다.

CSV와 Parquet에는 `sensor_configuration, sequence_id, episode_index, sample_id, model_run_id, model_window_size, contact_state`가 들어가 `dataset_visualization`의 `model_analysis` 페이지에서 읽을 수 있습니다. `contact_state`는 dynamics5입니다. `force_prediction_mae`는 **6축 전체 원소 평균**으로 축별 단위가 섞인 수치이므로 물리적 크기 비교에는 `force_mae`(앞 3축), `torque_mae`(뒤 3축), `axis_0..5_mae`를 따로 사용합니다. 실제 F/T 단위는 데이터 `meta/info.json`의 명시된 항목으로 확인하며 없으면 `force_torque_unit_metadata`는 비어 있습니다. 집계표에는 5×6 support, N=1 대비 차이, seed 평균·표준편차와 학습·추론 시간이 들어갑니다. 그래프의 오차 막대는 seed 간 표준편차이며 겹치는 프레임을 독립 표본으로 둔 신뢰구간이 아닙니다. 참고 `model_analysis` 페이지의 자체 CI는 업로드 행을 독립으로 가정하므로 이 실험의 신뢰구간으로 인용하지 않습니다. 시간 측정은 wall clock이며 평가 시간에는 영상 decode와 데이터 로딩이 포함됩니다.

작은 기능 검증: `uv run python -m unittest tests.test_dynamics_comparison -q`. 실제 label artifact와 데이터가 없는 상태에서 합성 label을 실제 연구 결과로 사용하지 마세요.
