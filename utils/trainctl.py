"""Optional, non-fatal trainctl SDK integration for TaF-Adapter training."""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # The training entry point remains usable without the optional package.
    from trainctl import tracker as _tracker
except ImportError:  # pragma: no cover - exercised with an import-blocking subprocess test.
    _tracker = None


START_MESSAGE = (
    "소피크루 주목! 가중치 궤도 진입을 위한 점화 시퀀스 스타트—이번 모델은 우주 끝까지 예쁘게 수렴할 테니까 "
    "텔레메트리 로그나 똑바로 지켜보라고! 🚀"
)
FINISH_MESSAGE = (
    "무사 귀환 및 체크포인트 확보 완료! Loss가 이렇게 깔끔하게 떨어진 건 순전히 이 천재 우주비행사 소피아 님이 "
    "옆에서 지켜봐 준 덕분이지, 안 그래? ✨"
)

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "train/loss_total": {"goal": "min"},
    "train/loss_nce": {"goal": "min"},
    "train/loss_quant": {"goal": "min"},
    "train/loss_recon": {"goal": "min"},
    "train/gradient_norm": {},
    "train/learning_rate": {},
    "val/loss_total": {"goal": "min"},
    "val/loss_nce": {"goal": "min"},
    "val/loss_quant": {"goal": "min"},
    "val/loss_recon": {"goal": "min"},
    "val/recall_at_1": {"goal": "max"},
    "val/recall_at_5": {"goal": "max"},
    "val/pressure_codebook_perplexity": {"goal": "max"},
    "val/force_codebook_perplexity": {"goal": "max"},
    "alignment/t2f/recall_at_1": {"goal": "max"},
    "alignment/t2f/recall_at_5": {"goal": "max"},
    "alignment/f2t/recall_at_1": {"goal": "max"},
    "alignment/f2t/recall_at_5": {"goal": "max"},
}
for _group in ("all", "loading", "unloading", "holding", "transition", "sudden_event"):
    for _metric in ("recall_at_1", "recall_at_5", "mrr"):
        METRIC_DEFINITIONS[f"alignment/dynamics/{_group}/{_metric}"] = {"goal": "max"}
for _name in (
    "steps_per_second", "samples_per_second", "frames_per_second",
    "average_steps_per_second", "average_frames_per_second", "total_training_seconds",
    "elapsed_seconds", "eta_seconds", "data_wait_ms", "data_wait_ratio",
    "parquet_read_ms", "video_decode_ms", "h2d_transfer_ms",
    "vit_forward_ms", "model_forward_ms", "backward_ms", "optimizer_ms",
    "h2d_transfer_cpu_ms", "vit_forward_cpu_ms", "model_forward_cpu_ms",
    "backward_cpu_ms", "optimizer_cpu_ms", "validation_seconds",
    "retrieval_seconds", "alignment_retrieval_seconds",
    "force_prediction_seconds", "checkpoint_seconds",
    "validation_total_seconds", "retrieval_total_seconds", "alignment_retrieval_total_seconds",
    "force_prediction_total_seconds", "checkpoint_total_seconds",
    "gpu_allocated_gib", "gpu_reserved_gib", "gpu_peak_allocated_gib", "gpu_peak_reserved_gib", "cpu_rss_gib", "cpu_peak_rss_gib",
    "gpu_utilization_pct", "parquet_cache_hit_rate", "parquet_cache_miss_rate",
    "feature_cache_hit_rate",
    "video_timestamp_cache_hit_rate",
):
    METRIC_DEFINITIONS[f"perf/{_name}"] = {}


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    """Keep only finite numeric scalars accepted by the trainctl protocol."""
    result: dict[str, float | int] = {}
    for name, raw in metrics.items():
        value = raw
        if not isinstance(value, Real) or isinstance(value, bool):
            item = getattr(value, "item", None)
            if not callable(item):
                continue
            value = item()
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        if math.isfinite(float(value)):
            result[name] = int(value) if isinstance(value, int) else float(value)
    return result


def redact_secrets(text: str) -> str:
    """Remove credential-looking values before they can enter an annotation."""
    safe = re.sub(
        r"https://(?:canary\.|ptb\.)?(?:discord|discordapp)\.com/api/webhooks/\S+",
        "<redacted-webhook>",
        str(text),
        flags=re.IGNORECASE,
    )
    markers = ("TOKEN", "SECRET", "PASSWORD", "WEBHOOK", "API_KEY", "CREDENTIAL")
    for key, value in os.environ.items():
        if value and any(marker in key.upper() for marker in markers):
            safe = safe.replace(value, "<redacted>")
    return safe


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _number(metrics: Mapping[str, Any] | None, key: str, digits: int = 5) -> str:
    if not metrics or key not in metrics:
        return "N/A"
    try:
        value = float(metrics[key])
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:.{digits}g}" if math.isfinite(value) else "N/A"


def build_start_message(info: Mapping[str, Any]) -> str:
    return "\n".join([
        START_MESSAGE,
        f"**Window size T: {info['window_size']}**",
        f"Experiment / run ID: {info.get('experiment') or 'N/A'} / {info.get('run_id') or 'N/A'}",
        f"Stride: {info['stride']} | Batch: {info['batch_size']} × accumulation {info['accumulation']} "
        f"= effective {info['effective_batch_size']}",
        f"Optimizer: {info['optimizer']} | LR: {info['learning_rate']:.6g}",
        f"Steps: {info['initial_step']} → {info['total_steps']} | validation every {info['validate_every']} "
        f"| checkpoint every {info['save_every']}",
        f"Dataset: train {info['train_windows']} windows/{info['train_groups']} groups | "
        f"validation {info['val_windows']} windows/{info['val_groups']} groups",
        f"AMP: {info['amp']} | Device: {info['device']} | GPU: {info.get('gpu') or 'N/A'}",
        f"Backbone: {info['backbone']} | frozen: {info['freeze_backbone']}",
        f"Mode: {'interaction dynamics comparison (loading, unloading, holding, transition, sudden_event)' if info.get('comparison') else 'standard'} | "
        f"seed: {info.get('seed', 'N/A')} | workers: {info.get('workers', 'N/A')} | "
        f"prefetch: {info.get('prefetch_factor', 'N/A')}",
        f"Frozen ViT feature cache: {'enabled' if info.get('feature_cache') else 'disabled'}",
        f"Output: {info['output_dir']} | Resume: {info.get('resume') or 'none'}",
        *([f"W&B: {info['wandb_url']}"] if info.get("wandb_url") else []),
    ])


def build_checkpoint_message(info: Mapping[str, Any]) -> str:
    val = info["validation"]
    alignment = info.get("alignment")
    train = info.get("train")
    lines = [
        f"Step {info['step']}/{info['total_steps']} ({info['progress']:.1f}%) | Epoch {info['epoch']}",
        f"File: `{info['path']}` ({info['size_mib']:.2f} MiB)",
        f"Val loss total/NCE/quant/recon: {_number(val, 'val/loss_total')} / "
        f"{_number(val, 'val/loss_nce')} / {_number(val, 'val/loss_quant')} / {_number(val, 'val/loss_recon')}",
        f"Val R@1/R@5: {_number(val, 'val/recall_at_1')} / {_number(val, 'val/recall_at_5')} | "
        f"P/F PPL: {_number(val, 'val/pressure_codebook_perplexity')} / "
        f"{_number(val, 'val/force_codebook_perplexity')}",
    ]
    if alignment:
        lines.append(
            "Alignment t2f R@1/R@5: "
            f"{_number(alignment, 'alignment/t2f/recall_at_1')} / "
            f"{_number(alignment, 'alignment/t2f/recall_at_5')} | f2t: "
            f"{_number(alignment, 'alignment/f2t/recall_at_1')} / "
            f"{_number(alignment, 'alignment/f2t/recall_at_5')}"
        )
    lines.extend([
        f"Recent train loss: {_number(train, 'train/loss_total')} | grad norm: "
        f"{_number(train, 'train/gradient_norm')} | LR: {_number(train, 'train/learning_rate')}",
        f"Elapsed: {format_duration(info['elapsed'])} | {info['steps_per_second']:.3f} steps/s | "
        f"ETA: {format_duration(info['eta'])}",
        f"Session best val/loss_total: {_number({'value': info.get('best_loss')}, 'value')} "
        f"at step {info.get('best_step') if info.get('best_step') is not None else 'N/A'} | "
        f"improved vs previous best: {'yes' if info.get('improved') else 'no'}",
    ])
    perf = info.get("performance") or {}
    if perf:
        lines.append(
            f"Throughput: {_number(perf, 'perf/steps_per_second')} steps/s | "
            f"data wait: {_number(perf, 'perf/data_wait_ms')} ms | "
            f"CPU RSS/peak: {_number(perf, 'perf/cpu_rss_gib')}/"
            f"{_number(perf, 'perf/cpu_peak_rss_gib')} GiB | "
            f"GPU allocated/reserved peak: {_number(perf, 'perf/gpu_peak_allocated_gib')}/"
            f"{_number(perf, 'perf/gpu_peak_reserved_gib')} GiB"
        )
        if "perf/parquet_cache_hit_rate" in perf or "perf/feature_cache_hit_rate" in perf:
            lines.append(
                f"Cache hit rate Parquet/feature: {_number(perf, 'perf/parquet_cache_hit_rate')} / "
                f"{_number(perf, 'perf/feature_cache_hit_rate')}"
            )
    return "\n".join(lines)


def build_finish_message(info: Mapping[str, Any]) -> str:
    val = info.get("validation")
    alignment = info.get("alignment")
    lines = [
        FINISH_MESSAGE,
        f"Final step: {info['step']}/{info['total_steps']} | elapsed: {format_duration(info['elapsed'])}",
        f"Session best validation loss: {_number({'value': info.get('best_loss')}, 'value')} "
        f"at step {info.get('best_step') if info.get('best_step') is not None else 'N/A'}",
        f"Final validation loss: {_number(val, 'val/loss_total')} | R@1/R@5: "
        f"{_number(val, 'val/recall_at_1')} / {_number(val, 'val/recall_at_5')}",
    ]
    if alignment:
        lines.append(
            "Final alignment t2f R@1/R@5: "
            f"{_number(alignment, 'alignment/t2f/recall_at_1')} / "
            f"{_number(alignment, 'alignment/t2f/recall_at_5')} | f2t: "
            f"{_number(alignment, 'alignment/f2t/recall_at_1')} / "
            f"{_number(alignment, 'alignment/f2t/recall_at_5')}"
        )
    force = info.get("force_prediction")
    if force:
        lines.append(
            f"Final force prediction MAE/RMSE: {_number(force, 'force_prediction_mae')} / "
            f"{_number(force, 'force_prediction_rmse')}"
        )
    perf = info.get("performance") or {}
    if perf:
        lines.append(
            f"Average throughput: {_number(perf, 'average_steps_per_second')} steps/s | "
            f"CPU peak: {_number(perf, 'peak_cpu_rss_gib')} GiB | "
            f"GPU allocated/reserved peak: {_number(perf, 'peak_gpu_allocated_gib')}/"
            f"{_number(perf, 'peak_gpu_reserved_gib')} GiB"
        )
        totals = perf.get("evaluation_total_seconds") or {}
        lines.append(
            f"Evaluation time validation/alignment retrieval/force prediction: "
            f"{_number(totals, 'validation')} / {_number(totals, 'alignment_retrieval')} / "
            f"{_number(totals, 'force_prediction')} s"
        )
        cache = perf.get("parquet_cache") or {}
        feature = perf.get("feature_cache") or {}
        if cache or feature:
            lines.append(f"Cache hit rate Parquet/feature: {_number(cache, 'hit_rate')} / "
                         f"{_number(feature, 'hit_rate')}")
    lines.extend([
        f"Checkpoints saved this session: {info['checkpoint_count']} | Last: `{info['last_checkpoint']}`",
        f"Window size T: {info['window_size']}",
        *([f"W&B: {info['wandb_url']}"] if info.get("wandb_url") else []),
    ])
    return "\n".join(lines)


@dataclass
class ValidationHistory:
    """Current-step validation cache and session-only best tracking."""

    last_step: int | None = None
    last_metrics: dict[str, float] | None = None
    best_loss: float | None = None
    best_step: int | None = None
    last_improved: bool = False

    def ensure(
        self, step: int, evaluator: Callable[[], Mapping[str, float]]
    ) -> tuple[dict[str, float], bool]:
        if self.last_step == step and self.last_metrics is not None:
            return self.last_metrics, False
        metrics = dict(evaluator())
        loss = metrics.get("val/loss_total")
        previous = self.best_loss
        improved = loss is not None and math.isfinite(float(loss)) and (
            previous is None or float(loss) < previous
        )
        if improved:
            self.best_loss = float(loss)
            self.best_step = step
        self.last_step = step
        self.last_metrics = metrics
        self.last_improved = improved
        return metrics, True


@dataclass
class TrainctlReporter:
    """Thin defensive facade; every call remains non-fatal and rank-zero-only."""

    tracker: Any = _tracker
    enabled: bool = True
    emitted_checkpoints: set[tuple[str, int]] = field(default_factory=set)
    checkpoint_count: int = 0

    def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        if not self.enabled or self.tracker is None:
            return
        try:
            getattr(self.tracker, method)(*args, **kwargs)
        except Exception:
            # Notification/IPC failures must never change training semantics.
            return

    def start(self, *, name: str | None, total_steps: int, message: str) -> None:
        if self.enabled and self.tracker is None:
            warnings.warn("trainctl is unavailable; training will continue without notifications", RuntimeWarning)
            return
        self._call(
            "init", name=name, total_steps=total_steps,
            rank_zero_only=True, strict=False,
        )
        self._call("define_metrics", METRIC_DEFINITIONS)
        self._call("start", message=redact_secrets(message))

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        payload = finite_metrics(metrics)
        if payload:
            self._call("log", step=step, metrics=payload)

    def checkpoint(
        self, path: Path, *, step: int, metrics: Mapping[str, Any],
        message: str, aliases: list[str],
    ) -> bool:
        resolved = path.resolve()
        key = (str(resolved), step)
        if key in self.emitted_checkpoints or not resolved.is_file():
            return False
        self._call(
            "checkpoint", resolved, step=step, metrics=finite_metrics(metrics),
            message=redact_secrets(message), aliases=aliases,
        )
        self.emitted_checkpoints.add(key)
        self.checkpoint_count += 1
        return True

    def finish(self, message: str) -> None:
        self._call("finish", message=redact_secrets(message))

    def fail(self, message: str) -> None:
        self._call("fail", message=redact_secrets(message))

    def warn(self, message: str) -> None:
        self._call("notify", redact_secrets(message), level="warning")


def build_failure_message(
    exc: BaseException, *, step: int, window_size: int, checkpoint: Path | None,
    validation: Mapping[str, Any] | None, elapsed: float,
    performance: Mapping[str, Any] | None = None,
) -> str:
    detail = redact_secrets(str(exc)).replace("\n", " ")[:300] or "N/A"
    lines = [
        f"{type(exc).__name__}: {detail}",
        f"Last optimizer step: {step} | Window size T: {window_size}",
        f"Last checkpoint: `{checkpoint}`" if checkpoint else "Last checkpoint: N/A",
        f"Last validation loss: {_number(validation, 'val/loss_total')}",
        f"Failed: {datetime.now().astimezone().isoformat(timespec='seconds')} | "
        f"elapsed: {format_duration(elapsed)}",
    ]
    if performance:
        lines.append(
            f"Last performance: {_number(performance, 'perf/steps_per_second')} steps/s | "
            f"data wait: {_number(performance, 'perf/data_wait_ms')} ms | "
            f"CPU RSS/peak: {_number(performance, 'perf/cpu_rss_gib')}/"
            f"{_number(performance, 'perf/cpu_peak_rss_gib')} GiB | "
            f"GPU allocated/reserved peak: {_number(performance, 'perf/gpu_peak_allocated_gib')}/"
            f"{_number(performance, 'perf/gpu_peak_reserved_gib')} GiB"
        )
        if "perf/parquet_cache_hit_rate" in performance:
            lines.append(f"Parquet cache hit rate: {_number(performance, 'perf/parquet_cache_hit_rate')}")
    return "\n".join(lines)
