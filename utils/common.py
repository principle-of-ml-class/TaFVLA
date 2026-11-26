"""Configuration, reproducibility, and path helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


_CONFIG_KEYS = {
    "data": {
        "root_dir", "window_size", "stride", "image_size", "max_sequences",
        "sensor_types", "split_seed", "split_unit", "validation_fraction",
        "skip_invalid_sequences", "timestamp_cache_enabled", "timestamp_cache_dir",
        "sensor_cache_max_mb", "sensor_arrow_conversion", "video_cache_max_entries",
        "batch_group_reads", "frame_cache_max_open_files",
        "sensor_mmap_enabled", "sensor_mmap_dir", "sensor_mmap_rebuild",
        "sensor_mmap_max_open_files", "sensor_mmap_batch_rows",
        "frame_cache_enabled", "frame_cache_dir", "frame_cache_rebuild", "frame_cache_batch_size",
        "show_initialization_progress", "alignment_decode_windows_per_episode",
        "normalization_eps", "normalization_chunk_frames",
    },
    "force_encoder": {
        "hidden_dim", "pressure_latent_dim", "force_latent_dim", "mlp_layers",
    },
    "vq": {"pressure_codebook_size", "force_codebook_size", "commitment_beta"},
    "tactile_encoder": {
        "backbone", "pretrained", "freeze_backbone", "transformer_dim",
        "transformer_layers", "num_heads", "dropout", "max_window_size", "backbone_chunk_frames",
        "backbone_type", "sparsh",
    },
    "loss": {"lambda_quant", "lambda_recon", "temperature", "nce_mode"},
    "training": {
        "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "max_steps", "num_workers", "eval_num_workers", "prefetch_factor", "seed", "amp", "grad_clip_norm", "log_every",
        "validate_every", "validation_batches", "save_every", "output_dir", "resume",
        "min_same_episode_window_gap", "batch_sampler",
    },
    "wandb": {
        "enabled", "project", "entity", "name", "group", "tags", "mode",
        "log_every", "watch_model",
    },
    "alignment_eval": {
        "enabled", "retrieval_every_n_steps", "retrieval_every_n_epochs",
        "visualize_every_n_epochs",
        "retrieval_max_pairs", "visualization_max_pairs", "min_window_start_gap",
        "random_seed", "run_on_training_end", "save_local_figures",
    },
    "check": {"overfit_steps", "skip_overfit", "no_pretrained", "window_sizes"},
    "comparison": {"enabled", "manifest", "sampling_seed", "gallery_seed", "gallery_max_pairs",
                   "min_anchor_gap", "window_sizes", "training_seeds"},
    "trainctl": {"enabled", "performance"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    """Resolve a user path relative to Hydra's original launch directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(HydraConfig.get().runtime.cwd) if HydraConfig.initialized() else Path.cwd()
    return (base / path).resolve()


def resolved_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    else:
        value = dict(config)
    if not isinstance(value, dict):
        raise TypeError("Config must resolve to a mapping")
    return value


def _mapping(config: DictConfig, section: str) -> DictConfig:
    value = config.get(section)
    if not isinstance(value, DictConfig):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None, inclusive_maximum: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None:
        invalid = number > maximum if inclusive_maximum else number >= maximum
        if invalid:
            operator = "<=" if inclusive_maximum else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {value}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def validate_config(config: DictConfig, *, require_data_root: bool = False) -> None:
    """Validate known keys plus the high-impact types and ranges."""
    unknown_sections = set(config.keys()) - set(_CONFIG_KEYS)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")
    missing_sections = set(_CONFIG_KEYS) - set(config.keys())
    if missing_sections:
        raise ValueError(f"Missing config section(s): {sorted(missing_sections)}")
    sections = {name: _mapping(config, name) for name in _CONFIG_KEYS}
    for name, allowed in _CONFIG_KEYS.items():
        unknown = set(sections[name].keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown config key(s) in {name}: {sorted(unknown)}")

    control = sections["trainctl"]
    _boolean(control.enabled, "trainctl.enabled")
    perf = control.performance
    perf_keys = {"enabled", "detailed_profiling", "log_every", "monitor_cpu_memory",
                 "monitor_gpu_memory", "monitor_throughput", "monitor_data_loading",
                 "monitor_cache", "monitor_gpu_utilization", "warnings"}
    if set(perf) != perf_keys:
        raise ValueError("trainctl.performance keys do not match the schema")
    for key in perf_keys - {"log_every", "warnings"}:
        _boolean(perf[key], f"trainctl.performance.{key}")
    _integer(perf.log_every, "trainctl.performance.log_every", minimum=1)
    warning_cfg = perf.warnings
    warning_keys = {"enabled", "data_wait_ratio_threshold", "stall_timeout_seconds",
                    "warning_cooldown_seconds", "cpu_rss_warning_gib",
                    "gpu_reserved_warning_gib", "cache_hit_rate_threshold", "send_to_trainctl"}
    if set(warning_cfg) != warning_keys:
        raise ValueError("trainctl.performance.warnings keys do not match the schema")
    _boolean(warning_cfg.enabled, "trainctl.performance.warnings.enabled")
    _boolean(warning_cfg.send_to_trainctl, "trainctl.performance.warnings.send_to_trainctl")
    for key in ("data_wait_ratio_threshold", "cache_hit_rate_threshold"):
        _number(warning_cfg[key], f"trainctl.performance.warnings.{key}", minimum=0,
                maximum=1)
    for key in ("stall_timeout_seconds", "warning_cooldown_seconds"):
        if _number(warning_cfg[key], f"trainctl.performance.warnings.{key}") <= 0:
            raise ValueError(f"trainctl.performance.warnings.{key} must be positive")
    for key in ("cpu_rss_warning_gib", "gpu_reserved_warning_gib"):
        if warning_cfg[key] is not None and _number(
            warning_cfg[key], f"trainctl.performance.warnings.{key}"
        ) <= 0:
            raise ValueError(f"trainctl.performance.warnings.{key} must be positive")

    data = sections["data"]
    if require_data_root and not data.root_dir:
        raise ValueError("data.root_dir is required for training")
    if data.root_dir is not None and not isinstance(data.root_dir, str):
        raise TypeError("data.root_dir must be null or a path string")
    _integer(data.window_size, "data.window_size", minimum=1)
    _integer(data.stride, "data.stride", minimum=1)
    if data.image_size is not None:
        if not isinstance(data.image_size, (list, tuple, ListConfig)) or len(data.image_size) != 2:
            raise TypeError("data.image_size must contain [height, width] or null for native images")
        for index, size in enumerate(data.image_size):
            _integer(size, f"data.image_size[{index}]", minimum=1)
    if data.max_sequences is not None:
        _integer(data.max_sequences, "data.max_sequences", minimum=1)
    if data.sensor_types is not None:
        if isinstance(data.sensor_types, str):
            pass
        elif not isinstance(data.sensor_types, (list, tuple, ListConfig)) or not all(
            isinstance(value, str) for value in data.sensor_types
        ):
            raise TypeError("data.sensor_types must be null, a string, or a list of strings")
    _integer(data.split_seed, "data.split_seed", minimum=0)
    if data.split_unit not in {"sequence", "episode"}:
        raise ValueError("data.split_unit must be 'sequence' or 'episode'")
    validation_fraction = _number(data.validation_fraction, "data.validation_fraction")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    _boolean(data.skip_invalid_sequences, "data.skip_invalid_sequences")
    _boolean(data.timestamp_cache_enabled, "data.timestamp_cache_enabled")
    _integer(data.sensor_cache_max_mb, "data.sensor_cache_max_mb", minimum=0)
    _boolean(data.sensor_mmap_enabled, "data.sensor_mmap_enabled")
    _boolean(data.sensor_mmap_rebuild, "data.sensor_mmap_rebuild")
    _integer(data.sensor_mmap_max_open_files, "data.sensor_mmap_max_open_files", minimum=1)
    _integer(data.sensor_mmap_batch_rows, "data.sensor_mmap_batch_rows", minimum=1)
    if not isinstance(data.sensor_mmap_dir, str) or not data.sensor_mmap_dir:
        raise TypeError("data.sensor_mmap_dir must be a non-empty path string")
    _boolean(data.sensor_arrow_conversion, "data.sensor_arrow_conversion")
    _boolean(data.batch_group_reads, "data.batch_group_reads")
    _integer(data.video_cache_max_entries, "data.video_cache_max_entries", minimum=0)
    _integer(data.frame_cache_max_open_files, "data.frame_cache_max_open_files", minimum=1)
    _boolean(data.frame_cache_enabled, "data.frame_cache_enabled")
    _boolean(data.frame_cache_rebuild, "data.frame_cache_rebuild")
    _integer(data.frame_cache_batch_size, "data.frame_cache_batch_size", minimum=1)
    if data.frame_cache_dir is not None and (
        not isinstance(data.frame_cache_dir, str) or not data.frame_cache_dir
    ):
        raise TypeError("data.frame_cache_dir must be null or a non-empty path string")
    if data.frame_cache_enabled and not sections["tactile_encoder"].freeze_backbone:
        raise ValueError("data.frame_cache_enabled requires a frozen tactile backbone")
    if data.timestamp_cache_dir is not None and (
        not isinstance(data.timestamp_cache_dir, str) or not data.timestamp_cache_dir
    ):
        raise TypeError("data.timestamp_cache_dir must be null or a non-empty path string")
    _boolean(data.show_initialization_progress, "data.show_initialization_progress")
    _integer(
        data.alignment_decode_windows_per_episode,
        "data.alignment_decode_windows_per_episode", minimum=1,
    )
    if _number(data.normalization_eps, "data.normalization_eps") <= 0:
        raise ValueError("data.normalization_eps must be positive")
    _integer(data.normalization_chunk_frames, "data.normalization_chunk_frames", minimum=1)

    force = sections["force_encoder"]
    for key in ("hidden_dim", "pressure_latent_dim", "force_latent_dim", "mlp_layers"):
        _integer(force[key], f"force_encoder.{key}", minimum=1)

    vq = sections["vq"]
    _integer(vq.pressure_codebook_size, "vq.pressure_codebook_size", minimum=1)
    _integer(vq.force_codebook_size, "vq.force_codebook_size", minimum=1)
    _number(vq.commitment_beta, "vq.commitment_beta", minimum=0.0)

    tactile = sections["tactile_encoder"]
    if tactile.get("backbone_type", "imagenet") not in {"imagenet", "sparsh"}:
        raise ValueError("tactile_encoder.backbone_type must be imagenet or sparsh")
    if tactile.get("backbone_type", "imagenet") == "sparsh":
        from model.sparsh import REPO_ID, REVISION, FILENAME, SHA256
        sparsh = tactile.sparsh
        allowed = {"repo_id", "revision", "filename", "sha256", "checkpoint_path", "temporal_stride",
                   "pairing_mode", "background_subtraction", "pooling", "pooling_heads"}
        if set(sparsh) != allowed:
            raise ValueError("Sparsh keys do not match the schema")
        if [sparsh.repo_id, sparsh.revision, sparsh.filename, sparsh.sha256] != [REPO_ID, REVISION, FILENAME, SHA256]:
            raise ValueError("Sparsh requires the pinned official DINO Small checkpoint identity")
        if not tactile.pretrained or tactile.backbone != "sparsh-dino-small":
            raise ValueError("Sparsh requires pretrained=true and backbone=sparsh-dino-small")
        if data.frame_cache_enabled:
            raise ValueError("Sparsh cannot use FrozenFrameCache with window-dependent pairing")
        if data.image_size is not None:
            raise ValueError("Sparsh requires data.image_size=null; native images must reach its preprocessing")
        _integer(sparsh.temporal_stride, "tactile_encoder.sparsh.temporal_stride", minimum=1)
        _integer(sparsh.pooling_heads, "tactile_encoder.sparsh.pooling_heads", minimum=1)
        if 384 % sparsh.pooling_heads:
            raise ValueError("Sparsh pooling_heads must divide 384")
        if sparsh.pairing_mode != "window_clamp" or sparsh.pooling != "attentive":
            raise ValueError("Sparsh requires window_clamp pairing and attentive pooling")
        _boolean(sparsh.background_subtraction, "tactile_encoder.sparsh.background_subtraction")
        if sparsh.checkpoint_path is not None and (not isinstance(sparsh.checkpoint_path, str) or not sparsh.checkpoint_path):
            raise ValueError("Sparsh checkpoint_path must be null or a nonempty path")
    if not isinstance(tactile.backbone, str) or not tactile.backbone:
        raise TypeError("tactile_encoder.backbone must be a non-empty string")
    _boolean(tactile.pretrained, "tactile_encoder.pretrained")
    _boolean(tactile.freeze_backbone, "tactile_encoder.freeze_backbone")
    transformer_dim = _integer(tactile.transformer_dim, "tactile_encoder.transformer_dim", minimum=1)
    _integer(tactile.transformer_layers, "tactile_encoder.transformer_layers", minimum=1)
    num_heads = _integer(tactile.num_heads, "tactile_encoder.num_heads", minimum=1)
    if transformer_dim % num_heads:
        raise ValueError("tactile_encoder.transformer_dim must be divisible by num_heads")
    _number(tactile.dropout, "tactile_encoder.dropout", minimum=0.0,
            maximum=1.0, inclusive_maximum=False)
    _integer(tactile.max_window_size, "tactile_encoder.max_window_size", minimum=1)
    _integer(tactile.backbone_chunk_frames, "tactile_encoder.backbone_chunk_frames", minimum=0)

    loss = sections["loss"]
    _number(loss.lambda_quant, "loss.lambda_quant", minimum=0.0)
    _number(loss.lambda_recon, "loss.lambda_recon", minimum=0.0)
    if _number(loss.temperature, "loss.temperature") <= 0:
        raise ValueError("loss.temperature must be positive")
    if loss.nce_mode not in {"paper", "symmetric"}:
        raise ValueError("loss.nce_mode must be 'paper' or 'symmetric'")

    training = sections["training"]
    if training.get("batch_sampler", "auto") not in {"auto", "epoch_shuffled"}:
        raise ValueError("training.batch_sampler must be 'auto' or 'epoch_shuffled'")
    _integer(training.batch_size, "training.batch_size", minimum=2)
    _integer(training.gradient_accumulation_steps, "training.gradient_accumulation_steps", minimum=1)
    if _number(training.learning_rate, "training.learning_rate") <= 0:
        raise ValueError("training.learning_rate must be positive")
    _number(training.weight_decay, "training.weight_decay", minimum=0.0)
    _integer(training.max_steps, "training.max_steps", minimum=1)
    _integer(training.num_workers, "training.num_workers", minimum=0)
    _integer(training.eval_num_workers, "training.eval_num_workers", minimum=0)
    _integer(training.prefetch_factor, "training.prefetch_factor", minimum=1)
    _integer(training.seed, "training.seed", minimum=0)
    _boolean(training.amp, "training.amp")
    if _number(training.grad_clip_norm, "training.grad_clip_norm") <= 0:
        raise ValueError("training.grad_clip_norm must be positive")
    _integer(training.log_every, "training.log_every", minimum=1)
    _integer(training.validate_every, "training.validate_every", minimum=0)
    if training.validation_batches is not None:
        _integer(training.validation_batches, "training.validation_batches", minimum=1)
    _integer(training.save_every, "training.save_every", minimum=1)
    if not isinstance(training.output_dir, str) or not training.output_dir:
        raise TypeError("training.output_dir must be a non-empty string")
    if training.resume is not None and not isinstance(training.resume, str):
        raise TypeError("training.resume must be null or a path string")
    _integer(
        training.min_same_episode_window_gap,
        "training.min_same_episode_window_gap", minimum=0,
    )

    wandb = sections["wandb"]
    _boolean(wandb.enabled, "wandb.enabled")
    if not isinstance(wandb.project, str) or not wandb.project:
        raise TypeError("wandb.project must be a non-empty string")
    for key in ("entity", "name", "group"):
        if wandb[key] is not None and not isinstance(wandb[key], str):
            raise TypeError(f"wandb.{key} must be null or a string")
    if not isinstance(wandb.tags, (list, tuple, ListConfig)) or not all(
        isinstance(tag, str) for tag in wandb.tags
    ):
        raise TypeError("wandb.tags must be a list of strings")
    if wandb.mode not in {"online", "offline"}:
        raise ValueError("wandb.mode must be 'online' or 'offline'")
    _integer(wandb.log_every, "wandb.log_every", minimum=1)
    _boolean(wandb.watch_model, "wandb.watch_model")

    alignment = sections["alignment_eval"]
    _boolean(alignment.enabled, "alignment_eval.enabled")
    _integer(
        alignment.retrieval_every_n_steps,
        "alignment_eval.retrieval_every_n_steps", minimum=0,
    )
    _integer(
        alignment.retrieval_every_n_epochs,
        "alignment_eval.retrieval_every_n_epochs", minimum=1,
    )
    _integer(
        alignment.visualize_every_n_epochs,
        "alignment_eval.visualize_every_n_epochs", minimum=1,
    )
    retrieval_pairs = _integer(
        alignment.retrieval_max_pairs,
        "alignment_eval.retrieval_max_pairs", minimum=2,
    )
    visualization_pairs = _integer(
        alignment.visualization_max_pairs,
        "alignment_eval.visualization_max_pairs", minimum=2,
    )
    if visualization_pairs > retrieval_pairs:
        raise ValueError(
            "alignment_eval.visualization_max_pairs must not exceed retrieval_max_pairs"
        )
    _integer(
        alignment.min_window_start_gap,
        "alignment_eval.min_window_start_gap", minimum=1,
    )
    _integer(alignment.random_seed, "alignment_eval.random_seed", minimum=0)
    _boolean(alignment.run_on_training_end, "alignment_eval.run_on_training_end")
    _boolean(alignment.save_local_figures, "alignment_eval.save_local_figures")

    check = sections["check"]
    _integer(check.overfit_steps, "check.overfit_steps", minimum=1)
    _boolean(check.skip_overfit, "check.skip_overfit")
    _boolean(check.no_pretrained, "check.no_pretrained")
    if not isinstance(check.window_sizes, (list, tuple, ListConfig)) or not check.window_sizes:
        raise TypeError("check.window_sizes must be a non-empty list")
    for index, size in enumerate(check.window_sizes):
        _integer(size, f"check.window_sizes[{index}]", minimum=1)

    comparison = sections["comparison"]
    _boolean(comparison.enabled, "comparison.enabled")
    if comparison.enabled:
        if not comparison.manifest or not isinstance(comparison.manifest, str):
            raise ValueError("comparison.manifest is required in comparison mode")
        if data.window_size not in comparison.window_sizes or data.window_size > tactile.max_window_size:
            raise ValueError("data.window_size must be in comparison.window_sizes and within tactile max")
        if training.min_same_episode_window_gap < comparison.min_anchor_gap:
            raise ValueError("training.min_same_episode_window_gap must meet comparison.min_anchor_gap")
        if data.stride != 1:
            raise ValueError("comparison mode requires stride=1")
        if list(comparison.window_sizes) != [1, 3, 5, 7, 10, 20]:
            raise ValueError("comparison.window_sizes must be [1, 3, 5, 7, 10, 20]")
        if tactile.max_window_size < 20 or comparison.min_anchor_gap < 20:
            raise ValueError("comparison requires tactile max_window_size and min_anchor_gap >= 20")
    for key in ("sampling_seed", "gallery_seed"):
        _integer(comparison[key], f"comparison.{key}", minimum=0)
    _integer(comparison.gallery_max_pairs, "comparison.gallery_max_pairs", minimum=2)
    _integer(comparison.min_anchor_gap, "comparison.min_anchor_gap", minimum=1)
    for key in ("window_sizes", "training_seeds"):
        if not comparison[key]:
            raise ValueError(f"comparison.{key} must not be empty")
        for value in comparison[key]:
            _integer(value, f"comparison.{key}", minimum=1)

    OmegaConf.set_struct(config, True)

