"""Backward-compatible checkpoint and RNG helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from utils.common import resolved_config


def tactile_signature(config) -> dict[str, Any]:
    """Semantic identity; paths may differ but pretrained bytes/protocol may not."""
    tactile = config["tactile_encoder"]
    kind = tactile.get("backbone_type", "imagenet")
    result = {"backbone_type": kind, "backbone": tactile["backbone"]}
    if kind == "sparsh":
        options = tactile["sparsh"]
        result.update({key: options[key] for key in ["repo_id", "revision", "filename", "sha256",
                      "temporal_stride", "pairing_mode", "background_subtraction", "pooling", "pooling_heads"]})
        result["preprocessing_version"] = 1
    return result


def assert_tactile_checkpoint_compatible(checkpoint, config) -> None:
    if "config" not in checkpoint:
        raise ValueError("Resume checkpoint has no backbone configuration identity")
    current = tactile_signature(config)
    saved = tactile_signature(checkpoint["config"])
    if current != saved:
        raise ValueError("Resume tactile backbone/input protocol mismatch; ImageNet and Sparsh checkpoints are incompatible")
    metadata = checkpoint.get("tactile_backbone")
    if metadata is not None and metadata["backbone_type"] != current["backbone_type"]:
        raise ValueError("Checkpoint backbone metadata disagrees with configuration")
    if current["backbone_type"] == "sparsh":
        from model.sparsh import UPSTREAM_REVISION
        if metadata is None:
            raise ValueError("Sparsh resume checkpoint is missing pretrained backbone identity")
        for key in ["repo_id", "revision", "filename", "sha256"]:
            if metadata["pretrained"].get(key) != current[key]:
                raise ValueError(f"Sparsh checkpoint pretrained identity mismatch: {key}")
        for key in ["temporal_stride", "pairing_mode", "background_subtraction", "pooling_heads"]:
            if metadata["input_protocol"].get(key) != current[key]:
                raise ValueError(f"Sparsh checkpoint input protocol mismatch: {key}")
        protocol = metadata["input_protocol"]
        if (metadata["pretrained"].get("upstream_revision") != UPSTREAM_REVISION
                or protocol.get("version") != 1
                or protocol.get("pooling") != "attentive_patch_only"
                or tuple(protocol.get("resolution", ())) != (320, 240)
                or protocol.get("channel_order") != "current_RGB,reference_RGB"
                or protocol.get("geometry") != "official_orientation_crop_tensor_bilinear_antialias"):
            raise ValueError("Sparsh checkpoint pooling/preprocessing implementation mismatch")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load current or pre-refactor checkpoints without requiring new fields."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a TaF-Adapter checkpoint: {path}")
    return checkpoint


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng: dict[str, Any] | None) -> None:
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def restore_rng(checkpoint: dict[str, Any]) -> None:
    restore_rng_state(checkpoint.get("rng"))


def save_checkpoint(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, step: int, epoch: int, config: DictConfig | dict[str, Any],
    normalization: dict[str, torch.Tensor], train_keys: list[str], val_keys: list[str],
    wandb_state: dict[str, Any] | None = None,
    alignment_state: dict[str, Any] | None = None,
    comparison_identity: dict[str, Any] | None = None,
    training_duration_seconds: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": resolved_config(config),
        "normalization": {key: value.cpu() for key, value in normalization.items()},
        "split": {"train_groups": train_keys, "validation_groups": val_keys},
        "rng": capture_rng(),
    }
    if hasattr(model, "tactile_encoder") and hasattr(model.tactile_encoder, "checkpoint_identity"):
        payload["tactile_backbone"] = model.tactile_encoder.checkpoint_identity()
    if wandb_state:
        payload["wandb"] = dict(wandb_state)
    if alignment_state is not None:
        payload["alignment"] = dict(alignment_state)
    if comparison_identity is not None:
        payload["comparison_identity"] = dict(comparison_identity)
        payload["model_parameters"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "force_encoder": sum(p.numel() for p in model.force_encoder.parameters()),
            "tactile_encoder": sum(p.numel() for p in model.tactile_encoder.parameters()),
        }
    if training_duration_seconds is not None:
        payload["training_duration_seconds"] = float(training_duration_seconds)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

