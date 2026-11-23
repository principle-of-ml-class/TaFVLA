"""Pre-train TaF-Adapter on temporally aligned TaF-Dataset episodes."""

from __future__ import annotations

import os
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset
from utils.dynamics import ManifestDataset
from utils.comparison_eval import write_rows
from utils.alignment import (
    AlignmentSelection,
    alignment_schedule,
    create_alignment_visualizations,
    evaluate_alignment,
    evaluate_alignment_dynamics,
    save_alignment_selection,
    select_alignment_indices,
    should_run_epoch_alignment,
    should_run_final_alignment,
    should_run_step_alignment,
)
from utils.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    restore_rng_state,
    save_checkpoint,
)
from utils.common import resolve_path, resolved_config, set_seed, validate_config
from utils.config import register_config_schema
from utils.data import (
    FixedValidationBatchSampler,
    TemporalDiverseBatchSampler,
    audit_temporal_alignment,
    compute_normalization,
    split_by_group,
)
from utils.logging import DeviceMetricAccumulator, ExperimentLogger, scalar_metrics
from utils.frame_cache import CachedWindowDataset, FrozenFrameCache
from utils.performance import PerformanceMonitor
from utils.model_factory import build_model
from utils.trainctl import (
    TrainctlReporter,
    ValidationHistory,
    build_checkpoint_message,
    build_failure_message,
    build_finish_message,
    build_start_message,
)
from utils.live_progress import emit as progress_event


register_config_schema()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        force = batch["force"].to(device, non_blocking=True)
        pressure = batch["pressure"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            if "backbone_features" in batch:
                features = batch["backbone_features"].to(device, non_blocking=True)
                outputs = model.forward_from_backbone_features(features, force, pressure)
            else:
                tactile = batch["tactile"].to(device, non_blocking=True)
                outputs = model(tactile=tactile, force=force, pressure=pressure)
        for key, value in scalar_metrics(outputs).items():
            sums[key] += value
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val/{key}": value / batches for key, value in sums.items()}


def _run_alignment_evaluation(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path,
    logger: ExperimentLogger,
    global_step: int,
    completed_epoch: int,
    evaluation_label: str,
    visualize: bool,
) -> dict[str, Any]:
    """고정 validation set의 retrieval을 평가하고 한 번만 기록한다."""
    rng_before_evaluation = capture_rng()
    try:
        result = evaluate_alignment(model, loader, device)
        images: dict[str, Any] = {}
        figure_paths: dict[str, str] = {}
        if visualize:
            if evaluation_label == "epoch":
                title_label = f"Epoch {completed_epoch}"
                file_prefix = f"epoch_{completed_epoch:04d}"
            else:
                title_label = f"Final at Step {global_step}"
                file_prefix = f"final_step_{global_step:08d}"
            images, figure_paths = create_alignment_visualizations(
                result,
                max_pairs=int(cfg.alignment_eval.visualization_max_pairs),
                random_seed=int(cfg.alignment_eval.random_seed),
                title_label=title_label,
                file_prefix=file_prefix,
                output_dir=output_dir / "alignment",
                save_local_figures=bool(cfg.alignment_eval.save_local_figures),
            )
    finally:
        # Evaluation DataLoader와 UMAP을 포함한 모든 평가 RNG 영향을 제거한다.
        restore_rng_state(rng_before_evaluation)

    record: dict[str, Any] = {
        "record_type": "alignment",
        "step": global_step,
        "global_step": global_step,
        "completed_epoch": completed_epoch,
        "evaluation_label": evaluation_label,
        "alignment/evaluation_label": evaluation_label,
        "alignment/is_final": evaluation_label == "final",
        "alignment/sample_count": len(result.tactile),
        "alignment/visualized": visualize,
        "alignment/figure_paths": figure_paths,
        **result.metrics,
    }
    if bool(cfg.comparison.enabled):
        selected = loader.dataset
        query_rows = [selected.dataset.rows[index] for index in selected.indices]
        queries, support = evaluate_alignment_dynamics(result, query_rows)
        for group, values in support.items():
            for metric in ("recall_at_1", "recall_at_5", "mrr"):
                if values[metric] is not None:
                    record[f"alignment/dynamics/{group}/{metric}"] = values[metric]
            record[f"alignment/dynamics/{group}/queries"] = values["queries"]
        record["dynamics_support"] = support
        query_path = output_dir / "alignment" / f"queries_step_{global_step:08d}.parquet"
        write_rows(queries, query_path)
    logger.write(record)
    logger.update_alignment(record)
    wandb_metrics = {
        key: value
        for key, value in record.items()
        if key.startswith("alignment/") and key not in {
            "alignment/figure_paths", "alignment/visualized"
        }
    }
    wandb_metrics["completed_epoch"] = completed_epoch
    wandb_metrics["alignment/evaluation_label"] = evaluation_label
    logger.log_wandb(
        wandb_metrics,
        global_step,
        allow_same_step=True,
        images=images,
    )
    return record


def run_training(cfg: DictConfig) -> None:
    """Run training from a fully composed and validated Hydra config."""
    validate_config(cfg, require_data_root=True)
    train_cfg, data_cfg = cfg.training, cfg.data
    seed = int(train_cfg.seed)
    set_seed(seed)

    comparison = bool(cfg.comparison.enabled)
    if comparison:
        suffix = f"N{int(data_cfg.window_size)}_seed{seed}"
    progress_event("Dataset Initialization")
    source = TaFDataset(
        resolve_path(data_cfg.root_dir),
        window_size=1 if comparison else int(data_cfg.window_size),
        stride=int(data_cfg.stride),
        image_size=tuple(data_cfg.image_size),
        sensor_types=data_cfg.sensor_types,
        max_sequences=data_cfg.max_sequences,
        skip_invalid_sequences=bool(data_cfg.skip_invalid_sequences),
        timestamp_cache_enabled=bool(data_cfg.timestamp_cache_enabled),
        timestamp_cache_dir=(
            resolve_path(data_cfg.timestamp_cache_dir)
            if data_cfg.timestamp_cache_dir is not None
            else None
        ),
        show_initialization_progress=(bool(data_cfg.show_initialization_progress)
                                      and os.environ.get("TAF_TERMINAL_MONITOR") != "1"),
        sensor_cache_max_bytes=int(data_cfg.sensor_cache_max_mb) * 1024 * 1024,
        image_dtype="uint8",
        sensor_cache_shared_stats=bool(cfg.trainctl.performance.enabled and cfg.trainctl.performance.monitor_cache),
        pipeline_shared_stats=bool(cfg.trainctl.performance.enabled and cfg.trainctl.performance.monitor_data_loading),
    )
    progress_event("Dataset Initialization", "Completed", detail=str(source.initialization_stats))
    progress_event("Temporal Alignment Audit", total=len(source.episodes))
    audit_temporal_alignment(source, int(data_cfg.alignment_decode_windows_per_episode))
    progress_event("Temporal Alignment Audit", "Completed", current=len(source.episodes),
                   total=len(source.episodes))
    progress_event("Manifest Loading & Validation", detail="reading manifest" if comparison else "standard mode")
    dataset = ManifestDataset(source, resolve_path(cfg.comparison.manifest), int(data_cfg.window_size)) if comparison else source
    progress_event("Manifest Loading & Validation", "Completed",
                   current=len(dataset.rows) if comparison else None,
                   total=len(dataset.rows) if comparison else None)
    if comparison:
        train_indices = [i for i, r in enumerate(dataset.rows) if r["split"] == "train"]
        val_indices = [i for i, r in enumerate(dataset.rows) if r["split"] == "validation"]
        test_indices = [i for i, r in enumerate(dataset.rows) if r["split"] == "test"]
        if not train_indices or not val_indices or not test_indices:
            raise ValueError("Manifest must contain nonempty train, validation and test splits")
        ids = {id(ep): i for i, ep in enumerate(dataset.episodes)}
        train_eps = {ids[id(dataset._locate(i)[0])] for i in train_indices}
        train_keys = sorted({dataset.rows[i]["sequence_id"] for i in train_indices})
        val_keys = sorted({dataset.rows[i]["sequence_id"] for i in val_indices})
        if int(dataset.rows[0]["split_seed"]) != int(data_cfg.split_seed):
            raise ValueError("Manifest split_seed differs from data.split_seed")
        gallery_path = resolve_path(cfg.comparison.manifest).with_name("gallery_plan.json")
        if not gallery_path.is_file():
            raise FileNotFoundError(f"Comparison gallery plan is required: {gallery_path}")
        gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
        if (gallery.get("manifest_id") != dataset.manifest_id or
            gallery.get("seed") != int(cfg.comparison.gallery_seed) or
            gallery.get("max_pairs") != int(cfg.comparison.gallery_max_pairs) or
            gallery.get("min_gap") != int(cfg.comparison.min_anchor_gap)):
            raise ValueError("Gallery plan differs from comparison settings/manifest")
        gallery_id = hashlib.sha256(gallery_path.read_bytes()).hexdigest()
        comparison_identity = {"manifest_id": dataset.manifest_id, "label_sha256": dataset.rows[0]["label_sha256"],
                               "split_seed": int(data_cfg.split_seed), "window_size": int(data_cfg.window_size),
                               "training_seed": seed, "sampling_seed": int(cfg.comparison.sampling_seed),
                               "gallery_seed": int(cfg.comparison.gallery_seed), "gallery_id": gallery_id,
                               "min_anchor_gap": int(cfg.comparison.min_anchor_gap),
                               "batch_size": int(train_cfg.batch_size),
                               "gradient_accumulation_steps": int(train_cfg.gradient_accumulation_steps)}
        cfg.wandb.name = f"{cfg.wandb.name}_{suffix}_{dataset.manifest_id[:8]}" if cfg.wandb.name else f"{suffix}_{dataset.manifest_id[:8]}"
    else:
        split = split_by_group(dataset, unit=str(data_cfg.split_unit),
                               validation_fraction=float(data_cfg.validation_fraction), seed=int(data_cfg.split_seed))
        train_indices, val_indices, train_eps, _, train_keys, val_keys = split
        comparison_identity = None
    print(
        f"Split by {data_cfg.split_unit}: "
        f"train={len(train_indices)} windows/{len(train_keys)} groups, "
        f"validation={len(val_indices)} windows/{len(val_keys)} groups"
    )

    resume_path = resolve_path(train_cfg.resume) if train_cfg.resume else None
    progress_event("Normalization")
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    if checkpoint is not None:
        if comparison and checkpoint.get("comparison_identity") != comparison_identity:
            raise ValueError("Resume checkpoint comparison identity differs from current manifest/N/seeds")
        saved_split = checkpoint.get("split", {})
        if (
            saved_split.get("train_groups") != train_keys
            or saved_split.get("validation_groups") != val_keys
        ):
            raise ValueError("Current dataset split does not match the resume checkpoint")
        if "normalization" not in checkpoint:
            raise ValueError("Resume checkpoint has no normalization statistics")
        normalization = checkpoint["normalization"]
    else:
        normalization_path = (resolve_path(cfg.comparison.manifest).with_name(
            f"normalization_{dataset.manifest_id[:8]}.pt") if comparison else None)
        if comparison and normalization_path.is_file():
            stored = torch.load(normalization_path, map_location="cpu", weights_only=True)
            if stored.get("manifest_id") != dataset.manifest_id or stored.get("train_groups") != train_keys:
                raise ValueError("Shared normalization identity differs from current manifest/split")
            normalization = stored["normalization"]
        else:
            normalization = compute_normalization(
                dataset, train_eps, int(data_cfg.normalization_chunk_frames),
                float(data_cfg.normalization_eps), selected_indices=train_indices if comparison else None,
                normalization_window_size=max(cfg.comparison.window_sizes) if comparison else None,
            )
            if comparison:
                temporary = normalization_path.with_suffix(".pt.tmp")
                torch.save({"manifest_id": dataset.manifest_id, "train_groups": train_keys,
                            "normalization": normalization}, temporary)
                temporary.replace(normalization_path)

    progress_event("Normalization", "Completed")
    progress_event("Model / Optimizer Initialization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if comparison:
        model = build_model(cfg, tactile_initialization_seed=seed + 10_000)
    else:
        model = build_model(cfg)
    model.set_normalization(**normalization)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )
    amp_enabled = bool(train_cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    step, epoch = 0, 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step, epoch = int(checkpoint["step"]), int(checkpoint.get("epoch", 0))
        restore_rng(checkpoint)
        print(f"Resumed {resume_path} at optimizer step {step}")
    progress_event("Model / Optimizer Initialization", "Completed")

    cached_dataset = dataset
    frame_cache = None
    if bool(data_cfg.frame_cache_enabled):
        cache_dir = (resolve_path(data_cfg.frame_cache_dir) if data_cfg.frame_cache_dir
                     else resolve_path("outputs/frame_cache"))
        frame_cache = FrozenFrameCache(
            source, model, cache_dir, device=device, amp_enabled=amp_enabled,
            batch_size=int(data_cfg.frame_cache_batch_size),
            rebuild=bool(data_cfg.frame_cache_rebuild),
        )
        cached_dataset = CachedWindowDataset(dataset, frame_cache)

    progress_event("Training Sampler Initialization", total=len(train_indices))
    batch_sampler = (FixedValidationBatchSampler if comparison else TemporalDiverseBatchSampler)(
        dataset,
        train_indices,
        batch_size=int(train_cfg.batch_size),
        min_gap=int(train_cfg.min_same_episode_window_gap),
        seed=int(cfg.comparison.sampling_seed) if comparison else seed,
        progress_stage="Training Sampler Initialization",
    )
    progress_event("Training Sampler Initialization", "Completed", current=len(train_indices), total=len(train_indices))
    workers = int(train_cfg.num_workers)
    prefetch = {"prefetch_factor": int(train_cfg.prefetch_factor)} if workers else {}
    train_loader = DataLoader(
        Subset(cached_dataset, train_indices),
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        **prefetch,
    )
    validation_gap = max(
        int(data_cfg.window_size), int(train_cfg.min_same_episode_window_gap),
    )
    progress_event("Validation Sampler Initialization", total=len(val_indices))
    validation_sampler = FixedValidationBatchSampler(
        dataset,
        val_indices,
        batch_size=int(train_cfg.batch_size),
        min_gap=validation_gap,
        seed=int(data_cfg.split_seed),
        progress_stage="Validation Sampler Initialization",
    )
    progress_event("Validation Sampler Initialization", "Completed", current=len(val_indices), total=len(val_indices))
    validation_generator = torch.Generator()
    validation_generator.manual_seed(int(data_cfg.split_seed))
    val_loader = DataLoader(
        Subset(cached_dataset, val_indices),
        batch_sampler=validation_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=validation_generator,
        **prefetch,
    )

    output_dir = resolve_path(train_cfg.output_dir)
    if comparison:
        output_dir = output_dir / f"{dataset.manifest_id[:8]}_N{int(data_cfg.window_size)}_seed{seed}"
        if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Comparison run directory already contains results: {output_dir}; use training.resume")
    saved_alignment = dict(checkpoint.get("alignment", {})) if checkpoint is not None else {}
    completed_epochs = int(
        saved_alignment.get(
            "completed_epochs",
            checkpoint.get("epoch", 0) if checkpoint is not None else 0,
        )
    )
    alignment_state: dict[str, Any] = {
        "completed_epochs": completed_epochs,
        "last_evaluated_step": int(saved_alignment.get("last_evaluated_step", -1)),
        "last_evaluated_completed_epoch": int(
            saved_alignment.get("last_evaluated_completed_epoch", -1)
        ),
        "last_evaluation_label": saved_alignment.get("last_evaluation_label"),
    }
    alignment_loader: DataLoader | None = None
    if bool(cfg.alignment_eval.enabled):
        if comparison:
            lookup = {row["sample_id"]: i for i, row in enumerate(dataset.rows)}
            ids = gallery["validation"]
            if len(ids) != len(set(ids)) or any(s not in lookup for s in ids):
                raise ValueError("Validation gallery plan has duplicate or unknown sample IDs")
            indices = [lookup[s] for s in ids]
            if any(dataset.rows[i]["split"] != "validation" for i in indices):
                raise ValueError("Validation gallery plan contains non-validation samples")
            selection = AlignmentSelection(
                indices=indices,
                samples=[{"dataset_index": i, **dataset.rows[i]} for i in indices],
                random_seed=int(gallery["seed"]),
                requested_max_pairs=int(gallery["max_pairs"]),
                requested_min_gap=int(gallery["min_gap"]),
                effective_min_gap=int(gallery["min_gap"]),
            )
        else:
            selection = select_alignment_indices(
                dataset, val_indices,
                max_pairs=int(cfg.alignment_eval.retrieval_max_pairs),
                min_window_start_gap=int(cfg.alignment_eval.min_window_start_gap),
                random_seed=int(cfg.alignment_eval.random_seed),
                training_indices=train_indices,
            )
        saved_indices = saved_alignment.get("sample_indices")
        if saved_indices is not None and list(saved_indices) != selection.indices:
            raise ValueError(
                "Alignment validation sample indices differ from the resume checkpoint"
            )
        alignment_state.update({
            "sample_indices": selection.indices,
            "random_seed": selection.random_seed,
            "effective_min_window_start_gap": selection.effective_min_gap,
        })
        save_alignment_selection(
            selection,
            output_dir / "alignment" / "validation_selection.json",
        )
        alignment_generator = torch.Generator()
        alignment_generator.manual_seed(selection.random_seed)
        alignment_loader = DataLoader(
            Subset(dataset, selection.indices),
            batch_size=min(int(train_cfg.batch_size), len(selection.indices)),
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=alignment_generator,
            **prefetch,
        )
        print(
            f"Fixed alignment retrieval set: {len(selection.indices)} validation pairs, "
            f"start gap >= {selection.effective_min_gap}, "
            f"seed={selection.random_seed}"
        )

    progress_event("W&B / trainctl Initialization")
    config_dict = resolved_config(cfg)
    logger = ExperimentLogger(
        output_dir,
        cfg.wandb,
        checkpoint.get("wandb") if checkpoint is not None else None,
    )
    if comparison:
        logger.write({"record_type": "comparison", **comparison_identity,
                      "sampling": batch_sampler.report.as_dict(),
                      "force_parameters": sum(p.numel() for p in model.force_encoder.parameters()),
                      "tactile_parameters": sum(p.numel() for p in model.tactile_encoder.parameters())})
    rng_before_wandb = capture_rng()
    try:
        logger.start(cfg, model)
    finally:
        # Experiment tracking must not change training randomness.
        restore_rng_state(rng_before_wandb)

    max_steps = int(train_cfg.max_steps)
    started = time.time()
    prior_duration = float(checkpoint.get("training_duration_seconds", 0.0)) if checkpoint is not None else 0.0
    reporter = TrainctlReporter(enabled=bool(cfg.trainctl.enabled))
    perf = PerformanceMonitor(cfg.trainctl.performance, device, source=source,
                              frame_cache=frame_cache)
    perf.attach_backbone(model.tactile_encoder.backbone)

    def emit_performance_warning(message: str) -> None:
        safe = f"Performance warning at step {step}: {message}"
        logger.message(safe)
        logger.write({"record_type": "performance_warning", "step": step,
                      "window_size": int(data_cfg.window_size), "message": safe})
        if bool(cfg.trainctl.performance.warnings.send_to_trainctl):
            reporter.warn(safe)
    wandb_state = logger.wandb_state() or {}
    run_id = os.environ.get("TRAINCTL_RUN_ID")
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        world_size = 1
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    start_message = build_start_message({
        "window_size": int(data_cfg.window_size),
        "experiment": wandb_state.get("run_name") or cfg.wandb.name,
        "run_id": run_id or wandb_state.get("run_id"),
        "stride": int(data_cfg.stride),
        "batch_size": int(train_cfg.batch_size),
        "accumulation": int(train_cfg.gradient_accumulation_steps),
        "effective_batch_size": (
            int(train_cfg.batch_size)
            * int(train_cfg.gradient_accumulation_steps)
            * world_size
        ),
        "optimizer": "AdamW",
        "learning_rate": float(train_cfg.learning_rate),
        "initial_step": step,
        "total_steps": max_steps,
        "validate_every": int(train_cfg.validate_every),
        "save_every": int(train_cfg.save_every),
        "train_windows": len(train_indices),
        "train_groups": len(train_keys),
        "val_windows": len(val_indices),
        "val_groups": len(val_keys),
        "amp": amp_enabled,
        "device": str(device),
        "gpu": gpu_name,
        "backbone": str(cfg.tactile_encoder.backbone),
        "freeze_backbone": bool(cfg.tactile_encoder.freeze_backbone),
        "comparison": comparison,
        "seed": seed,
        "workers": workers,
        "prefetch_factor": int(train_cfg.prefetch_factor) if workers else None,
        "feature_cache": frame_cache is not None,
        "output_dir": str(output_dir),
        "resume": str(resume_path) if resume_path else None,
        "wandb_url": logger.wandb_url(),
    })
    reporter.start(
        name=wandb_state.get("run_name"), total_steps=max_steps,
        message=start_message,
    )
    progress_event("W&B / trainctl Initialization", "Completed")
    validation_sampling_record: dict[str, Any] = {
        "record_type": "validation_sampling",
        **{
            f"val/sampling_{key}": value
            for key, value in validation_sampler.report.as_dict().items()
        },
    }
    logger.write(validation_sampling_record)
    logger.message(
        "Fixed validation sampling: "
        f"{validation_sampler.report.batch_count} full batches x "
        f"{validation_sampler.report.batch_size}, "
        f"used={validation_sampler.report.used_samples}/"
        f"{validation_sampler.report.candidate_samples}, "
        f"dropped={validation_sampler.report.dropped_samples}, "
        f"same-episode start gap >= {validation_sampler.report.min_window_start_gap}, "
        f"violating_batches={validation_sampler.report.violating_batches}, "
        f"seed={validation_sampler.report.seed}"
    )
    logger.start_progress(max_steps, initial_step=step)
    progress_event("Training Ready", "Completed")
    perf.start_watchdog(emit_performance_warning)
    accumulation = int(train_cfg.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    local_rolling = DeviceMetricAccumulator()
    wandb_rolling = DeviceMetricAccumulator()
    micro_step = step * accumulation
    validation_history = ValidationHistory()
    last_alignment_record: dict[str, Any] | None = None
    last_train_record: dict[str, Any] | None = None
    last_checkpoint_path: Path | None = None
    initial_step = step
    pending_checkpoint: tuple[Path, int, int, dict[str, Any]] | None = None
    last_optimization: dict[str, Any] = {
        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train/gradient_norm": float("nan"),
    }

    def publish_performance_event(name: str, seconds: float, *, wandb: bool = True) -> None:
        perf.record_event(name, seconds)
        if not perf.enabled:
            return
        metric = {f"perf/{name}_seconds": seconds}
        logger.write({"record_type": "performance_event", "step": step, **metric})
        reporter.log(step, metric)
        if wandb:
            logger.log_wandb(metric, step, allow_same_step=True)

    def save_monitored_checkpoint(*args: Any, **kwargs: Any) -> None:
        progress_event("Activity", detail=f"Saving checkpoint at step {step}")
        perf.phase(False)
        begin = time.perf_counter()
        try:
            save_checkpoint(*args, **kwargs)
        except BaseException:
            perf.phase(True)
            raise
        publish_performance_event("checkpoint", time.perf_counter() - begin)
        progress_event("Activity", "Completed", detail="Training")
        perf.phase(True)

    def run_monitored_alignment(**kwargs: Any) -> dict[str, Any]:
        progress_event("Activity", detail=f"Retrieval at step {step}")
        perf.phase(False)
        begin = time.perf_counter()
        try:
            result = _run_alignment_evaluation(**kwargs)
        except BaseException:
            perf.phase(True)
            raise
        publish_performance_event("alignment_retrieval", time.perf_counter() - begin)
        progress_event("Activity", "Completed", detail="Training")
        perf.phase(True)
        return result

    def report_saved_checkpoint(
        path: Path, checkpoint_step: int, checkpoint_epoch: int,
        optimization_metrics: dict[str, Any], aliases: list[str],
    ) -> None:
        nonlocal last_checkpoint_path
        last_checkpoint_path = path
        validation = validation_history.last_metrics
        if validation_history.last_step != checkpoint_step or validation is None:
            raise RuntimeError(
                f"Checkpoint {path} has no validation from step {checkpoint_step}"
            )
        alignment = (
            last_alignment_record
            if last_alignment_record is not None
            and int(last_alignment_record["global_step"]) == checkpoint_step
            else None
        )
        train_summary = dict(last_train_record or {})
        train_summary.update({
            "train/gradient_norm": optimization_metrics["train/gradient_norm"],
            "train/learning_rate": optimization_metrics["train/learning_rate"],
        })
        elapsed = max(0.0, time.time() - started)
        completed_steps = max(0, checkpoint_step - initial_step)
        rate = completed_steps / elapsed if elapsed > 0 else 0.0
        eta = (
            (max_steps - checkpoint_step) / rate
            if rate > 0 and checkpoint_step < max_steps else 0.0
        )
        info = {
            "step": checkpoint_step,
            "total_steps": max_steps,
            "progress": 100.0 * checkpoint_step / max_steps,
            "epoch": checkpoint_epoch,
            "path": path,
            "size_mib": path.stat().st_size / (1024 * 1024),
            "validation": validation,
            "alignment": alignment,
            "train": train_summary,
            "elapsed": elapsed,
            "steps_per_second": rate,
            "eta": eta,
            "best_loss": validation_history.best_loss,
            "best_step": validation_history.best_step,
            "improved": validation_history.last_improved,
            "performance": perf.last_metrics if perf.enabled else None,
        }
        metrics = {**validation, **train_summary, **(perf.last_metrics if perf.enabled else {})}
        if alignment:
            metrics.update({
                key: value for key, value in alignment.items()
                if key.startswith("alignment/")
            })
        reporter.checkpoint(
            path, step=checkpoint_step, metrics=metrics,
            message=build_checkpoint_message(info), aliases=aliases,
        )

    failed = True
    try:
        while step < max_steps:
            batch_sampler.set_epoch(epoch)
            model.train()
            perf.phase(True)
            epoch_completed = False
            # The sampler uses only a local RNG, so counting its deterministic
            # plan cannot perturb the actual DataLoader or training RNG.
            epoch_batch_count = sum(1 for _ in batch_sampler)
            epoch_batch_number = 0
            train_iterator = iter(train_loader)
            while True:
                iteration_start = time.perf_counter() if perf.enabled else 0.0
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    epoch_completed = True
                    break
                if perf.enabled:
                    perf.record_wait(time.perf_counter() - iteration_start)
                epoch_batch_number += 1
                is_last_epoch_batch = epoch_batch_number == epoch_batch_count
                with perf.stage("h2d_transfer"):
                    force = batch["force"].to(device, non_blocking=True)
                    pressure = batch["pressure"].to(device, non_blocking=True)
                    if "backbone_features" in batch:
                        features = batch["backbone_features"].to(device, non_blocking=True)
                    else:
                        tactile = batch["tactile"].to(device, non_blocking=True)
                with perf.stage("model_forward"):
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                    ):
                        if "backbone_features" in batch:
                            outputs = model.forward_from_backbone_features(features, force, pressure)
                        else:
                            outputs = model(tactile=tactile, force=force, pressure=pressure)
                        scaled_loss = outputs["loss_total"] / accumulation
                with perf.stage("backward"):
                    scaler.scale(scaled_loss).backward()
                local_rolling.add(outputs)
                if cfg.wandb.enabled:
                    wandb_rolling.add(outputs)
                micro_step += 1
                if micro_step % accumulation:
                    if perf.enabled:
                        perf.record_iteration(time.perf_counter() - iteration_start,
                                              int(force.shape[0]), int(force.shape[0] * force.shape[1]))
                    continue

                with perf.stage("optimizer"):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        float(train_cfg.grad_clip_norm),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if perf.enabled:
                    perf.record_iteration(time.perf_counter() - iteration_start,
                                          int(force.shape[0]), int(force.shape[0] * force.shape[1]))
                perf.completed_step()
                # Micro-step이 아니라 실제 optimizer.step()과 1:1로 진행한다.
                logger.advance(step)
                elapsed = time.time() - started
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimization = {
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": grad_norm.detach(),
                    "global_step": step,
                    "epoch": epoch,
                    "train/elapsed_time_seconds": elapsed,
                }
                last_optimization = dict(optimization)

                def logged_optimization() -> dict[str, Any]:
                    return {**optimization,
                            "train/gradient_norm": float(optimization["train/gradient_norm"])}

                if step % int(train_cfg.log_every) == 0:
                    local_record: dict[str, Any] = {
                        "step": step,
                        "epoch": epoch,
                        "seconds": elapsed,
                        "learning_rate": learning_rate,
                        "grad_norm": float(grad_norm),
                        **local_rolling.pop_means("train"),
                    }
                    logger.write(local_record)
                    logger.update_training(local_record)
                    last_train_record = dict(local_record)
                    reporter.log(step, {
                        key: value for key, value in local_record.items()
                        if key.startswith("train/")
                    } | {
                        "train/gradient_norm": local_record["grad_norm"],
                        "train/learning_rate": learning_rate,
                    })

                wandb_payload: dict[str, float | int] = {}
                if cfg.wandb.enabled and step % int(cfg.wandb.log_every) == 0:
                    wandb_payload.update(wandb_rolling.pop_means("train"))
                    wandb_payload.update(logged_optimization())

                if perf.enabled and step % int(cfg.trainctl.performance.log_every) == 0:
                    perf_metrics = perf.snapshot(step, max_steps)
                    logger.write({"record_type": "performance", "step": step,
                                  "window_size": int(data_cfg.window_size), **perf_metrics})
                    reporter.log(step, perf_metrics)
                    wandb_payload.update(perf_metrics)
                    perf.warn_if_needed(perf_metrics, step, emit_performance_warning)

                validate_every = int(train_cfg.validate_every)
                checkpoint_due = step % int(train_cfg.save_every) == 0
                scheduled_validation = (
                    validate_every > 0 and step % validate_every == 0
                )
                if (checkpoint_due and perf.enabled
                        and step % int(cfg.trainctl.performance.log_every) != 0):
                    perf_metrics = perf.snapshot(step, max_steps)
                    logger.write({"record_type": "performance", "step": step,
                                  "window_size": int(data_cfg.window_size), **perf_metrics})
                    reporter.log(step, perf_metrics)
                    wandb_payload.update(perf_metrics)
                    perf.warn_if_needed(perf_metrics, step, emit_performance_warning)
                if scheduled_validation or checkpoint_due:
                    def evaluate_current_step() -> dict[str, float]:
                        progress_event("Activity", detail=f"Validation at step {step}")
                        perf.phase(False)
                        validation_start = time.perf_counter()
                        val_metrics = validate(
                            model,
                            val_loader,
                            device,
                            amp_enabled,
                            int(train_cfg.validation_batches)
                            if train_cfg.validation_batches is not None
                            else None,
                        )
                        duration = time.perf_counter() - validation_start
                        publish_performance_event("validation", duration, wandb=False)
                        perf.phase(True)
                        wandb_payload["perf/validation_seconds"] = duration
                        requested_batches = (
                            int(train_cfg.validation_batches)
                            if train_cfg.validation_batches is not None
                            else None
                        )
                        evaluated_batches = min(
                            len(validation_sampler),
                            requested_batches
                            if requested_batches is not None
                            else len(validation_sampler),
                        )
                        diagnostics = {
                            "val/evaluated_batches": evaluated_batches,
                            "val/evaluated_samples": (
                                evaluated_batches * int(train_cfg.batch_size)
                            ),
                            "val/sampling_dropped_samples": (
                                validation_sampler.report.dropped_samples
                            ),
                            "val/sampling_min_window_start_gap": validation_gap,
                        }
                        record = {
                            "step": step, "epoch": epoch,
                            **diagnostics, **val_metrics,
                        }
                        logger.write(record)
                        logger.update_validation(record)
                        progress_event("Activity", "Completed", detail="Training")
                        model.train()
                        return record

                    val_record, validation_ran = validation_history.ensure(
                        step, evaluate_current_step
                    )
                    if validation_ran:
                        val_payload = {
                            key: value for key, value in val_record.items()
                            if key.startswith("val/")
                        }
                        reporter.log(step, val_payload)
                        wandb_payload.update(val_payload)
                        wandb_payload.update(logged_optimization())

                if wandb_payload:
                    logger.log_wandb(wandb_payload, step)

                if should_run_step_alignment(
                    enabled=bool(cfg.alignment_eval.enabled),
                    retrieval_every_n_steps=int(
                        cfg.alignment_eval.retrieval_every_n_steps
                    ),
                    current_step=step,
                    last_evaluated_step=int(alignment_state["last_evaluated_step"]),
                ):
                    assert alignment_loader is not None
                    next_completed_epoch = completed_epochs + 1
                    epoch_due, epoch_visualize = alignment_schedule(
                        next_completed_epoch,
                        retrieval_every_n_epochs=int(
                            cfg.alignment_eval.retrieval_every_n_epochs
                        ),
                        visualize_every_n_epochs=int(
                            cfg.alignment_eval.visualize_every_n_epochs
                        ),
                    )
                    # At an epoch boundary, one epoch-labelled evaluation takes
                    # precedence so its visualization schedule is preserved.
                    evaluation_label = (
                        "epoch" if is_last_epoch_batch and epoch_due else "step"
                    )
                    last_alignment_record = run_monitored_alignment(
                        model=model,
                        loader=alignment_loader,
                        device=device,
                        cfg=cfg,
                        output_dir=output_dir,
                        logger=logger,
                        global_step=step,
                        completed_epoch=(
                            next_completed_epoch
                            if evaluation_label == "epoch"
                            else completed_epochs
                        ),
                        evaluation_label=evaluation_label,
                        visualize=(
                            epoch_visualize if evaluation_label == "epoch" else False
                        ),
                    )
                    alignment_state.update({
                        "last_evaluated_step": step,
                        "last_evaluated_completed_epoch": int(
                            last_alignment_record["completed_epoch"]
                        ),
                        "last_evaluation_label": evaluation_label,
                    })
                    reporter.log(step, {
                        key: value for key, value in last_alignment_record.items()
                        if key.startswith("alignment/")
                    })

                if checkpoint_due:
                    checkpoint_path = output_dir / f"step_{step:08d}.pt"
                    save_monitored_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        epoch=epoch,
                        config=config_dict,
                        normalization=normalization,
                        train_keys=train_keys,
                        val_keys=val_keys,
                        wandb_state=logger.wandb_state(),
                        alignment_state=alignment_state,
                        comparison_identity=comparison_identity,
                        training_duration_seconds=prior_duration + time.time() - started,
                    )
                    last_checkpoint_path = checkpoint_path
                    aliases = ["latest"]
                    if validation_history.last_improved:
                        aliases.append("best")
                    if is_last_epoch_batch:
                        pending_checkpoint = (
                            checkpoint_path, step, epoch + 1,
                            logged_optimization(),
                        )
                    else:
                        report_saved_checkpoint(
                            checkpoint_path, step, epoch + 1,
                            logged_optimization(), aliases,
                        )
                if step >= max_steps:
                    # DataLoader 길이는 custom sampler의 실제 yield 수와 다를 수
                    # 있으므로 한 번 peek해 정말 마지막 batch였는지 판정한다.
                    try:
                        next(train_iterator)
                    except StopIteration:
                        epoch_completed = True
                    break

            if epoch_completed:
                completed_epochs += 1
                alignment_state["completed_epochs"] = completed_epochs
                should_evaluate, should_visualize = alignment_schedule(
                    completed_epochs,
                    retrieval_every_n_epochs=int(
                        cfg.alignment_eval.retrieval_every_n_epochs
                    ),
                    visualize_every_n_epochs=int(
                        cfg.alignment_eval.visualize_every_n_epochs
                    ),
                )
                if should_run_epoch_alignment(
                    enabled=bool(cfg.alignment_eval.enabled),
                    scheduled=should_evaluate,
                    current_step=step,
                    last_evaluated_step=int(alignment_state["last_evaluated_step"]),
                ):
                    assert alignment_loader is not None
                    last_alignment_record = run_monitored_alignment(
                        model=model,
                        loader=alignment_loader,
                        device=device,
                        cfg=cfg,
                        output_dir=output_dir,
                        logger=logger,
                        global_step=step,
                        completed_epoch=completed_epochs,
                        evaluation_label="epoch",
                        visualize=should_visualize,
                    )
                    alignment_state.update({
                        "last_evaluated_step": step,
                        "last_evaluated_completed_epoch": completed_epochs,
                        "last_evaluation_label": "epoch",
                    })
                    reporter.log(step, {
                        key: value for key, value in last_alignment_record.items()
                        if key.startswith("alignment/")
                    })
            epoch += 1
            if epoch_completed and step % int(train_cfg.save_every) == 0:
                # Epoch 마지막 step에서 먼저 저장된 numbered checkpoint를
                # completed epoch/evaluation/W&B 상태로 갱신한다.
                save_monitored_checkpoint(
                    output_dir / f"step_{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    epoch=epoch,
                    config=config_dict,
                    normalization=normalization,
                    train_keys=train_keys,
                    val_keys=val_keys,
                    wandb_state=logger.wandb_state(),
                    alignment_state=alignment_state,
                    comparison_identity=comparison_identity,
                    training_duration_seconds=prior_duration + time.time() - started,
                )
                last_checkpoint_path = output_dir / f"step_{step:08d}.pt"
                if pending_checkpoint is not None:
                    path_value, step_value, epoch_value, optimization_value = (
                        pending_checkpoint
                    )
                    aliases = ["latest"]
                    if validation_history.last_improved:
                        aliases.append("best")
                    report_saved_checkpoint(
                        path_value, step_value, epoch_value,
                        optimization_value, aliases,
                    )
                    pending_checkpoint = None

        if should_run_final_alignment(
            enabled=bool(cfg.alignment_eval.enabled),
            run_on_training_end=bool(cfg.alignment_eval.run_on_training_end),
            current_step=step,
            last_evaluated_step=int(alignment_state["last_evaluated_step"]),
        ):
            assert alignment_loader is not None
            last_alignment_record = run_monitored_alignment(
                model=model,
                loader=alignment_loader,
                device=device,
                cfg=cfg,
                output_dir=output_dir,
                logger=logger,
                global_step=step,
                completed_epoch=completed_epochs,
                evaluation_label="final",
                visualize=False,
            )
            alignment_state.update({
                "last_evaluated_step": step,
                "last_evaluated_completed_epoch": completed_epochs,
                "last_evaluation_label": "final",
            })
            reporter.log(step, {
                key: value for key, value in last_alignment_record.items()
                if key.startswith("alignment/")
            })

        def evaluate_final_step() -> dict[str, float]:
            progress_event("Activity", detail=f"Final validation at step {step}")
            perf.phase(False)
            validation_start = time.perf_counter()
            val_metrics = validate(
                model, val_loader, device, amp_enabled,
                int(train_cfg.validation_batches)
                if train_cfg.validation_batches is not None else None,
            )
            publish_performance_event("validation", time.perf_counter() - validation_start)
            perf.phase(True)
            requested_batches = (
                int(train_cfg.validation_batches)
                if train_cfg.validation_batches is not None else None
            )
            evaluated_batches = min(
                len(validation_sampler),
                requested_batches
                if requested_batches is not None else len(validation_sampler),
            )
            diagnostics = {
                "val/evaluated_batches": evaluated_batches,
                "val/evaluated_samples": (
                    evaluated_batches * int(train_cfg.batch_size)
                ),
                "val/sampling_dropped_samples": (
                    validation_sampler.report.dropped_samples
                ),
                "val/sampling_min_window_start_gap": validation_gap,
            }
            record = {"step": step, "epoch": epoch, **diagnostics, **val_metrics}
            logger.write(record)
            logger.update_validation(record)
            progress_event("Activity", "Completed", detail="Training")
            model.train()
            return record

        final_validation, final_validation_ran = validation_history.ensure(
            step, evaluate_final_step
        )
        if final_validation_ran:
            final_val_payload = {
                key: value for key, value in final_validation.items()
                if key.startswith("val/")
            }
            reporter.log(step, final_val_payload)
            logger.log_wandb(
                {**final_val_payload, **last_optimization,
                 "train/gradient_norm": float(last_optimization["train/gradient_norm"])},
                step, allow_same_step=True,
            )

        if perf.enabled and perf.interval["steps"] > 0:
            final_interval = perf.snapshot(step, max_steps)
            logger.write({"record_type": "performance", "step": step,
                          "window_size": int(data_cfg.window_size), **final_interval})
            reporter.log(step, final_interval)
            logger.log_wandb(final_interval, step, allow_same_step=True)

        last_path = output_dir / "last.pt"
        save_monitored_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            epoch=epoch,
            config=config_dict,
            normalization=normalization,
            train_keys=train_keys,
            val_keys=val_keys,
            wandb_state=logger.wandb_state(),
            alignment_state=alignment_state,
            comparison_identity=comparison_identity,
            training_duration_seconds=prior_duration + time.time() - started,
        )
        last_checkpoint_path = last_path
        last_aliases = ["last", "latest"]
        if validation_history.last_improved:
            last_aliases.append("best")
        report_saved_checkpoint(
            last_path, step, epoch,
            {**last_optimization,
             "train/gradient_norm": float(last_optimization["train/gradient_norm"])},
            last_aliases,
        )
        perf.phase(False)
        performance_summary = None
        if perf.enabled:
            final_perf = perf.snapshot(step, max_steps)
            if final_perf:
                logger.write({"record_type": "performance", "step": step,
                              "window_size": int(data_cfg.window_size), **final_perf})
                reporter.log(step, final_perf)
                logger.log_wandb(final_perf, step, allow_same_step=True)
            performance_summary = perf.summary(
                window_size=int(data_cfg.window_size), seed=seed,
                train_windows=len(train_indices),
                hardware={"host": platform.node(), "machine": platform.machine(),
                          "processor": platform.processor(), "torch": torch.__version__,
                          "device": str(device), "gpu": gpu_name},
            )
            performance_summary.update({
                "comparison_mode": comparison,
                "manifest_id": dataset.manifest_id if comparison else None,
                "initial_step": initial_step, "final_step": step,
                "target_steps": max_steps,
                "training_duration_seconds": prior_duration + time.time() - started,
                "checkpoint": str(last_path),
                "batch_size": int(train_cfg.batch_size),
                "gradient_accumulation_steps": accumulation,
                "num_workers": workers,
                "prefetch_factor": int(train_cfg.prefetch_factor) if workers else None,
                "amp": amp_enabled,
                "feature_cache_enabled": frame_cache is not None,
                "split_unit": str(data_cfg.split_unit),
                "loss_mode": str(cfg.loss.nce_mode),
                "split_seed": int(data_cfg.split_seed),
                "stride": int(data_cfg.stride),
                "image_size": list(data_cfg.image_size),
                "backbone": str(cfg.tactile_encoder.backbone),
            })
            summary_metrics = {
                "perf/average_steps_per_second": performance_summary["average_steps_per_second"],
                "perf/average_frames_per_second": performance_summary["average_frames_per_second"],
                "perf/total_training_seconds": performance_summary["training_duration_seconds"],
            }
            summary_metrics.update({
                f"perf/{name}_total_seconds": seconds
                for name, seconds in performance_summary["evaluation_total_seconds"].items()
            })
            summary_metrics = {key: value for key, value in summary_metrics.items()
                               if value is not None}
            logger.write({"record_type": "performance_summary", "step": step,
                          **summary_metrics})
            reporter.log(step, summary_metrics)
            logger.log_wandb(summary_metrics, step, allow_same_step=True)
            with (output_dir / "performance_summary.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(performance_summary, sort_keys=True) + "\n")
        logger.close_progress()
        logger.message(
            f"Training complete at step {step}; checkpoint: {last_path}"
        )
        final_wandb_url = logger.wandb_url()
        logger.finish(exit_code=0)
        current_alignment = (
            last_alignment_record
            if last_alignment_record is not None
            and int(last_alignment_record["global_step"]) == step
            else None
        )
        reporter.finish(build_finish_message({
            "step": step,
            "total_steps": max_steps,
            "elapsed": time.time() - started,
            "best_loss": validation_history.best_loss,
            "best_step": validation_history.best_step,
            "validation": final_validation,
            "alignment": current_alignment,
            "checkpoint_count": reporter.checkpoint_count,
            "last_checkpoint": last_path,
            "wandb_url": final_wandb_url,
            "window_size": int(data_cfg.window_size),
            "performance": performance_summary,
        }))
        failed = False
    except BaseException as exc:
        reporter.fail(build_failure_message(
            exc,
            step=step,
            window_size=int(data_cfg.window_size),
            checkpoint=last_checkpoint_path,
            validation=validation_history.last_metrics,
            elapsed=time.time() - started,
            performance=perf.failure_metrics() if perf.enabled else None,
        ))
        raise
    finally:
        perf.close()
        if failed:
            logger.finish(exit_code=1)


@hydra.main(version_base="1.3", config_path="../config", config_name="taf_adapter")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
