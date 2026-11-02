"""Pre-train TaF-Adapter on temporally aligned TaF-Dataset episodes."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset
from utils.alignment import (
    alignment_schedule,
    create_alignment_visualizations,
    evaluate_alignment,
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
from utils.logging import ExperimentLogger, scalar_metrics
from utils.model_factory import build_model
from utils.trainctl import (
    TrainctlReporter,
    ValidationHistory,
    build_checkpoint_message,
    build_failure_message,
    build_finish_message,
    build_start_message,
)


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
        tactile = batch["tactile"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        pressure = batch["pressure"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(tactile=tactile, force=force, pressure=pressure)
        for key, value in scalar_metrics(outputs).items():
            sums[key] += value
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val/{key}": value / batches for key, value in sums.items()}


def _mean_metrics(values: dict[str, list[float]], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": sum(items) / len(items) for key, items in values.items()}


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

    dataset = TaFDataset(
        resolve_path(data_cfg.root_dir),
        window_size=int(data_cfg.window_size),
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
        show_initialization_progress=bool(data_cfg.show_initialization_progress),
    )
    audit_temporal_alignment(dataset, int(data_cfg.alignment_decode_windows_per_episode))
    split = split_by_group(
        dataset,
        unit=str(data_cfg.split_unit),
        validation_fraction=float(data_cfg.validation_fraction),
        seed=int(data_cfg.split_seed),
    )
    train_indices, val_indices, train_eps, _, train_keys, val_keys = split
    print(
        f"Split by {data_cfg.split_unit}: "
        f"train={len(train_indices)} windows/{len(train_keys)} groups, "
        f"validation={len(val_indices)} windows/{len(val_keys)} groups"
    )

    resume_path = resolve_path(train_cfg.resume) if train_cfg.resume else None
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    if checkpoint is not None:
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
        normalization = compute_normalization(
            dataset,
            train_eps,
            int(data_cfg.normalization_chunk_frames),
            float(data_cfg.normalization_eps),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    batch_sampler = TemporalDiverseBatchSampler(
        dataset,
        train_indices,
        batch_size=int(train_cfg.batch_size),
        min_gap=int(train_cfg.min_same_episode_window_gap),
        seed=seed,
    )
    workers = int(train_cfg.num_workers)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_gap = max(
        int(data_cfg.window_size), int(train_cfg.min_same_episode_window_gap),
    )
    validation_sampler = FixedValidationBatchSampler(
        dataset,
        val_indices,
        batch_size=int(train_cfg.batch_size),
        min_gap=validation_gap,
        seed=int(data_cfg.split_seed),
    )
    validation_generator = torch.Generator()
    validation_generator.manual_seed(int(data_cfg.split_seed))
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_sampler=validation_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=validation_generator,
    )

    output_dir = resolve_path(train_cfg.output_dir)
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
        selection = select_alignment_indices(
            dataset,
            val_indices,
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
            "random_seed": int(cfg.alignment_eval.random_seed),
            "effective_min_window_start_gap": selection.effective_min_gap,
        })
        save_alignment_selection(
            selection,
            output_dir / "alignment" / "validation_selection.json",
        )
        alignment_generator = torch.Generator()
        alignment_generator.manual_seed(int(cfg.alignment_eval.random_seed))
        alignment_loader = DataLoader(
            Subset(dataset, selection.indices),
            batch_size=min(int(train_cfg.batch_size), len(selection.indices)),
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=alignment_generator,
        )
        print(
            f"Fixed alignment retrieval set: {len(selection.indices)} validation pairs, "
            f"start gap >= {selection.effective_min_gap}, "
            f"seed={selection.random_seed}"
        )

    config_dict = resolved_config(cfg)
    logger = ExperimentLogger(
        output_dir,
        cfg.wandb,
        checkpoint.get("wandb") if checkpoint is not None else None,
    )
    rng_before_wandb = capture_rng()
    try:
        logger.start(cfg, model)
    finally:
        # Experiment tracking must not change training randomness.
        restore_rng_state(rng_before_wandb)

    max_steps = int(train_cfg.max_steps)
    started = time.time()
    reporter = TrainctlReporter()
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
        "output_dir": str(output_dir),
        "resume": str(resume_path) if resume_path else None,
        "wandb_url": logger.wandb_url(),
    })
    reporter.start(
        name=wandb_state.get("run_name"), total_steps=max_steps,
        message=start_message,
    )
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
    accumulation = int(train_cfg.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    local_rolling: dict[str, list[float]] = defaultdict(list)
    wandb_rolling: dict[str, list[float]] = defaultdict(list)
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
        }
        metrics = {**validation, **train_summary}
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
            epoch_completed = False
            # The sampler uses only a local RNG, so counting its deterministic
            # plan cannot perturb the actual DataLoader or training RNG.
            epoch_batch_count = sum(1 for _ in batch_sampler)
            epoch_batch_number = 0
            train_iterator = iter(train_loader)
            while True:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    epoch_completed = True
                    break
                epoch_batch_number += 1
                is_last_epoch_batch = epoch_batch_number == epoch_batch_count
                tactile = batch["tactile"].to(device, non_blocking=True)
                force = batch["force"].to(device, non_blocking=True)
                pressure = batch["pressure"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    outputs = model(tactile=tactile, force=force, pressure=pressure)
                    scaled_loss = outputs["loss_total"] / accumulation
                scaler.scale(scaled_loss).backward()
                batch_metrics = scalar_metrics(outputs)
                for key, value in batch_metrics.items():
                    local_rolling[key].append(value)
                    if cfg.wandb.enabled:
                        wandb_rolling[key].append(value)
                micro_step += 1
                if micro_step % accumulation:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(train_cfg.grad_clip_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                # Micro-step이 아니라 실제 optimizer.step()과 1:1로 진행한다.
                logger.advance(step)
                elapsed = time.time() - started
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimization = {
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": float(grad_norm),
                    "global_step": step,
                    "epoch": epoch,
                    "train/elapsed_time_seconds": elapsed,
                }
                last_optimization = dict(optimization)

                if step % int(train_cfg.log_every) == 0:
                    local_record: dict[str, Any] = {
                        "step": step,
                        "epoch": epoch,
                        "seconds": elapsed,
                        "learning_rate": learning_rate,
                        "grad_norm": float(grad_norm),
                        **_mean_metrics(local_rolling, "train"),
                    }
                    logger.write(local_record)
                    logger.update_training(local_record)
                    last_train_record = dict(local_record)
                    reporter.log(step, {
                        key: value for key, value in local_record.items()
                        if key.startswith("train/")
                    } | {
                        "train/gradient_norm": float(grad_norm),
                        "train/learning_rate": learning_rate,
                    })
                    local_rolling.clear()

                wandb_payload: dict[str, float | int] = {}
                if cfg.wandb.enabled and step % int(cfg.wandb.log_every) == 0:
                    wandb_payload.update(_mean_metrics(wandb_rolling, "train"))
                    wandb_payload.update(optimization)
                    wandb_rolling.clear()

                validate_every = int(train_cfg.validate_every)
                checkpoint_due = step % int(train_cfg.save_every) == 0
                scheduled_validation = (
                    validate_every > 0 and step % validate_every == 0
                )
                if scheduled_validation or checkpoint_due:
                    def evaluate_current_step() -> dict[str, float]:
                        val_metrics = validate(
                            model,
                            val_loader,
                            device,
                            amp_enabled,
                            int(train_cfg.validation_batches)
                            if train_cfg.validation_batches is not None
                            else None,
                        )
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
                        wandb_payload.update(optimization)

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
                    last_alignment_record = _run_alignment_evaluation(
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
                    save_checkpoint(
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
                    )
                    last_checkpoint_path = checkpoint_path
                    aliases = ["latest"]
                    if validation_history.last_improved:
                        aliases.append("best")
                    if is_last_epoch_batch:
                        pending_checkpoint = (
                            checkpoint_path, step, epoch + 1,
                            dict(optimization),
                        )
                    else:
                        report_saved_checkpoint(
                            checkpoint_path, step, epoch + 1,
                            optimization, aliases,
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
                    last_alignment_record = _run_alignment_evaluation(
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
                save_checkpoint(
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
            last_alignment_record = _run_alignment_evaluation(
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
            val_metrics = validate(
                model, val_loader, device, amp_enabled,
                int(train_cfg.validation_batches)
                if train_cfg.validation_batches is not None else None,
            )
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
                {**final_val_payload, **last_optimization},
                step, allow_same_step=True,
            )

        last_path = output_dir / "last.pt"
        save_checkpoint(
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
        )
        last_checkpoint_path = last_path
        last_aliases = ["last", "latest"]
        if validation_history.last_improved:
            last_aliases.append("best")
        report_saved_checkpoint(
            last_path, step, epoch, last_optimization, last_aliases,
        )
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
        ))
        raise
    finally:
        if failed:
            logger.finish(exit_code=1)


@hydra.main(version_base="1.3", config_path="../config", config_name="taf_adapter")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
