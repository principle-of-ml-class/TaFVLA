"""User-run comparison using the training manifest, normalization and fixed batches.

No checkpoints, W&B or trainctl messages are produced. CUDA synchronizes only at
phase/report boundaries. Full-data initialization and cache builds are user-owned.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, default_collate

from dataset.taf_dataset import TaFDataset
from utils.common import set_seed
from utils.config import register_config_schema
from utils.data import FixedValidationBatchSampler
from utils.dynamics import ManifestDataset
from utils.frame_cache import CachedWindowDataset, FrozenFrameCache
from utils.model_factory import build_model
from utils.performance import PerformanceMonitor, process_resources
from utils.pipeline import PipelineCollate, PipelineDataset
from utils.sensor_cache import SensorMmapCache


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure(source, mode, args):
    source._close_video()
    mmap_cache = getattr(source, "_sensor_mmap", None)
    if mmap_cache is not None:
        mmap_cache.close()
    source._sensor_cache.clear()
    source._sensor_cache_bytes = 0
    source.sensor_arrow_conversion = mode == "optimized"
    source.video_cache_max_entries = args.video_cache_entries if mode == "optimized" else 0


def _step(model, batch, optimizer, scaler, cfg, device, amp, perf):
    with perf.stage("h2d_transfer"):
        force = batch["force"].to(device, non_blocking=True)
        pressure = batch["pressure"].to(device, non_blocking=True)
        key = "backbone_features" if "backbone_features" in batch else "tactile"
        value = batch[key].to(device, non_blocking=True)
    with perf.stage("model_forward"):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = (model.forward_from_backbone_features(value, force, pressure)
                      if key == "backbone_features" else model(value, force, pressure))
    with perf.stage("backward"):
        scaler.scale(output["loss_total"]).backward()
    with perf.stage("optimizer"):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                     float(cfg.training.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return int(force.shape[0]), int(force.shape[0] * force.shape[1])


def _run(model, dataset, source, indices, plan, initial_state, cfg, args, device, mode, workers):
    _configure(source, mode, args)
    set_seed(args.seed)
    model.load_state_dict(initial_state)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(cfg.training.learning_rate),
                                 weight_decay=float(cfg.training.weight_decay))
    amp = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    prepared = PipelineDataset(dataset, source, group_reads=mode == "optimized" and args.group_reads)
    kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda",
              "persistent_workers": workers > 0, "collate_fn": PipelineCollate(source, detailed=True)}
    if workers:
        kwargs.update(prefetch_factor=args.prefetch_factor, multiprocessing_context="spawn")
    loader = DataLoader(Subset(prepared, indices), batch_sampler=plan, **kwargs)
    perf = PerformanceMonitor(cfg.trainctl.performance, device, source=source)
    perf.attach_backbone(model.tactile_encoder.backbone)
    perf.attach_backbone(model.tactile_encoder.temporal, name="temporal_forward")
    iterator = None
    try:
        start = time.perf_counter()
        iterator = iter(loader)
        startup = time.perf_counter() - start
        first_wait = None
        warmup_start = time.perf_counter()
        for _ in range(args.warmup_steps):
            start = time.perf_counter()
            batch = next(iterator)
            if first_wait is None:
                first_wait = time.perf_counter() - start
            _step(model, batch, optimizer, scaler, cfg, device, amp, perf)
        _sync(device)
        warmup = time.perf_counter() - warmup_start
        perf.snapshot(0, args.steps)  # Exclude all startup/warmup pipeline work.
        worker_pids = [w.pid for w in getattr(iterator, "_workers", [])]
        resources_before = process_resources(worker_pids)
        source_before = source.sensor_cache_total_stats()
        ids = hashlib.sha256()
        sample_ids, frame_ids = hashlib.sha256(), hashlib.sha256()
        peak_rss = 0
        peak_pss = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            iteration = time.perf_counter()
            batch = next(iterator)
            perf.record_wait(time.perf_counter() - iteration)
            perf.record_batch(batch)
            samples, frames = _step(model, batch, optimizer, scaler, cfg, device, amp, perf)
            perf.record_iteration(time.perf_counter() - iteration, samples, frames)
            perf.completed_step()
            metadata = batch["metadata"]
            encoded_ids = json.dumps(metadata["sample_id"], separators=(",", ":")).encode()
            encoded_frames = metadata["frame_indices"].numpy().tobytes()
            ids.update(encoded_ids)
            ids.update(encoded_frames)
            sample_ids.update(encoded_ids)
            frame_ids.update(encoded_frames)
            pids = [os.getpid(), *worker_pids]
            resources = process_resources(pids, include_pss=True)
            peak_rss = max(peak_rss, sum(r.get("VmRSS", 0) for r in resources.values()))
            if len(resources) == len(pids) and all("Pss" in r for r in resources.values()):
                peak_pss = max(peak_pss or 0, sum(r["Pss"] for r in resources.values()))
        _sync(device)
        elapsed = time.perf_counter() - start
        # End-to-end wall time includes final GPU completion and memory sampling.
        perf.interval["active_seconds"] = elapsed
        metrics = perf.snapshot(args.steps, args.steps)
        resources_after = process_resources(worker_pids)
        cpu = {str(pid): max(0.0, r["cpu_seconds"] - resources_before.get(pid, {}).get("cpu_seconds", 0))
               / elapsed * 100 for pid, r in resources_after.items()}
        after = source.sensor_cache_total_stats()
        hits = after["cache_hits"] - source_before["cache_hits"]
        misses = after["cache_misses"] - source_before["cache_misses"]
        return {"mode": mode, "workers": workers, "N": dataset.window_size,
                "cache": isinstance(dataset, CachedWindowDataset), "batch_size": args.batch_size,
                "iterator_startup_seconds": startup, "first_batch_wait_seconds": first_wait,
                "warmup_seconds": warmup, "steady_seconds": elapsed,
                "seconds_per_step": elapsed / args.steps,
                "sample_frame_order_sha256": ids.hexdigest(),
                "sample_id_order_sha256": sample_ids.hexdigest(), "frame_indices_sha256": frame_ids.hexdigest(),
                "cpu_peak_sampled_rss_bytes": peak_rss,
                "cpu_peak_sampled_pss_bytes": peak_pss,
                "worker_peak_rss_bytes": {str(pid): r.get("VmHWM") for pid, r in resources_after.items()},
                "worker_cpu_pct": cpu,
                "sensor_hit_rate": hits / (hits + misses) if hits + misses else None,
                "gpu_utilization_note": "report-boundary sample, not time average",
                **metrics}
    finally:
        perf.close()
        if iterator is not None and workers:
            iterator._shutdown_workers()
        source._close_video()
        mmap_cache = getattr(source, "_sensor_mmap", None)
        if mmap_cache is not None:
            mmap_cache.close()
        if isinstance(dataset, CachedWindowDataset):
            dataset.cache.close()


def _check_raw(dataset, source, indices, plan, args):
    selected = list(itertools.islice((indices[i] for batch in plan for i in batch), args.check_samples))
    _configure(source, "baseline", args)
    reference = [dataset[i] for i in selected]
    _configure(source, "optimized", args)
    prepared = PipelineDataset(dataset, source, group_reads=args.group_reads)
    candidates = prepared.__getitems__(selected)
    for left, right in zip(reference, candidates, strict=True):
        for key in ("tactile", "force", "pressure"):
            if not torch.equal(left[key], right[key]):
                raise AssertionError(f"Raw {key} differs")
        for key, value in left["metadata"].items():
            other = right["metadata"][key]
            equal = torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
            if not equal:
                raise AssertionError(f"Metadata {key} differs")
    source._close_video()
    return {"checked_samples": len(selected), "tensor_metadata_bitwise_equal": True}


def _equivalence(model, raw_dataset, cached, indices, device, amp):
    raw = default_collate([raw_dataset[index] for index in indices])
    stored = default_collate([cached[index] for index in indices])
    for key in ("force", "pressure"):
        if not torch.equal(raw[key], stored[key]):
            raise AssertionError(f"Cached sensor tensor differs: {key}")
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        force, pressure = raw["force"].to(device), raw["pressure"].to(device)
        tactile = raw["tactile"].to(device)
        direct_features = model.tactile_encoder.encode_backbone(tactile)
        cached_features = stored["backbone_features"].to(device)
        direct = model(tactile, force, pressure)
        indirect = model.forward_from_backbone_features(cached_features, force, pressure)
    atol = rtol = 1e-3 if amp else 1e-5
    pairs = {"backbone": (direct_features, cached_features)}
    pairs.update({key: (direct[key], indirect[key]) for key in ("z_tactile", "loss_total", "loss_nce")})
    errors = {}
    for key, (left, right) in pairs.items():
        errors[key] = float((left.float() - right.float()).abs().max())
        if not torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol):
            raise AssertionError(f"Cached {key} differs: max_abs={errors[key]}")
    return {"max_abs_errors": errors, "atol": atol, "rtol": rtol,
            "checked_batch_size": len(indices),
            "note": "full InfoNCE batch numerical check; not bitwise or full-training equivalence"}


def _sensor_sources(source):
    return sorted({ep.parquet for ep in source.all_episodes})


def _new_sensor_cache(source, directory, cfg, args):
    directory = Path(directory).expanduser().resolve()
    data_root = source.root_dir if source.root_dir.name == "taf_dataset" else source.root_dir / "taf_dataset"
    if directory.is_relative_to(data_root.resolve()):
        raise ValueError("Sensor cache must be outside the original TaF-Dataset")
    return SensorMmapCache(directory, max_open_files=int(cfg.data.sensor_mmap_max_open_files),
        batch_rows=int(cfg.data.sensor_mmap_batch_rows), rebuild=args.sensor_mmap_rebuild,
        shared_stats=True, stats_enabled=True, profile_page_faults=True)


def _check_sensor_rows(dataset, source, indices, plan, args):
    cache = source._sensor_mmap
    selected = list(itertools.islice((indices[i] for batch in plan for i in batch), args.check_samples))
    try:
        source._sensor_mmap = None
        reference = []
        for i in selected:
            ep, start = dataset._locate(i)
            offsets = np.arange(start, start + dataset.window_size, dtype=np.int64)
            reference.append(source._sensor_rows(ep, offsets))
        source._sensor_mmap = cache
        for i, (force, pressure) in zip(selected, reference, strict=True):
            ep, start = dataset._locate(i)
            offsets = np.arange(start, start + dataset.window_size, dtype=np.int64)
            actual_force, actual_pressure = source._sensor_rows(ep, offsets)
            if force.tobytes() != actual_force.tobytes() or pressure.tobytes() != actual_pressure.tobytes():
                raise AssertionError("Sensor mmap tensors differ from Parquet")
        return {"checked_samples": len(selected), "sensor_tensors_bitwise_equal": True}
    finally:
        source._sensor_mmap = cache
        cache.close()


def _sensor_benchmark(model, dataset, source, indices, plan, initial_state, cfg, args,
                      device, expected_order, digest, emit):
    """Only the sensor backend changes; OS page cache is deliberately uncontrolled."""
    paths = sorted({dataset._locate(indices[i])[0].parquet for batch in plan for i in batch})
    full_estimate = SensorMmapCache.estimate(_sensor_sources(source))
    emit({"record_type": "sensor_mmap_estimate", **full_estimate})
    cases = ["parquet", "mmap_cold", "mmap_warm"] if args.sensor_mmap == "compare" else ["mmap_existing"]
    rows = []
    for workers in args.workers:
        if args.sensor_mmap == "compare":
            root = args.sensor_mmap_dir or args.output_dir / "sensor_mmap"
            directory = root / f"{args.output_dir.name}_N{dataset.window_size}_workers{workers}"
            directory.mkdir(parents=True, exist_ok=False)
        else:
            directory = args.sensor_mmap_dir or Path(cfg.data.sensor_mmap_dir)
        cold = None
        for case in cases:
            case_started = time.perf_counter()
            source._sensor_mmap = None
            preparation = None
            if case != "parquet":
                store = _new_sensor_cache(source, directory, cfg, args)
                # Warm means reuse the same publication, even when a rebuild was requested for cold.
                if case == "mmap_warm":
                    store.rebuild = False
                source._sensor_mmap = store
                preparation = store.prepare(paths)
                if case == "mmap_warm" and preparation["stats"]["builds"]:
                    raise RuntimeError("Warm cache required a rebuild; comparison is not valid")
                if case == "mmap_cold":
                    cold = preparation
                emit({"record_type": "sensor_mmap_preparation", "sensor_case": case,
                      "N": dataset.window_size, "workers": workers, "cache_dir": str(directory),
                      "scope": "complete source files touched by this benchmark plan", **preparation,
                      "os_page_cache": "uncontrolled; creating NPY files also populates OS page cache"})
            row = _run(model, dataset, source, indices, plan, initial_state, cfg, args, device, "optimized", workers)
            if row["sample_frame_order_sha256"] != expected_order:
                raise AssertionError("Sensor backend changed the consumed sample/frame order")
            row.update(record_type="benchmark", sensor_case=case, plan_sha256=digest,
                       manifest_id=dataset.manifest_id, sensor_cache_mb=args.sensor_cache_mb,
                       sensor_mmap_dir=str(directory) if preparation else None,
                       sensor_mmap_preparation=preparation,
                       end_to_end_case_seconds=time.perf_counter() - case_started,
                       os_page_cache="uncontrolled; no OS page cache was dropped")
            if preparation is not None:
                row["perf/sensor_mmap_disk_bytes"] = preparation["disk_bytes"]
                row["perf/sensor_mmap_preparation_seconds"] = preparation["prepare_seconds"]
                # Correctness reads run AFTER timing to avoid pre-touching measured row blocks.
                row["sensor_correctness"] = _check_sensor_rows(dataset, source, indices, plan, args)
            emit(row)
            rows.append(row)
            source._sensor_mmap = None
        if cold is not None:
            pair = {row["sensor_case"]: row for row in rows if row["workers"] == workers}
            saved = pair["parquet"]["seconds_per_step"] - pair["mmap_warm"]["seconds_per_step"]
            build = cold["stats"]["build_seconds"]
            # Extrapolation is explicitly an estimate, not a full-cache measurement.
            full_build = build * full_estimate["rows"] / cold["rows"] if cold["rows"] else None
            emit({"record_type": "sensor_mmap_comparison", "N": dataset.window_size, "workers": workers,
                  "parquet_seconds_per_step": pair["parquet"]["seconds_per_step"],
                  "mmap_cold_seconds_per_step": pair["mmap_cold"]["seconds_per_step"],
                  "mmap_warm_seconds_per_step": pair["mmap_warm"]["seconds_per_step"],
                  "measured_steady_speedup": pair["parquet"]["seconds_per_step"] / pair["mmap_warm"]["seconds_per_step"],
                  "subset_build_seconds": build, "full_build_seconds_estimated": full_build,
                  "subset_break_even_steps": build / saved if saved > 0 else None,
                  "full_break_even_steps_estimated": full_build / saved if saved > 0 and full_build is not None else None,
                  "sample_frame_order_equal": True,
                  "note": "cold refers to an absent application cache before preparation, not a cold OS page cache; full cost is extrapolated"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--batch-size", type=int, choices=[256], default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, nargs="+", default=[2])
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sensor-cache-mb", type=int, default=256)
    parser.add_argument("--video-cache-entries", type=int, default=2)
    parser.add_argument("--group-reads", action="store_true")
    parser.add_argument("--mode", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--cache", choices=["off", "on", "both"], default="off")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-samples", type=int, default=4)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--cache-build-fps", type=float)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--sensor-mmap", choices=["off", "on", "compare"], default="off")
    parser.add_argument("--sensor-mmap-dir", type=Path)
    parser.add_argument("--sensor-mmap-rebuild", action="store_true")
    parser.add_argument("--sensor-mmap-build-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps, args.check_samples, args.prefetch_factor, args.feature_dim) < 1:
        parser.error("steps/warmup/check-samples/prefetch-factor/feature-dim must be positive")
    if min(args.workers) < 0 or min(args.sensor_cache_mb, args.video_cache_entries) < 0:
        parser.error("workers and cache limits must be nonnegative")
    if any(n not in [1, 3, 5, 7, 10, 20] for n in args.window_sizes):
        parser.error("window sizes must belong to the comparison plan")
    if args.cache_build_fps is not None and args.cache_build_fps <= 0:
        parser.error("cache-build-fps must be positive")
    if args.sensor_mmap != "off" and (args.mode != "optimized" or args.cache != "off"):
        parser.error("sensor mmap comparisons require --mode optimized --cache off to isolate the sensor backend")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "metrics.jsonl"
    register_config_schema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_directory = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base="1.3", config_dir=config_directory):
        base_cfg = compose(config_name="taf_adapter", overrides=args.override)
    init = time.perf_counter()
    source = TaFDataset(args.data_root, window_size=1, image_size=tuple(base_cfg.data.image_size) if base_cfg.data.image_size is not None else None,
                        stride=int(base_cfg.data.stride), sensor_types=base_cfg.data.sensor_types,
                        timestamp_cache_enabled=bool(base_cfg.data.timestamp_cache_enabled),
                        timestamp_cache_dir=base_cfg.data.timestamp_cache_dir,
                        show_initialization_progress=False, image_dtype="uint8",
                        sensor_cache_max_bytes=args.sensor_cache_mb * 1024 ** 2,
                        sensor_cache_shared_stats=True, pipeline_shared_stats=True, detailed_profiling=True)
    source_init = time.perf_counter() - init
    if args.sensor_mmap_build_only:
        cache = _new_sensor_cache(source, args.sensor_mmap_dir or base_cfg.data.sensor_mmap_dir, base_cfg, args)
        estimate = SensorMmapCache.estimate(_sensor_sources(source))
        print(json.dumps({"record_type": "sensor_mmap_estimate", **estimate}, indent=2), flush=True)
        if args.estimate_only:
            record = {"record_type": "sensor_mmap_estimate", "source_seconds": source_init,
                      "cache_dir": str(cache.directory), **estimate}
            output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            return
        preparation = cache.prepare(_sensor_sources(source))
        record = {"record_type": "sensor_mmap_full_build", "source_seconds": source_init,
                  "cache_dir": str(cache.directory), "estimate": estimate, **preparation}
        output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
        return
    estimate = FrozenFrameCache.estimate(source, args.feature_dim,
        amp_enabled=bool(base_cfg.training.amp) and device.type == "cuda")
    estimate["assumed_feature_dim"] = args.feature_dim
    estimate["feature_gib"] = estimate["feature_bytes"] / 1024 ** 3
    estimate["build_seconds_estimated"] = (estimate["unique_frames"] / args.cache_build_fps
                                           if args.cache_build_fps else None)
    estimate["note"] = "payload only; FPS must include decode, preprocessing, ViT and writes; no full cache built"
    def emit(row):
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, indent=2))
    emit({"record_type": "initialization", "source_seconds": source_init, "feature_cache_estimate": estimate,
          "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "device": str(device),
          "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
          "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
          "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
          "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    if args.estimate_only:
        return
    results = []
    modes = ["baseline", "optimized"] if args.mode == "both" else [args.mode]
    for n in args.window_sizes:
        set_seed(args.seed)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "config")):
            cfg = compose(config_name="taf_adapter", overrides=[
                f"data.window_size={n}", "training.batch_size=256", f"training.seed={args.seed}",
                "training.gradient_accumulation_steps=1", "training.min_same_episode_window_gap=20",
                "trainctl.performance.detailed_profiling=true", "trainctl.performance.monitor_cache=true",
                "trainctl.performance.monitor_gpu_utilization=true", *args.override])
        if cfg.training.batch_size != 256 or cfg.training.gradient_accumulation_steps != 1:
            raise ValueError("Benchmark requires batch=256 and accumulation=1")
        if not cfg.trainctl.performance.enabled or not cfg.trainctl.performance.detailed_profiling:
            raise ValueError("Benchmark requires enabled detailed profiling")
        OmegaConf.save(cfg, args.output_dir / f"config_N{n}.yaml")
        plan_started = time.perf_counter()
        dataset = ManifestDataset(source, args.manifest, n)
        indices = [i for i, row in enumerate(dataset.rows) if row["split"] == "train"]
        normalization_path = args.normalization or args.manifest.with_name(f"normalization_{dataset.manifest_id[:8]}.pt")
        normalization = torch.load(normalization_path, map_location="cpu", weights_only=True)
        train_keys = sorted({dataset.rows[i]["sequence_id"] for i in indices})
        if normalization.get("manifest_id") != dataset.manifest_id or normalization.get("train_groups") != train_keys:
            raise ValueError("Normalization identity differs from training manifest")
        sampler = FixedValidationBatchSampler(dataset, indices, batch_size=256,
                                              min_gap=int(cfg.training.min_same_episode_window_gap),
                                              seed=int(cfg.comparison.sampling_seed))
        plan = list(itertools.islice(iter(sampler), args.steps + args.warmup_steps))
        if len(plan) < args.steps + args.warmup_steps:
            raise ValueError("Not enough full batches; reduce steps")
        digest = hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()
        expected_order = hashlib.sha256()
        for batch in plan[args.warmup_steps:]:
            global_indices = [indices[i] for i in batch]
            expected_order.update(json.dumps([dataset.rows[i]["sample_id"] for i in global_indices],
                                             separators=(",", ":")).encode())
            frame_indices = []
            for i in global_indices:
                ep, frame_start = dataset._locate(i)
                frame_indices.append(ep.frames[frame_start:frame_start+n])
            expected_order.update(np.stack(frame_indices).tobytes())
        emit({"record_type": "plan", "N": n, "manifest_id": dataset.manifest_id,
              "plan_sha256": digest, "plan_initialization_seconds": time.perf_counter() - plan_started,
              "sampling_report": sampler.report.as_dict(), "correctness": _check_raw(dataset, source, indices, plan, args)})
        model_started = time.perf_counter()
        model = build_model(cfg, tactile_initialization_seed=args.seed + 10_000).to(device)
        if model.tactile_encoder.backbone_dim != args.feature_dim:
            raise ValueError("--feature-dim must match the configured backbone dimension")
        model.set_normalization(**normalization["normalization"])
        initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        emit({"record_type": "model_initialization", "N": n,
              "seconds": time.perf_counter() - model_started})
        if args.sensor_mmap != "off":
            results.extend(_sensor_benchmark(model, dataset, source, indices, plan, initial_state,
                cfg, args, device, expected_order.hexdigest(), digest, emit))
            del model
            continue
        cache = None
        if args.cache in ("on", "both"):
            selected = {}
            for batch in plan:
                for local in batch:
                    ep, start = dataset._locate(indices[local])
                    selected.setdefault(ep.video, set()).update(map(int, ep.video_frame_indices[start:start+n]))
            model.load_state_dict(initial_state)
            _configure(source, "optimized", args)
            cache = FrozenFrameCache(source, model, args.cache_dir or args.output_dir / "frame_cache",
                                     device=device, amp_enabled=bool(cfg.training.amp) and device.type == "cuda",
                                     selected_ordinals=selected, max_open_files=int(cfg.data.frame_cache_max_open_files))
            cached = CachedWindowDataset(dataset, cache)
            check = _equivalence(model, dataset, cached, [indices[i] for i in plan[0]], device,
                                 bool(cfg.training.amp) and device.type == "cuda")
            subset_frames = sum(map(len, selected.values()))
            cold_build_fps = (subset_frames / cache.build_seconds
                              if cache.build_misses and not cache.build_hits and cache.build_seconds else None)
            emit({"record_type": "subset_cache_build", "N": n, "seconds": cache.build_seconds,
                  "subset_frames": subset_frames, "cold_subset_build_fps": cold_build_fps,
                  "full_build_seconds_extrapolated": (estimate["unique_frames"] / cold_build_fps
                                                       if cold_build_fps else None),
                  "bytes": cache.disk_bytes, "hits": cache.build_hits, "misses": cache.build_misses,
                  "correctness": check, "note": "only benchmark frames; full estimate extrapolates cold subset, random seek/IO can bias it"})
        cache_cases = [False, True] if args.cache == "both" else [args.cache == "on"]
        for enabled in cache_cases:
            case_dataset = CachedWindowDataset(dataset, cache) if enabled else dataset
            for workers in args.workers:
                for mode in modes:
                    row = _run(model, case_dataset, source, indices, plan, initial_state,
                               cfg, args, device, mode, workers)
                    if row["sample_frame_order_sha256"] != expected_order.hexdigest():
                        raise AssertionError("Consumed sample/frame order differs from the batch plan")
                    row.update(record_type="benchmark", manifest_id=dataset.manifest_id, plan_sha256=digest)
                    emit(row)
                    results.append(row)
        orders = {r["sample_frame_order_sha256"] for r in results if r["N"] == n}
        if len(orders) != 1:
            raise AssertionError("Sample/frame order differs between benchmark runs")
        for workers in args.workers:
            for mode in modes:
                pair = {r["cache"]: r for r in results if r["N"] == n and r["workers"] == workers and r["mode"] == mode}
                if len(pair) == 2:
                    saved = pair[False]["seconds_per_step"] - pair[True]["seconds_per_step"]
                    emit({"record_type": "cache_break_even", "N": n, "workers": workers, "mode": mode,
                          "measured_steady_speedup": pair[False]["seconds_per_step"] / pair[True]["seconds_per_step"],
                          "subset_build_break_even_steps": cache.build_seconds / saved if saved > 0 else None,
                          "note": "subset build only; full build break-even uses estimated full build cost"})
            pair = {r["mode"]: r for r in results if r["N"] == n and r["workers"] == workers and not r["cache"]}
            if len(pair) == 2:
                emit({"record_type": "before_after", "N": n, "workers": workers,
                      "before_steps_s": pair["baseline"]["perf/steps_per_second"],
                      "after_steps_s": pair["optimized"]["perf/steps_per_second"],
                      "measured_speedup": pair["baseline"]["seconds_per_step"] / pair["optimized"]["seconds_per_step"],
                      "before_memory_bytes": pair["baseline"]["cpu_peak_sampled_rss_bytes"],
                      "after_memory_bytes": pair["optimized"]["cpu_peak_sampled_rss_bytes"],
                      "sample_frame_order_equal": True})
        if cache is not None:
            cache.close()
        source._close_video()
        del model


if __name__ == "__main__":
    main()
