"""Measure real N=5/N=20 training throughput with identical samples and model state.

Example: .venv/bin/python -m script.profile_throughput --data-root /path/to/TaF-Dataset
This script does not write checkpoints or modify the source data.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset
from script.train_adapter import validate
from utils.config import register_config_schema
from utils.data import FixedValidationBatchSampler, TemporalDiverseBatchSampler, split_by_group
from utils.frame_cache import CachedWindowDataset, FrozenFrameCache
from utils.logging import DeviceMetricAccumulator
from utils.model_factory import build_model


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _memory(iterator):
    workers = getattr(iterator, "_workers", [])
    return sum(_rss(pid) for pid in [os.getpid(), *(w.pid for w in workers)])


def _loaders(dataset, train_indices, val_indices, train_batches, val_batches, args, device):
    kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return (DataLoader(Subset(dataset, train_indices), batch_sampler=train_batches, **kwargs),
            DataLoader(Subset(dataset, val_indices), batch_sampler=val_batches, **kwargs))


def _timed_run(model, dataset, original, train_indices, val_indices, initial_state,
               train_batches, val_batches, args, device, amp):
    model.load_state_dict(initial_state)
    model.train()
    torch.manual_seed(args.seed)
    train_loader, val_loader = _loaders(dataset, train_indices, val_indices,
                                        train_batches, val_batches, args, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    iterator = iter(train_loader)
    metrics = DeviceMetricAccumulator()
    loading, steps = 0.0, 0.0
    frames = 0
    peak_cpu = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.steps):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        loading += time.perf_counter() - start
        peak_cpu = max(peak_cpu, _memory(iterator))
        _sync(device)
        start = time.perf_counter()
        force = batch["force"].to(device, non_blocking=True)
        pressure = batch["pressure"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if "backbone_features" in batch:
                features = batch["backbone_features"].to(device, non_blocking=True)
                output = model.forward_from_backbone_features(features, force, pressure)
                frames += features.shape[0] * features.shape[1]
            else:
                tactile = batch["tactile"].to(device, non_blocking=True)
                output = model(tactile, force, pressure)
                frames += tactile.shape[0] * tactile.shape[1]
        output["loss_total"].backward()
        optimizer.step()
        metrics.add(output)
        _sync(device)
        steps += time.perf_counter() - start
    validation_start = time.perf_counter()
    validate(model, val_loader, device, amp, args.validation_batches)
    _sync(device)
    validation_time = time.perf_counter() - validation_start
    peak_cpu = max(peak_cpu, _memory(iterator))
    metrics.pop_means("train")
    return {"data_loading_seconds": loading, "training_step_seconds": steps / args.steps,
            "validation_seconds": validation_time, "steps_per_second": args.steps / (loading + steps),
            "frames_per_second": frames / (loading + steps), "frames": frames,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cpu_peak_rss_bytes": peak_cpu}


def _separate_decode_time(source, count):
    start = time.perf_counter()
    for index in range(min(count, len(source))):
        ep, offset = source._locate(index)
        import numpy as np
        source._images(ep, np.arange(offset, offset + source.window_size, dtype=np.int64))
    return time.perf_counter() - start


def _separate_backbone_time(model, source, device, amp, count):
    model.eval()
    frames = []
    for index in range(min(count, len(source))):
        frames.append(source[index]["tactile"])
    batch = torch.stack(frames).to(device)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        model.tactile_encoder.encode_backbone(batch)
    _sync(device)
    return time.perf_counter() - start


def _equivalence(model, source, cached, index, device, amp):
    raw = source[index]
    stored = cached[index]
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        raw_features = model.tactile_encoder.encode_backbone(raw["tactile"].unsqueeze(0).to(device))
        cached_features = stored["backbone_features"].unsqueeze(0).to(device)
        raw_projected = model.tactile_encoder.project_backbone_features(raw_features)
        cached_projected = model.tactile_encoder.project_backbone_features(cached_features)
    difference = (raw_projected.float() - cached_projected.float()).abs().max().item()
    tolerance = 1e-3 if amp else 1e-5
    if not torch.allclose(raw_projected, cached_projected, atol=tolerance, rtol=tolerance):
        raise ValueError(f"Cached ViT output differs from raw path: max error={difference}")
    return difference


def _selected_ordinals(source, train_indices, val_indices, train_batches, val_batches):
    import numpy as np
    selected = {}
    for indices, batches in ((train_indices, train_batches), (val_indices, val_batches)):
        for batch in batches:
            for local in batch:
                ep, start = source._locate(indices[local])
                offsets = np.arange(start, start + source.window_size)
                selected.setdefault(ep.video, set()).update(map(int, ep.video_frame_indices[offsets]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/frame_cache"))
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sensor-cache-mb", type=int, default=256)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    register_config_schema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for n in args.window_sizes:
        with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "config")):
            cfg = compose(config_name="taf_adapter", overrides=[
                f"data.window_size={n}", f"training.batch_size={args.batch_size}",
                f"tactile_encoder.pretrained={str(not args.no_pretrained).lower()}"])
        source = TaFDataset(args.data_root, window_size=n, image_size=tuple(cfg.data.image_size),
                            max_sequences=args.max_sequences, show_initialization_progress=False,
                            timestamp_cache_dir=args.cache_dir / "timestamps",
                            sensor_cache_max_bytes=args.sensor_cache_mb * 1024 * 1024,
                            image_dtype="uint8", sensor_cache_shared_stats=True)
        train_indices, val_indices, *_ = split_by_group(
            source, unit=str(cfg.data.split_unit), validation_fraction=float(cfg.data.validation_fraction),
            seed=args.seed)
        train_plan = list(TemporalDiverseBatchSampler(
            source, train_indices, batch_size=args.batch_size,
            min_gap=n, seed=args.seed))[:args.steps]
        val_plan = list(FixedValidationBatchSampler(
            source, val_indices, batch_size=args.batch_size,
            min_gap=n, seed=args.seed))[:args.validation_batches]
        if not train_plan or not val_plan:
            raise ValueError("Selected sequences have no full train/validation batch")
        selected = _selected_ordinals(source, train_indices, val_indices, train_plan, val_plan)
        torch.manual_seed(args.seed)
        model = build_model(cfg).to(device)
        amp = bool(cfg.training.amp) and device.type == "cuda"
        initial_state = copy.deepcopy(model.state_dict())
        decode_time = _separate_decode_time(source, args.validation_batches)
        backbone_time = _separate_backbone_time(model, source, device, amp, 1)
        before_raw = source.sensor_cache_total_stats()
        raw = _timed_run(model, source, source, train_indices, val_indices,
                         initial_state, train_plan, val_plan, args, device, amp)
        after_raw = source.sensor_cache_total_stats()
        raw["sensor_parquet_reads"] = after_raw["parquet_reads"] - before_raw["parquet_reads"]
        raw["sensor_cache_hits"] = after_raw["cache_hits"] - before_raw["cache_hits"]
        start = time.perf_counter()
        cache = FrozenFrameCache(source, model, args.cache_dir, device=device,
                                 amp_enabled=amp, batch_size=cfg.data.frame_cache_batch_size,
                                 selected_ordinals=selected)
        build_time = time.perf_counter() - start
        cached_dataset = CachedWindowDataset(source, cache)
        error = _equivalence(model, source, cached_dataset,
                             train_indices[train_plan[0][0]], device, amp)
        before_cached = source.sensor_cache_total_stats()
        cached = _timed_run(model, cached_dataset, source, train_indices, val_indices,
                            initial_state, train_plan, val_plan, args, device, amp)
        after_cached = source.sensor_cache_total_stats()
        cached["sensor_parquet_reads"] = after_cached["parquet_reads"] - before_cached["parquet_reads"]
        cached["sensor_cache_hits"] = after_cached["cache_hits"] - before_cached["cache_hits"]
        rows.append({"N": n, "device": str(device), "batch_size": args.batch_size,
                     "raw": {**raw, "video_decode_seconds_sample": decode_time,
                             "frozen_vit_forward_seconds_sample": backbone_time},
                     "cached": {**cached, "video_decode_seconds_per_step": 0,
                                "frozen_vit_forward_seconds_per_step": 0},
                     "cache_build_seconds": build_time, "max_projected_feature_error": error,
                     "measured_step_speedup": raw["training_step_seconds"] / cached["training_step_seconds"],
                     "measured_throughput_speedup": cached["steps_per_second"] / raw["steps_per_second"],
                     "sensor_cache_stats_all_workers": source.sensor_cache_total_stats()})
        print(json.dumps(rows[-1], indent=2))
    print(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()
