"""Optional training telemetry; no CUDA synchronization on individual steps."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

_GIB = 1024 ** 3


def _rss_gib() -> float | None:
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / _GIB
    except OSError:
        return None
    return None


def _peak_rss_gib() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (_GIB if sys.platform == "darwin" else 1024 * 1024)
    except (ImportError, OSError):
        return None


def process_resources(pids, *, include_pss=False):
    """Linux RSS/HWM and CPU seconds; unavailable processes are explicitly omitted."""
    result = {}
    for pid in pids:
        try:
            status = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value, _ = line.split()
                    status[key[:-1]] = int(value) * 1024
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            status["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            if include_pss:
                try:
                    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                        if line.startswith("Pss:"):
                            status["Pss"] = int(line.split()[1]) * 1024
                            break
                except OSError:
                    pass  # PSS permissions vary; omit it instead of reporting a partial sum.
            result[pid] = status
        except (OSError, ValueError, IndexError):
            continue
    return result


def child_pids():
    try:
        return [int(pid) for pid in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()]
    except OSError:
        return []


class _StageTimer:
    def __init__(self, owner: "PerformanceMonitor", name: str):
        self.owner, self.name = owner, name
        self.start = None

    def __enter__(self):
        if self.owner.cuda_events:
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.owner.cuda_events:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.owner.events[self.name].append((self.start, end))
        else:
            self.owner.stage_ms[self.name] += (time.perf_counter() - self.start) * 1000
            self.owner.stage_counts[self.name] += 1


class PerformanceMonitor:
    """Interval telemetry and cumulative run diagnostics.

    Data wait and throughput are CPU wall time. Detailed CUDA stages use event
    pairs and synchronize once at a reporting boundary, never per optimizer step.
    """

    def __init__(self, config, device: torch.device, *, source=None, frame_cache=None,
                 started: float | None = None):
        self.config = config
        self.enabled = bool(config.enabled)
        self.detailed = self.enabled and bool(config.detailed_profiling)
        self.device = device
        self.cuda_events = self.detailed and device.type == "cuda"
        self.source = source
        self.frame_cache = frame_cache
        self.started = time.monotonic() if started is None else started
        self.last_progress = time.monotonic()
        self.training_phase = False
        self.interval = defaultdict(float)
        self.total = defaultdict(float)
        self.stage_ms = defaultdict(float)
        self.stage_counts = defaultdict(int)
        self.events = defaultdict(list)
        self.event_last = {}
        self.event_totals = defaultdict(float)
        self.last_metrics = {}
        self.peak_cpu_gib = 0.0
        self.peak_total_cpu_gib = 0.0
        self.peak_gpu_allocated_gib = 0.0
        self.peak_gpu_reserved_gib = 0.0
        self.warning_times = {}
        self._gpu_utilization_unavailable = False
        self._warning_lock = threading.Lock()
        self._stop = threading.Event()
        self._watchdog = None
        self._resource_time = time.monotonic()
        self._resources = process_resources(child_pids()) if self.detailed else {}
        self._last_pipeline = source.pipeline_total_stats() if source and self.enabled else None
        self._last_sensor = source.sensor_cache_total_stats() if source and self.enabled and config.monitor_cache else None
        mmap_cache = getattr(source, "_sensor_mmap", None)
        self._last_mmap = mmap_cache.stats() if mmap_cache is not None and self.enabled else None
        self._initial_mmap = dict(self._last_mmap or {})
        if self.enabled and device.type == "cuda" and config.monitor_gpu_memory:
            torch.cuda.reset_peak_memory_stats(device)

    def stage(self, name: str):
        return _StageTimer(self, name) if self.detailed else contextlib.nullcontext()

    def attach_backbone(self, backbone, name="vit_forward") -> None:
        self._hooks = getattr(self, "_hooks", [])
        if not self.detailed:
            return
        starts = []
        def before(_module, _args):
            if self.cuda_events:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                starts.append(event)
            else:
                starts.append(time.perf_counter())
        def after(_module, _args, _result):
            start = starts.pop()
            if self.cuda_events:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.events[name].append((start, event))
            else:
                self.stage_ms[name] += (time.perf_counter() - start) * 1000
                self.stage_counts[name] += 1
        self._hooks.extend([backbone.register_forward_pre_hook(before),
                            backbone.register_forward_hook(after)])

    def close(self) -> None:
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1)
        for hook in getattr(self, "_hooks", []):
            hook.remove()

    def record_wait(self, seconds: float) -> None:
        if self.enabled and self.config.monitor_data_loading:
            self.interval["wait_seconds"] += seconds
            self.interval["wait_count"] += 1
            self.total["wait_seconds"] += seconds

    def record_batch(self, batch):
        if not self.detailed or not hasattr(batch, "collated_at"):
            return
        # Includes IPC, serialization, queue residence, scheduling and unpickling.
        # It is deliberately NOT labelled pure IPC latency.
        self.interval["delivery_queue_seconds"] += max(
            0.0, time.monotonic() - batch.collated_at - batch.pin_seconds)
        self.interval["pin_memory_seconds"] += batch.pin_seconds
        self.interval["delivered_batches"] += 1

    def record_iteration(self, seconds: float, samples: int, frames: int) -> None:
        if not self.enabled:
            return
        self.interval["active_seconds"] += seconds
        self.interval["samples"] += samples
        self.interval["frames"] += frames
        self.total["active_seconds"] += seconds
        self.total["samples"] += samples
        self.total["frames"] += frames

    def completed_step(self) -> None:
        if self.enabled:
            self.interval["steps"] += 1
            self.total["steps"] += 1
            self.last_progress = time.monotonic()

    def record_event(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.event_last[name] = seconds
        self.event_totals[name] += seconds
        self.last_metrics[f"perf/{name}_seconds"] = seconds

    def phase(self, training: bool) -> None:
        self.training_phase = training
        if training:
            self.last_progress = time.monotonic()

    def _drain_events(self) -> None:
        if not self.cuda_events or not self.events:
            return
        torch.cuda.synchronize(self.device)
        for name, pairs in self.events.items():
            self.stage_ms[name] += sum(start.elapsed_time(end) for start, end in pairs)
            self.stage_counts[name] += len(pairs)
        self.events.clear()

    def snapshot(self, step: int, max_steps: int) -> dict[str, float]:
        if not self.enabled:
            return {}
        metrics = {}
        active = self.interval["active_seconds"]
        if self.config.monitor_throughput and active > 0:
            metrics.update({
                "perf/steps_per_second": self.interval["steps"] / active,
                "perf/samples_per_second": self.interval["samples"] / active,
                "perf/frames_per_second": self.interval["frames"] / active,
            })
            elapsed = max(0.0, time.monotonic() - self.started)
            metrics["perf/elapsed_seconds"] = elapsed
            if self.total["steps"]:
                rate = self.total["steps"] / elapsed if elapsed else 0
                if rate:
                    metrics["perf/eta_seconds"] = (max_steps - step) / rate
        if self.config.monitor_data_loading and self.interval["wait_count"]:
            metrics["perf/data_wait_ms"] = 1000 * self.interval["wait_seconds"] / self.interval["wait_count"]
            if active:
                metrics["perf/data_wait_ratio"] = self.interval["wait_seconds"] / active
        if self.interval["delivered_batches"]:
            for name in ("delivery_queue", "pin_memory"):
                metrics[f"perf/{name}_ms_per_batch"] = (
                    self.interval[name + "_seconds"] * 1000 / self.interval["delivered_batches"])
        if self.detailed:
            self._drain_events()
            suffix = "_ms" if self.cuda_events else "_cpu_ms"
            for name, total_ms in self.stage_ms.items():
                count = self.stage_counts[name]
                if count:
                    metrics[f"perf/{name}{suffix}"] = total_ms / count
                    metrics[f"perf/{name}_calls"] = count
                    metrics[f"perf/{name}_total{suffix}"] = total_ms
        if self.source and self.config.monitor_cache:
            sensor = self.source.sensor_cache_total_stats()
            previous = self._last_sensor or {"cache_hits": 0, "cache_misses": 0,
                                             "parquet_reads": 0}
            hits = sensor["cache_hits"] - previous["cache_hits"]
            misses = sensor["cache_misses"] - previous["cache_misses"]
            if hits + misses:
                metrics["perf/parquet_cache_hit_rate"] = hits / (hits + misses)
                metrics["perf/parquet_cache_miss_rate"] = misses / (hits + misses)
            for key in ("parquet_reads", "cache_hits", "cache_misses", "evictions", "oversize", "loaded_bytes"):
                if key in sensor:
                    metrics[f"perf/sensor_{key}"] = sensor[key] - previous.get(key, 0)
            self._last_sensor = sensor
        if self.source and self.config.monitor_data_loading:
            pipeline = self.source.pipeline_total_stats()
            previous = self._last_pipeline or {key: 0 for key in pipeline}
            parquet_delta = pipeline["parquet_read_seconds"] - previous["parquet_read_seconds"]
            reads = pipeline["parquet_read_calls"] - previous["parquet_read_calls"]
            decode_delta = pipeline["video_decode_seconds"] - previous["video_decode_seconds"]
            calls = pipeline["video_decode_calls"] - previous["video_decode_calls"]
            if reads > 0:
                metrics["perf/parquet_read_ms"] = parquet_delta * 1000 / reads
            if calls > 0:
                metrics["perf/video_decode_ms"] = decode_delta * 1000 / calls
            if self.detailed:
                for key, value in pipeline.items():
                    delta = value - previous.get(key, 0)
                    metrics[f"perf/worker_{key}"] = value if "_resident_" in key else delta
                    if key.endswith("_seconds"):
                        name = key[:-8]
                        count = pipeline.get(name + "_calls", 0) - previous.get(name + "_calls", 0)
                        if count:
                            metrics[f"perf/{name}_worker_ms_per_call"] = delta * 1000 / count
                        if self.interval["wait_count"]:
                            metrics[f"perf/{name}_worker_ms_per_batch"] = delta * 1000 / self.interval["wait_count"]
                sensor_hits = pipeline.get("video_cache_hits", 0) - previous.get("video_cache_hits", 0)
                sensor_misses = pipeline.get("video_cache_misses", 0) - previous.get("video_cache_misses", 0)
                if sensor_hits + sensor_misses:
                    metrics["perf/video_cache_hit_rate"] = sensor_hits / (sensor_hits + sensor_misses)
            self._last_pipeline = pipeline
        mmap_cache = getattr(self.source, "_sensor_mmap", None)
        if self.source and (self.config.monitor_cache or self.config.monitor_data_loading):
            metrics["perf/sensor_mmap_enabled"] = float(mmap_cache is not None)
        if mmap_cache is not None and (self.config.monitor_cache or self.config.monitor_data_loading):
            current = mmap_cache.stats()
            previous = self._last_mmap or dict.fromkeys(current, 0)
            delta = {key: value - previous.get(key, 0) for key, value in current.items()}
            metrics.update({f"perf/sensor_mmap_{key}": value for key, value in delta.items()})
            metrics["perf/sensor_mmap_initial_build_seconds"] = self._initial_mmap.get("build_seconds", 0)
            metrics["perf/sensor_mmap_initial_build_files"] = self._initial_mmap.get("builds", 0)
            requests = delta["cache_hits"] + delta["cache_misses"]
            if requests:
                metrics["perf/sensor_mmap_hit_rate"] = delta["cache_hits"] / requests
            for name in ("load", "access", "parquet_read", "conversion"):
                if delta[name + "_calls"]:
                    metrics[f"perf/sensor_mmap_{name}_ms_per_call"] = (
                        delta[name + "_seconds"] * 1000 / delta[name + "_calls"])
            self._last_mmap = current
        if self.frame_cache is not None:
            if self.frame_cache.hit_rate is not None:
                metrics["perf/feature_cache_hit_rate"] = self.frame_cache.hit_rate
            metrics["perf/feature_cache_build_seconds"] = self.frame_cache.build_seconds
            metrics["perf/feature_cache_disk_bytes"] = self.frame_cache.disk_bytes
        if (self.source and self.config.monitor_cache
                and getattr(getattr(self.source, "_timestamp_store", None), "enabled", False)):
            initialization = getattr(self.source, "initialization_stats", {})
            processed = initialization.get("processed_videos", 0)
            if processed:
                metrics["perf/video_timestamp_cache_hit_rate"] = (
                    initialization.get("cache_hits", 0) / processed
                )
        if self.detailed:
            now = time.monotonic()
            elapsed = max(now - self._resource_time, 1e-9)
            resources = process_resources(child_pids())
            worker_cpu = 0.0
            for pid, values in resources.items():
                cpu = max(0.0, values["cpu_seconds"] - self._resources.get(pid, {}).get("cpu_seconds", 0))
                worker_cpu += cpu
                metrics[f"perf/worker_{pid}_cpu_pct"] = cpu / elapsed * 100
                metrics[f"perf/worker_{pid}_rss_gib"] = values.get("VmRSS", 0) / _GIB
                metrics[f"perf/worker_{pid}_peak_rss_gib"] = values.get("VmHWM", 0) / _GIB
            metrics["perf/workers_cpu_pct"] = worker_cpu / elapsed * 100
            metrics["perf/workers_rss_gib"] = sum(v.get("VmRSS", 0) for v in resources.values()) / _GIB
            metrics["perf/total_rss_gib"] = metrics["perf/workers_rss_gib"] + (_rss_gib() or 0)
            self._resources, self._resource_time = resources, now
        if self.config.monitor_cpu_memory:
            if not self.detailed:
                resources = process_resources(child_pids())
            worker_rss = sum(v.get("VmRSS", 0) for v in resources.values()) / _GIB
            metrics["perf/workers_rss_gib"] = worker_rss
            rss = _rss_gib()
            if rss is not None:
                total_rss = worker_rss + rss
                self.peak_total_cpu_gib = max(self.peak_total_cpu_gib, total_rss)
                metrics["perf/total_rss_gib"] = total_rss
                metrics["perf/total_peak_sampled_rss_gib"] = self.peak_total_cpu_gib
                metrics["perf/cpu_rss_gib"] = rss
                self.peak_cpu_gib = max(self.peak_cpu_gib, rss)
            peak = _peak_rss_gib()
            if peak is not None:
                self.peak_cpu_gib = max(self.peak_cpu_gib, peak)
                metrics["perf/cpu_peak_rss_gib"] = self.peak_cpu_gib
        if self.detailed and self.config.monitor_cpu_memory:
            pids = [os.getpid(), *child_pids()]
            memory = process_resources(pids, include_pss=True)
            available = len(memory) == len(pids) and all("Pss" in value for value in memory.values())
            metrics["perf/pss_available"] = float(available)
            if available:
                metrics["perf/total_pss_gib"] = sum(value["Pss"] for value in memory.values()) / _GIB
        if self.device.type == "cuda" and self.config.monitor_gpu_memory:
            metrics["perf/gpu_allocated_gib"] = torch.cuda.memory_allocated(self.device) / _GIB
            metrics["perf/gpu_reserved_gib"] = torch.cuda.memory_reserved(self.device) / _GIB
            allocated = torch.cuda.max_memory_allocated(self.device) / _GIB
            reserved = torch.cuda.max_memory_reserved(self.device) / _GIB
            self.peak_gpu_allocated_gib = max(self.peak_gpu_allocated_gib, allocated)
            self.peak_gpu_reserved_gib = max(self.peak_gpu_reserved_gib, reserved)
            metrics["perf/gpu_peak_allocated_gib"] = self.peak_gpu_allocated_gib
            metrics["perf/gpu_peak_reserved_gib"] = self.peak_gpu_reserved_gib
        if self.device.type == "cuda" and self.config.monitor_gpu_utilization:
            if not self._gpu_utilization_unavailable:
                try:
                    metrics["perf/gpu_utilization_pct"] = float(torch.cuda.utilization(self.device))
                except (ImportError, AttributeError, OSError, RuntimeError) as exc:
                    self._gpu_utilization_unavailable = True
                    self.last_metrics.pop("perf/gpu_utilization_pct", None)
                    warnings.warn(
                        f"GPU utilization telemetry unavailable; disabling this optional metric: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            metrics["perf/gpu_utilization_available"] = float(not self._gpu_utilization_unavailable)
        for name, seconds in self.event_last.items():
            metrics[f"perf/{name}_seconds"] = seconds
        self.interval.clear()
        self.stage_ms.clear()
        self.stage_counts.clear()
        self.last_metrics.update(metrics)
        return metrics

    def summary(self, *, window_size: int, seed: int, train_windows: int,
                hardware: dict) -> dict:
        active = self.total["active_seconds"]
        result = {"window_size": window_size, "seed": seed,
                  "train_windows": train_windows, "hardware": hardware,
                  "training_duration_seconds": max(0.0, time.monotonic() - self.started),
                  "training_active_seconds": active,
                  "average_steps_per_second": self.total["steps"] / active if active else None,
                  "average_frames_per_second": self.total["frames"] / active if active else None,
                  "peak_cpu_rss_gib": self.peak_cpu_gib or None,
                  "peak_gpu_allocated_gib": self.peak_gpu_allocated_gib or None,
                  "peak_gpu_reserved_gib": self.peak_gpu_reserved_gib or None,
                  "evaluation_total_seconds": dict(self.event_totals),
                  "last_metrics": dict(self.last_metrics)}
        if self.source and self.config.monitor_cache:
            result["parquet_cache"] = self.source.sensor_cache_total_stats()
        mmap_cache = getattr(self.source, "_sensor_mmap", None)
        if mmap_cache is not None:
            result["sensor_mmap"] = {"directory": str(mmap_cache.directory),
                "max_open_files": mmap_cache.max_open_files, "batch_rows": mmap_cache.batch_rows,
                "disk_bytes": mmap_cache.disk_bytes(),
                "initialization_stats": self._initial_mmap, "total_stats": mmap_cache.stats()}
        if self.frame_cache is not None:
            result["feature_cache"] = {"hits": self.frame_cache.build_hits,
                                       "misses": self.frame_cache.build_misses,
                                       "hit_rate": self.frame_cache.hit_rate,
                                       "build_seconds": self.frame_cache.build_seconds,
                                       "disk_bytes": self.frame_cache.disk_bytes}
        return result

    def failure_metrics(self) -> dict:
        """Read the latest memory counters without synchronizing a failing GPU."""
        result = dict(self.last_metrics)
        if self.enabled and self.config.monitor_cpu_memory:
            rss = _rss_gib()
            if rss is not None:
                self.peak_cpu_gib = max(self.peak_cpu_gib, rss)
                result["perf/cpu_rss_gib"] = self.peak_cpu_gib
            peak = _peak_rss_gib()
            if peak is not None:
                self.peak_cpu_gib = max(self.peak_cpu_gib, peak)
                result["perf/cpu_peak_rss_gib"] = self.peak_cpu_gib
        if self.enabled and self.device.type == "cuda" and self.config.monitor_gpu_memory:
            try:
                result["perf/gpu_peak_allocated_gib"] = max(
                    self.peak_gpu_allocated_gib,
                    torch.cuda.max_memory_allocated(self.device) / _GIB,
                )
                result["perf/gpu_peak_reserved_gib"] = max(
                    self.peak_gpu_reserved_gib,
                    torch.cuda.max_memory_reserved(self.device) / _GIB,
                )
            except RuntimeError:
                pass
        return result

    def warn_if_needed(self, metrics: dict, step: int, emit: Callable[[str], None]) -> None:
        warning = self.config.warnings
        if not self.enabled or not warning.enabled:
            return
        if (step >= self.config.log_every
                and metrics.get("perf/data_wait_ratio", 0) > warning.data_wait_ratio_threshold):
            self._warn("data_wait", "DataLoader waiting exceeds the configured training-time ratio", emit)
        for key, threshold, label in (
            ("perf/cpu_rss_gib", warning.cpu_rss_warning_gib, "CPU RSS"),
            ("perf/gpu_peak_reserved_gib", warning.gpu_reserved_warning_gib, "GPU reserved memory"),
        ):
            if threshold is not None and metrics.get(key, 0) >= threshold:
                self._warn(label, f"{label} reached {metrics[key]:.2f} GiB", emit)
        if (self.total["steps"] >= 100 and "perf/parquet_cache_hit_rate" in metrics
                and metrics["perf/parquet_cache_hit_rate"] < warning.cache_hit_rate_threshold):
            self._warn("cache", "Parquet cache hit rate is below the configured threshold", emit)

    def _warn(self, key: str, message: str, emit: Callable[[str], None]) -> None:
        with self._warning_lock:
            now = time.monotonic()
            if now - self.warning_times.get(key, -float("inf")) < self.config.warnings.warning_cooldown_seconds:
                return
            self.warning_times[key] = now
        emit(message)

    def start_watchdog(self, emit: Callable[[str], None]) -> None:
        warning = self.config.warnings
        if not self.enabled or not warning.enabled:
            return
        def watch():
            period = min(30.0, max(1.0, warning.stall_timeout_seconds / 4))
            while not self._stop.wait(period):
                if self.training_phase and time.monotonic() - self.last_progress >= warning.stall_timeout_seconds:
                    self._warn("stall", "No optimizer-step progress during training", emit)
        self._watchdog = threading.Thread(target=watch, daemon=True)
        self._watchdog.start()
