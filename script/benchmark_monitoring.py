"""Measure passive performance-monitor overhead in a deterministic CPU training loop."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from utils.performance import PerformanceMonitor


def _config(enabled):
    return SimpleNamespace(
        enabled=enabled, detailed_profiling=False, log_every=20,
        monitor_cpu_memory=True, monitor_gpu_memory=False,
        monitor_throughput=True, monitor_data_loading=True,
        monitor_cache=False, monitor_gpu_utilization=False,
        warnings=SimpleNamespace(enabled=False),
    )


def _run(steps: int, enabled: bool) -> tuple[float, float]:
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(),
                                torch.nn.Linear(256, 64))
    optimizer = torch.optim.AdamW(model.parameters())
    x = torch.randn(32, 256)
    target = torch.randn(32, 64)
    monitor = PerformanceMonitor(_config(enabled), torch.device("cpu"))
    started = time.perf_counter()
    cpu_started = time.process_time()
    for step in range(1, steps + 1):
        iteration = time.perf_counter()
        monitor.record_wait(0.0)
        with monitor.stage("h2d_transfer"):
            batch = x
        with monitor.stage("model_forward"):
            loss = (model(batch) - target).square().mean()
        with monitor.stage("backward"):
            loss.backward()
        with monitor.stage("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        monitor.record_iteration(time.perf_counter() - iteration, 32, 160)
        monitor.completed_step()
        if enabled and step % 20 == 0:
            monitor.snapshot(step, steps)
    duration = time.perf_counter() - started
    cpu_duration = time.process_time() - cpu_started
    monitor.close()
    return duration, cpu_duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _run(10, False)
    results = {"disabled": {"wall": [], "cpu": []},
               "enabled": {"wall": [], "cpu": []}}
    for index in range(args.repeats):
        for enabled in ((False, True) if index % 2 == 0 else (True, False)):
            wall, cpu = _run(args.steps, enabled)
            group = results["enabled" if enabled else "disabled"]
            group["wall"].append(wall)
            group["cpu"].append(cpu)
    baseline = statistics.median(results["disabled"]["wall"])
    monitored = statistics.median(results["enabled"]["wall"])
    baseline_cpu = statistics.median(results["disabled"]["cpu"])
    monitored_cpu = statistics.median(results["enabled"]["cpu"])
    paired_wall = [100 * (on / off - 1) for on, off in zip(
        results["enabled"]["wall"], results["disabled"]["wall"], strict=True)]
    paired_cpu = [100 * (on / off - 1) for on, off in zip(
        results["enabled"]["cpu"], results["disabled"]["cpu"], strict=True)]
    print(json.dumps({"device": "cpu", "workload": "synthetic 2-layer MLP, batch 32",
                      "steps_per_repeat": args.steps, "repeats": args.repeats,
                      "seconds": results, "median_seconds_disabled": baseline,
                      "median_seconds_enabled": monitored,
                      "measured_overhead_percent": 100 * (monitored / baseline - 1),
                      "measured_overhead_ms_per_step": 1000 * (monitored - baseline) / args.steps,
                      "median_cpu_seconds_disabled": baseline_cpu,
                      "median_cpu_seconds_enabled": monitored_cpu,
                      "measured_cpu_overhead_percent": 100 * (monitored_cpu / baseline_cpu - 1),
                      "measured_cpu_overhead_ms_per_step":
                          1000 * (monitored_cpu - baseline_cpu) / args.steps,
                      "paired_wall_overhead_percent": paired_wall,
                      "paired_cpu_overhead_percent": paired_cpu,
                      "median_paired_wall_overhead_percent": statistics.median(paired_wall),
                      "median_paired_cpu_overhead_percent": statistics.median(paired_cpu)},
                     indent=2))


if __name__ == "__main__":
    main()
