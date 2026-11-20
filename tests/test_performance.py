"""Performance telemetry and existing trainctl protocol tests, without Discord."""

from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.performance import PerformanceMonitor
from utils.trainctl import (TrainctlReporter, build_checkpoint_message,
                            build_failure_message, build_finish_message)
from utils.logging import ExperimentLogger
from script.compare_performance import collect, write_comparison


class _Source:
    def __init__(self):
        self.sensor = {"parquet_reads": 0, "cache_hits": 0, "cache_misses": 0,
                       "hit_rate": 0.0}
        self.pipeline = {"parquet_read_seconds": 0.0, "parquet_read_calls": 0.0,
                         "video_decode_seconds": 0.0, "video_decode_calls": 0.0}

    def sensor_cache_total_stats(self):
        return dict(self.sensor)

    def pipeline_total_stats(self):
        return dict(self.pipeline)


class _FakeTracker:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return call


def _config(*, detailed=False, warnings=False):
    return SimpleNamespace(
        enabled=True, detailed_profiling=detailed, log_every=2,
        monitor_cpu_memory=False, monitor_gpu_memory=False,
        monitor_throughput=True, monitor_data_loading=True,
        monitor_cache=True, monitor_gpu_utilization=False,
        warnings=SimpleNamespace(
            enabled=warnings, data_wait_ratio_threshold=0.5,
            stall_timeout_seconds=600, warning_cooldown_seconds=1800,
            cpu_rss_warning_gib=None, gpu_reserved_warning_gib=None,
            cache_hit_rate_threshold=0.1, send_to_trainctl=False,
        ),
    )


class PerformanceTests(unittest.TestCase):
    def test_interval_metrics_and_no_cuda_synchronization(self):
        source = _Source()
        monitor = PerformanceMonitor(_config(), torch.device("cpu"), source=source)
        with patch("torch.cuda.synchronize", side_effect=AssertionError("unexpected CUDA sync")):
            monitor.record_wait(0.2)
            monitor.record_iteration(1.0, 4, 20)
            monitor.completed_step()
            source.sensor.update(parquet_reads=1, cache_hits=3, cache_misses=1)
            source.pipeline.update(parquet_read_seconds=0.04, parquet_read_calls=1,
                                   video_decode_seconds=0.08, video_decode_calls=2)
            result = monitor.snapshot(1, 10)
        self.assertEqual(result["perf/steps_per_second"], 1)
        self.assertEqual(result["perf/samples_per_second"], 4)
        self.assertEqual(result["perf/frames_per_second"], 20)
        self.assertEqual(result["perf/data_wait_ms"], 200)
        self.assertEqual(result["perf/data_wait_ratio"], 0.2)
        self.assertEqual(result["perf/parquet_read_ms"], 40)
        self.assertEqual(result["perf/video_decode_ms"], 40)
        self.assertEqual(result["perf/parquet_cache_hit_rate"], .75)
        self.assertEqual(result["perf/parquet_cache_miss_rate"], .25)
        self.assertNotIn("perf/model_forward_ms", result)
        self.assertNotIn("perf/validation_seconds", result)
        monitor.record_event("validation", 2.5)
        self.assertEqual(monitor.last_metrics["perf/validation_seconds"], 2.5)
        self.assertEqual(monitor.summary(window_size=5, seed=42, train_windows=100,
                                         hardware={})["evaluation_total_seconds"]["validation"], 2.5)
        monitor.close()

    def test_detailed_cpu_stage_and_rate_limited_warning(self):
        config = _config(detailed=True, warnings=True)
        monitor = PerformanceMonitor(config, torch.device("cpu"))
        with monitor.stage("model_forward"):
            _ = sum(range(100))
        metrics = monitor.snapshot(2, 10)
        self.assertIn("perf/model_forward_cpu_ms", metrics)
        self.assertNotIn("perf/model_forward_ms", metrics)
        emitted = []
        monitor.warn_if_needed({"perf/data_wait_ratio": .9}, 2, emitted.append)
        monitor.warn_if_needed({"perf/data_wait_ratio": .9}, 2, emitted.append)
        self.assertEqual(len(emitted), 1)
        monitor.close()

    def test_cuda_events_synchronize_only_at_reporting_boundary(self):
        class FakeEvent:
            def __init__(self, **_kwargs):
                pass
            def record(self):
                pass
            def elapsed_time(self, _other):
                return 2.0
        config = _config(detailed=True)
        with patch("torch.cuda.Event", FakeEvent), patch("torch.cuda.synchronize") as sync:
            monitor = PerformanceMonitor(config, torch.device("cuda"))
            with monitor.stage("model_forward"):
                pass
            with monitor.stage("model_forward"):
                pass
            backbone = torch.nn.Linear(2, 2)
            monitor.attach_backbone(backbone)
            backbone(torch.zeros(1, 2))
            sync.assert_not_called()
            result = monitor.snapshot(2, 10)
            sync.assert_called_once()
            self.assertEqual(result["perf/model_forward_ms"], 2.0)
            self.assertEqual(result["perf/vit_forward_ms"], 2.0)
            monitor.close()

    def test_fake_tracker_registers_perf_and_sanitizes_warning(self):
        fake = _FakeTracker()
        reporter = TrainctlReporter(tracker=fake)
        reporter.start(name="demo", total_steps=10, message="start")
        reporter.log(2, {"perf/steps_per_second": 4.5, "perf/cpu_rss_gib": 1.2})
        reporter.warn("slow data")
        definitions = next(x[2]["metrics"] if "metrics" in x[2] else x[1][0]
                           for x in fake.calls if x[0] == "define_metrics")
        self.assertIn("perf/steps_per_second", definitions)
        self.assertTrue(any(x[0] == "log" and x[2]["metrics"]["perf/steps_per_second"] == 4.5
                            for x in fake.calls))
        self.assertTrue(any(x[0] == "notify" and x[2]["level"] == "warning"
                            for x in fake.calls))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step.pt"
            path.write_bytes(b"checkpoint")
            reporter.checkpoint(path, step=2,
                                metrics={"val/loss_total": .5, "perf/data_wait_ms": 12.0},
                                message="saved", aliases=["latest"])
        self.assertTrue(any(x[0] == "checkpoint" and
                            x[2]["metrics"]["perf/data_wait_ms"] == 12.0
                            for x in fake.calls))
        checkpoint = build_checkpoint_message({
            "step": 2, "total_steps": 10, "progress": 20, "epoch": 1,
            "path": Path("step.pt"), "size_mib": 1, "validation": {"val/loss_total": .5},
            "train": {}, "elapsed": 10, "steps_per_second": .2, "eta": 40,
            "best_loss": .5, "best_step": 2, "improved": True,
            "performance": {"perf/steps_per_second": 4.5, "perf/cpu_rss_gib": 1.2},
        })
        self.assertIn("4.5 steps/s", checkpoint)
        failure = build_failure_message(RuntimeError("stopped"), step=2,
            window_size=20, checkpoint=None, validation=None, elapsed=10,
            performance={"perf/cpu_rss_gib": 1.2})
        self.assertIn("CPU RSS/peak: 1.2", failure)
        finish = build_finish_message({
            "step": 2, "total_steps": 2, "elapsed": 10,
            "best_loss": .5, "best_step": 2,
            "validation": {"val/loss_total": .5}, "alignment": None,
            "force_prediction": {"force_prediction_mae": .25,
                                 "force_prediction_rmse": .4},
            "checkpoint_count": 1, "last_checkpoint": Path("last.pt"),
            "window_size": 5, "wandb_url": None,
            "performance": {"average_steps_per_second": 1.5,
                            "peak_cpu_rss_gib": 1.2,
                            "evaluation_total_seconds": {"validation": 2.0}},
        })
        self.assertIn("Final force prediction MAE/RMSE: 0.25 / 0.4", finish)
        self.assertIn("Average throughput: 1.5", finish)

    def test_performance_metric_uses_same_wandb_name(self):
        class FakeRun:
            def __init__(self):
                self.logged = []
            def log(self, payload, **kwargs):
                self.logged.append((payload, kwargs))
        with tempfile.TemporaryDirectory() as temporary:
            logger = ExperimentLogger(Path(temporary), SimpleNamespace(enabled=True))
            run = FakeRun()
            logger.run = run
            logger.log_wandb({"perf/steps_per_second": 3.25}, step=4)
            self.assertEqual(run.logged[0][0]["perf/steps_per_second"], 3.25)
            self.assertEqual(run.logged[0][0]["global_step"], 4)

    def test_comparison_uses_only_measured_same_hardware_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for n in (5, 20):
                path = root / f"N{n}.jsonl"
                summary = {
                    "window_size": n, "seed": 42, "train_windows": 100,
                    "hardware": {"host": "same", "machine": "x86_64", "processor": "cpu",
                                 "device": "cpu", "gpu": None},
                    "initial_step": 0, "final_step": 10,
                    "training_active_seconds": 5.0,
                    "average_frames_per_second": float(2 * n),
                    "training_duration_seconds": 8.0,
                    "peak_cpu_rss_gib": 1.0,
                    "peak_gpu_allocated_gib": None,
                    "evaluation_total_seconds": {"validation": 2.0, "retrieval": 1.0},
                    "parquet_cache": {"hit_rate": .5},
                }
                offline = {"record_type": "offline_evaluation", "window_size": n,
                           "seed": 42, "evaluation": "probe", "duration_seconds": 3.0,
                           "hardware": {"host": "same", "gpu": None}}
                path.write_text(json.dumps(summary) + "\n" + json.dumps(offline) + "\n")
                paths.append(path)
            rows, missing = collect(paths)
            self.assertEqual([row["N"] for row in rows], [5, 20])
            self.assertEqual(rows[0]["steps_per_second"], 2.0)
            self.assertEqual(rows[1]["frames_per_second"], 40.0)
            self.assertEqual(rows[0]["force_prediction_seconds"], 3.0)
            self.assertIn(1, missing)
            graphs = write_comparison(rows, missing, root / "comparison")
            self.assertIn("steps_per_second_by_N.png", graphs)
            self.assertNotIn("gpu_memory_by_N.png", graphs)
            modified = paths[1].read_text().replace('"host": "same"', '"host": "other"')
            paths[1].write_text(modified)
            with self.assertRaisesRegex(ValueError, "different hardware"):
                collect(paths)
            paths[1].write_text(paths[0].read_text().replace('"window_size": 5',
                                                               '"window_size": 20').replace(
                '"train_windows": 100', '"train_windows": 100, "batch_size": 4'))
            with self.assertRaisesRegex(ValueError, "different training protocols"):
                collect(paths)


if __name__ == "__main__":
    unittest.main()
