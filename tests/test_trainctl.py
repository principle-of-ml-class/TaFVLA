"""Mock-only tests for the optional trainctl training integration."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from utils.config import register_config_schema
from utils.trainctl import (
    FINISH_MESSAGE,
    START_MESSAGE,
    TrainctlReporter,
    ValidationHistory,
    build_checkpoint_message,
    build_finish_message,
    build_start_message,
    finite_metrics,
    redact_secrets,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def start_info(config) -> dict:
    return {
        "window_size": int(config.data.window_size),
        "experiment": "override-test",
        "run_id": "safe-run-id",
        "stride": int(config.data.stride),
        "batch_size": int(config.training.batch_size),
        "accumulation": int(config.training.gradient_accumulation_steps),
        "effective_batch_size": (
            int(config.training.batch_size)
            * int(config.training.gradient_accumulation_steps)
        ),
        "optimizer": "AdamW",
        "learning_rate": float(config.training.learning_rate),
        "initial_step": 0,
        "total_steps": int(config.training.max_steps),
        "validate_every": int(config.training.validate_every),
        "save_every": int(config.training.save_every),
        "train_windows": 120,
        "train_groups": 8,
        "val_windows": 24,
        "val_groups": 2,
        "amp": bool(config.training.amp),
        "device": "cpu",
        "gpu": None,
        "backbone": str(config.tactile_encoder.backbone),
        "freeze_backbone": bool(config.tactile_encoder.freeze_backbone),
        "output_dir": str(config.training.output_dir),
        "resume": None,
        "wandb_url": None,
    }


class TrainctlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_config_schema()

    def config(self, *overrides: str):
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
            return compose(config_name="taf_adapter", overrides=list(overrides))

    def test_exact_start_and_finish_phrases_are_sent(self) -> None:
        fake = FakeTracker()
        reporter = TrainctlReporter(tracker=fake)
        start = build_start_message(start_info(self.config("wandb.enabled=false")))
        finish = build_finish_message({
            "step": 10, "total_steps": 10, "elapsed": 12.0,
            "best_loss": 0.4, "best_step": 10,
            "validation": {"val/loss_total": 0.4, "val/recall_at_1": 0.5,
                           "val/recall_at_5": 0.8},
            "alignment": None, "checkpoint_count": 2,
            "last_checkpoint": Path("last.pt"), "wandb_url": None,
            "window_size": 5,
        })
        reporter.start(name=None, total_steps=10, message=start)
        reporter.finish(finish)
        start_call = next(call for call in fake.calls if call[0] == "start")
        finish_call = next(call for call in fake.calls if call[0] == "finish")
        self.assertEqual(start_call[2]["message"].splitlines()[0], START_MESSAGE)
        self.assertEqual(finish_call[2]["message"].splitlines()[0], FINISH_MESSAGE)
        init = next(call for call in fake.calls if call[0] == "init")
        self.assertFalse(init[2]["strict"])
        self.assertTrue(init[2]["rank_zero_only"])

    def test_start_message_reflects_hydra_overrides(self) -> None:
        config = self.config(
            "data.window_size=9", "data.stride=3", "training.batch_size=7",
            "training.gradient_accumulation_steps=4",
            "training.learning_rate=3e-5", "training.max_steps=1234",
            "training.output_dir=outputs/override", "wandb.enabled=false",
        )
        message = build_start_message(start_info(config))
        self.assertIn("**Window size T: 9**", message)
        self.assertIn("Stride: 3", message)
        self.assertIn("Batch: 7 × accumulation 4 = effective 28", message)
        self.assertIn("LR: 3e-05", message)
        self.assertIn("Steps: 0 → 1234", message)
        self.assertIn("Output: outputs/override", message)

    def test_validation_cache_handles_overlapping_and_nondivisible_schedules(self) -> None:
        for validate_every, save_every, total in ((500, 1000, 2000), (600, 1000, 3000)):
            history = ValidationHistory()
            calls: list[int] = []
            checkpoint_steps: list[int] = []
            for step in range(1, total + 1):
                if step % validate_every == 0 or step % save_every == 0:
                    history.ensure(
                        step,
                        lambda step=step: (
                            calls.append(step) or {"val/loss_total": 1 / step}
                        ),
                    )
                if step % save_every == 0:
                    checkpoint_steps.append(step)
                    self.assertEqual(history.last_step, step)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertTrue(set(checkpoint_steps).issubset(calls))
        self.assertEqual(calls, [600, 1000, 1200, 1800, 2000, 2400, 3000])

    def test_last_checkpoint_validation_is_never_missing(self) -> None:
        history = ValidationHistory()
        calls = 0

        def evaluate():
            nonlocal calls
            calls += 1
            return {"val/loss_total": 0.7}

        history.ensure(900, evaluate)
        final, ran = history.ensure(950, evaluate)
        self.assertTrue(ran)
        self.assertEqual(calls, 2)
        self.assertEqual(history.last_step, 950)
        self.assertEqual(final["val/loss_total"], 0.7)
        _, ran_again = history.ensure(950, evaluate)
        self.assertFalse(ran_again)
        self.assertEqual(calls, 2)

    def test_alignment_is_rendered_only_when_present(self) -> None:
        base = {
            "step": 1000, "total_steps": 2000, "progress": 50.0,
            "epoch": 2, "path": Path("step_00001000.pt"), "size_mib": 1.2,
            "validation": {"val/loss_total": 0.5}, "train": {},
            "elapsed": 60.0, "steps_per_second": 10.0, "eta": 100.0,
            "best_loss": 0.5, "best_step": 1000, "improved": True,
        }
        without = build_checkpoint_message({**base, "alignment": None})
        self.assertNotIn("Alignment t2f", without)
        with_alignment = build_checkpoint_message({
            **base,
            "alignment": {
                "alignment/t2f/recall_at_1": 0.2,
                "alignment/t2f/recall_at_5": 0.6,
                "alignment/f2t/recall_at_1": 0.3,
                "alignment/f2t/recall_at_5": 0.7,
            },
        })
        self.assertIn("Alignment t2f R@1/R@5", with_alignment)

    def test_missing_or_failed_checkpoint_has_no_success_event_and_deduplicates(self) -> None:
        fake = FakeTracker()
        reporter = TrainctlReporter(tracker=fake)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_00001000.pt"
            self.assertFalse(reporter.checkpoint(
                path, step=1000, metrics={}, message="not saved", aliases=[],
            ))
            self.assertFalse(any(call[0] == "checkpoint" for call in fake.calls))
            path.write_bytes(b"saved")
            self.assertTrue(reporter.checkpoint(
                path, step=1000, metrics={"val/loss_total": 1.0},
                message="saved", aliases=["latest"],
            ))
            self.assertFalse(reporter.checkpoint(
                path, step=1000, metrics={"val/loss_total": 1.0},
                message="epoch rewrite", aliases=["latest"],
            ))
        self.assertEqual(sum(call[0] == "checkpoint" for call in fake.calls), 1)

    def test_failure_does_not_emit_success_finish(self) -> None:
        fake = FakeTracker()
        reporter = TrainctlReporter(tracker=fake)
        reporter.fail("RuntimeError: boom")
        self.assertEqual([call[0] for call in fake.calls], ["fail"])
        self.assertNotIn(FINISH_MESSAGE, str(fake.calls))

    def test_absent_trainctl_module_is_a_safe_noop(self) -> None:
        reporter = TrainctlReporter(tracker=None)
        reporter.start(name=None, total_steps=1, message=START_MESSAGE)
        reporter.log(1, {"train/loss_total": 1.0})
        reporter.finish(FINISH_MESSAGE)

    def test_credentials_are_redacted_and_finite_metrics_only(self) -> None:
        secret = "https://discord.com/api/" + "webhooks/123/unit-test-secret"
        with patch.dict(os.environ, {"TRAINCTL_DISCORD_WEBHOOK_URL": secret}):
            safe = redact_secrets(f"failure contacting {secret}")
        self.assertNotIn("unit-test-secret", safe)
        self.assertNotIn(secret, build_start_message(start_info(self.config())))
        filtered = finite_metrics({
            "good": 1.25,
            "tensor": torch.tensor(2.0),
            "nan": math.nan,
            "inf": math.inf,
            "bool": True,
            "text": "credential",
        })
        self.assertEqual(filtered, {"good": 1.25, "tensor": 2.0})
        self.assertTrue(all(math.isfinite(float(value)) for value in filtered.values()))


if __name__ == "__main__":
    unittest.main()
