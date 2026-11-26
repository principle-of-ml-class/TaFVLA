"""Bounded CPU checks for initialization, actual AMP skips and update-based LR."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from model.force_encoder import ForceEncoder
from model.vector_quantizer import VectorQuantizer
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.codebook_init import initialize_codebooks, kmeans_plus_plus, sample_train_windows
from utils.common import validate_config
from utils.config import register_config_schema
from utils.optimization import UpdateLRScheduler, gradient_metrics, optimizer_update


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(type="cosine", warmup_steps=500, warmup_start_factor=.01, min_lr=1e-5)


class SensorDataset:
    window_size = 2

    def __init__(self):
        self.locations, self.rows = [], []
        for sequence in ("a", "b", "c"):
            for episode in range(3):
                ep = SimpleNamespace(sensor_type="sensor", sequence_id=sequence, episode_index=episode)
                for start in range(100):
                    self.locations.append((ep, start))
                    self.rows.append({"split": "train" if episode < 2 else "validation"})

    def _locate(self, index):
        return self.locations[index]

    def _sensor_rows(self, ep, offsets):
        value = np.asarray(offsets, dtype=np.float32) + ep.episode_index * .1
        return np.broadcast_to(value[:, None], (2, 6)).copy(), np.broadcast_to(value[:, None, None], (2, 12, 12)).copy()

    def __getitem__(self, index):
        raise AssertionError("Initialization must never load tactile images")


class SensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.force_encoder = ForceEncoder(2, hidden_dim=16, pressure_latent_dim=4, force_latent_dim=3,
                                          pressure_codebook_size=8, force_codebook_size=7)

    def _normalize(self, pressure, force):
        return (pressure-50)/30, (force-50)/30


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kmeans_shape_finite_reproducible_and_distance(self):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(128, 5, generator=generator) + 4
        original = torch.zeros(8, 5)
        a, stats = kmeans_plus_plus(x, 8, 20, 42)
        b, repeat = kmeans_plus_plus(x, 8, 20, 42)
        self.assertEqual(tuple(a.shape), (8, 5))
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast_centers, _ = kmeans_plus_plus(x, 8, 20, 42)
        torch.testing.assert_close(a, autocast_centers, rtol=0, atol=0)
        self.assertEqual(stats, repeat)
        self.assertLess(stats["quantization_distance_after"], float(torch.cdist(x, original).square().min(1).values.mean()))
        self.assertEqual(sum(stats["cluster_occupancy"]), 128)
        with self.assertRaisesRegex(ValueError, "valid latents"):
            kmeans_plus_plus(torch.full((10, 5), float("nan")), 8, 2, 42)
        with self.assertRaisesRegex(ValueError, "valid latents"):
            kmeans_plus_plus(x[:7], 8, 2, 42)

    def test_empty_clusters_reinitialized_from_valid_pool(self):
        centers, stats = kmeans_plus_plus(torch.ones(10, 3), 8, 2, 42)
        torch.testing.assert_close(centers, torch.ones(8, 3))
        self.assertGreater(stats["empty_cluster_reinitializations"], 0)
        self.assertEqual(stats["empty_clusters"], 7)

    def test_train_only_sampling_balanced_and_temporally_spread(self):
        dataset = SensorDataset()
        train = [i for i, row in enumerate(dataset.rows) if row["split"] == "train"]
        selected, stats = sample_train_windows(dataset, train, 60, 42)
        self.assertEqual(selected, sample_train_windows(dataset, train, 60, 42)[0])
        self.assertEqual(len(set(selected)), 60)
        self.assertTrue(set(selected) <= set(train))
        self.assertEqual(stats["episode_count"], 6)
        self.assertEqual(stats["sequence_count"], 3)
        self.assertEqual(set(stats["sequence_support"].values()), {20})
        self.assertEqual(set(stats["episode_support"].values()), {10})
        self.assertGreater(stats["mean_window_start_gap"], 8)
        with self.assertRaisesRegex(ValueError, "non-train"):
            sample_train_windows(dataset, [200], 1, 42)

    def test_sensor_initialization_and_resume_uniform_guards(self):
        dataset = SensorDataset()
        train = [i for i, row in enumerate(dataset.rows) if row["split"] == "train"]
        torch.manual_seed(42)
        model = SensorModel()
        other = copy.deepcopy(model)
        options = dict(initialization="kmeans", kmeans_samples=60, kmeans_iters=4, kmeans_seed=42)
        rng = torch.get_rng_state().clone()
        report = initialize_codebooks(model, dataset, train, options)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        initialize_codebooks(other, dataset, train, options)
        for branch in ("pressure", "force"):
            w = getattr(model.force_encoder, branch).quantizer.codebook.weight
            torch.testing.assert_close(w, getattr(other.force_encoder, branch).quantizer.codebook.weight, rtol=0, atol=0)
            self.assertLess(report[branch]["quantization_distance_after"], report[branch]["quantization_distance_before"])
        before = copy.deepcopy(model.state_dict())
        # A real checkpoint roundtrip restores both initialized codebooks first.
        with tempfile.TemporaryDirectory() as directory:
            optimizer = torch.optim.AdamW(model.parameters())
            path = Path(directory)/"initialized.pt"
            save_checkpoint(path, model=model, optimizer=optimizer,
                            scaler=torch.amp.GradScaler("cpu", enabled=False), step=0, epoch=0,
                            config={}, normalization={}, train_keys=[], val_keys=[])
            model = SensorModel()
            model.load_state_dict(load_checkpoint(path)["model"])
        with patch("utils.codebook_init.sample_train_windows", side_effect=AssertionError("must not resample")):
            self.assertIsNone(initialize_codebooks(model, dataset, train, options, resume=True))
            self.assertIsNone(initialize_codebooks(model, dataset, train, {"initialization": "uniform"}))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "distinct train windows"):
            initialize_codebooks(model, dataset, train[:3], options)

    def test_uniform_and_straight_through_regression(self):
        torch.manual_seed(9)
        quantizer = VectorQuantizer(8, 4)
        self.assertTrue((quantizer.codebook.weight.abs() <= 1/8).all())
        x = torch.randn(6, 4, requires_grad=True)
        output = quantizer(x)
        labels = torch.cdist(x, quantizer.codebook.weight).argmin(1)
        self.assertTrue(torch.equal(labels, output["indices"]))
        output["quantized"].sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        self.assertIsNone(quantizer.codebook.weight.grad)
        quantizer.zero_grad()
        quantizer(x)["loss"].backward()
        self.assertGreater(float(quantizer.codebook.weight.grad.norm()), 0)

    def test_lr_boundaries_and_no_scheduler_legacy(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        scheduler = UpdateLRScheduler(optimizer, OPTIONS, 10000, 1e-4)
        rates = []
        for _ in range(10000):
            rates.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        self.assertAlmostEqual(rates[0], 1e-6, places=14)
        self.assertAlmostEqual(rates[499], 1e-4, places=14)
        self.assertAlmostEqual(rates[-1], 1e-5, places=14)
        self.assertTrue(all(a <= b for a, b in zip(rates[:499], rates[1:500])))
        self.assertTrue(all(a >= b for a, b in zip(rates[499:-1], rates[500:])))
        self.assertEqual(scheduler.phase(500), "warmup")
        self.assertEqual(scheduler.phase(501), "cosine")
        legacy = UpdateLRScheduler(optimizer, None, 10, 1e-4)
        legacy.restore(None, 3)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_invalid_scheduler_options_are_explicit(self):
        for change in ({"type": "unknown"}, {"warmup_steps": 1}, {"warmup_steps": 10000},
                       {"min_lr": .1}, {"warmup_start_factor": 0}, {"typo": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                UpdateLRScheduler.validate_options(OPTIONS | change, 1e-4, 10000)

    def test_accumulation_and_actual_amp_overflow(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        scheduler = UpdateLRScheduler(optimizer, OPTIONS, 10000, 1e-4)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.)
        # Two accumulated microbatches cause exactly one optimizer/scheduler update.
        for _ in range(2):
            scaler.scale(model(torch.ones(2, 2)).square().mean()/2).backward()
            self.assertEqual(scheduler.completed_updates, 0)
        scaler.unscale_(optimizer)
        stats = gradient_metrics(model, .01)
        self.assertLessEqual(float(stats["train/gradient_norm_after_clipping"]), .010001)
        self.assertLess(float(stats["train/gradient_clipping_coefficient"]), 1.)
        self.assertTrue(optimizer_update(optimizer, scaler, scheduler))
        self.assertEqual(scheduler.completed_updates, 1)
        optimizer.zero_grad(set_to_none=True)
        # Real GradScaler detects injected inf, skips weights, and lowers its scale.
        scaler.scale(model(torch.ones(2, 2)).sum()).backward()
        next(model.parameters()).grad.fill_(float("inf"))
        scaler.unscale_(optimizer)
        state = copy.deepcopy(model.state_dict())
        rate = optimizer.param_groups[0]["lr"]
        self.assertFalse(optimizer_update(optimizer, scaler, scheduler))
        self.assertEqual(scheduler.completed_updates, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], rate)
        for key, value in state.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_lr_continuity_and_mismatch_rejected(self):
        model = torch.nn.Linear(2, 1)
        for boundary in (300, 500, 5000):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = UpdateLRScheduler(optimizer, OPTIONS, 10000, 1e-4)
            for _ in range(boundary):
                scheduler.step()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"checkpoint.pt"
                save_checkpoint(path, model=model, optimizer=optimizer,
                                scaler=torch.amp.GradScaler("cpu"), step=boundary, epoch=1,
                                config={}, normalization={}, train_keys=[], val_keys=[],
                                scheduler_state=scheduler.state_dict(), stability_state={"amp_skipped_updates": 2})
                checkpoint = load_checkpoint(path)
            resumed_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            resumed = UpdateLRScheduler(resumed_optimizer, OPTIONS, 10000, 1e-4)
            resumed.restore(checkpoint["scheduler"], boundary)
            for _ in range(4):
                self.assertEqual(optimizer.param_groups[0]["lr"], resumed_optimizer.param_groups[0]["lr"])
                scheduler.step()
                resumed.step()
            self.assertEqual(checkpoint["stability"]["amp_skipped_updates"], 2)
            with self.assertRaisesRegex(ValueError, "settings/step"):
                resumed.restore(checkpoint["scheduler"], boundary+1)
            with self.assertRaisesRegex(ValueError, "without saved"):
                resumed.restore(None, boundary)

    def test_hydra_preserves_episode_settings_and_independent_switches(self):
        register_config_schema()
        with initialize_config_dir(config_dir=str(ROOT/"config"), version_base="1.3"):
            cfg = compose(config_name="taf_adapter_sparsh_episode_stable")
            legacy = compose(config_name="taf_adapter_sparsh_episode")
            uniform_cosine = compose(config_name="taf_adapter_sparsh_episode_stable", overrides=["vq.initialization=uniform"])
            kmeans_constant = compose(config_name="taf_adapter_sparsh_episode_stable", overrides=["training.scheduler.type=constant"])
            no_scheduler = compose(config_name="taf_adapter_sparsh_episode", overrides=["training.scheduler=null"])
        for config in (cfg, legacy, uniform_cosine, kmeans_constant, no_scheduler):
            validate_config(config)
        saved = OmegaConf.load(ROOT/"outputs/episode_split_validation/resolved_config.yaml")
        current = OmegaConf.to_container(cfg, resolve=True)
        allowed = {"vq": {"initialization", "kmeans_samples", "kmeans_iters", "kmeans_seed"},
                   "training": {"scheduler", "output_dir"}, "wandb": {"name", "group"}}
        for section, values in current.items():
            for key, value in values.items():
                if key not in allowed.get(section, set()):
                    self.assertEqual(value, OmegaConf.to_container(saved, resolve=True)[section][key], f"{section}.{key}")
        self.assertEqual(legacy.vq.initialization, "uniform")
        self.assertEqual(legacy.training.scheduler.type, "constant")

    def test_small_learning_rate_is_visible(self):
        from utils.logging import TrainingProgress
        self.assertEqual(TrainingProgress._value({"train/learning_rate": 1e-6}, "train/learning_rate"), "1.00e-06")

    def test_resume_preserves_actual_adamw_updates(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
        scheduler = UpdateLRScheduler(optimizer, OPTIONS | {"warmup_steps": 2}, 7, 1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        def update(module, opt, schedule, scale):
            opt.zero_grad(set_to_none=True)
            module(torch.tensor([[1., 2.], [3., 4.]])).square().mean().backward()
            lr = opt.param_groups[0]["lr"]
            self.assertTrue(optimizer_update(opt, scale, schedule))
            return lr
        for _ in range(3):
            update(model, optimizer, scheduler, scaler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resume.pt"
            save_checkpoint(path, model=model, optimizer=optimizer, scaler=scaler,
                            step=3, epoch=0, config={}, normalization={}, train_keys=[], val_keys=[],
                            scheduler_state=scheduler.state_dict())
            saved = load_checkpoint(path)
        resumed = torch.nn.Linear(2, 1); resumed.load_state_dict(saved["model"])
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=.001)
        resumed_opt.load_state_dict(saved["optimizer"])
        resumed_scheduler = UpdateLRScheduler(resumed_opt, OPTIONS | {"warmup_steps": 2}, 7, 1e-4)
        resumed_scheduler.restore(saved["scheduler"], saved["step"])
        resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
        resumed_scaler.load_state_dict(saved["scaler"])
        for _ in range(4):
            self.assertEqual(update(model, optimizer, scheduler, scaler),
                             update(resumed, resumed_opt, resumed_scheduler, resumed_scaler))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)

    def test_start_notification_contains_stability_settings(self):
        from tests.test_trainctl import start_info
        from utils.trainctl import build_start_message
        register_config_schema()
        with initialize_config_dir(config_dir=str(ROOT/"config"), version_base="1.3"):
            cfg = compose(config_name="taf_adapter_sparsh_episode_stable")
        info = start_info(cfg) | {"codebook_initialization": cfg.vq.initialization,
                                  "scheduler_type": cfg.training.scheduler.type,
                                  "warmup_steps": cfg.training.scheduler.warmup_steps,
                                  "minimum_lr": cfg.training.scheduler.min_lr}
        message = build_start_message(info)
        for expected in ("initialization: kmeans", "Scheduler: cosine", "warmup steps: 500",
                         "peak LR: 0.0001", "minimum LR: 1e-05"):
            self.assertIn(expected, message)


if __name__ == "__main__":
    unittest.main()
