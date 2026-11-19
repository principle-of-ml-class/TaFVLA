"""Small, weight-free checks for the common-anchor comparison protocol."""

from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from utils.comparison_eval import (force_errors, gallery_selection, retrieval_by_dynamics,
                                   ridge_fit, evaluate_probe)
from utils.dynamics import ManifestDataset, create_manifest, dynamics5
from utils.data import FixedValidationBatchSampler, compute_normalization
from model.taf_adapter import TaFAdapter


class FakeSource:
    window_size = 1
    stride = 1
    invalid_sequences = []

    def __init__(self):
        self.episodes = [SimpleNamespace(sensor_type="sensor", sequence_id=f"seq{i}",
                                         episode_index=0, length=45,
                                         frames=np.arange(100, 145),
                                         timestamps=np.arange(45, dtype=float) * .1,
                                         video_timestamps=np.arange(45, dtype=float) * .1,
                                         video_frame_indices=np.arange(45)) for i in range(3)]

    def _sensor_rows(self, ep, offsets):
        return np.repeat(offsets[:, None], 6, axis=1).astype("float32"), np.broadcast_to(
            offsets[:, None, None], (len(offsets), 12, 12)).astype("float32").copy()

    def _images(self, ep, offsets):
        return np.broadcast_to(offsets[:, None, None, None], (len(offsets), 2, 2, 3)).astype("uint8").copy(), ep.timestamps[offsets]


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tactile_encoder = torch.nn.Linear(1, 1)
        self.register_buffer("force_mean", torch.zeros(1, 1, 6))
        self.register_buffer("force_std", torch.ones(1, 1, 6))

    def encode_tactile(self, tactile):
        return self.tactile_encoder(tactile[:, -1, :1, :1, :1].reshape(-1, 1))


class DynamicsComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = FakeSource()
        labels = []
        phases = ("Loading", "Unloading", "Holding", "Transition / Unknown", "No Contact")
        for ep in self.source.episodes:
            for anchor in range(ep.length):
                labels.append({"sensor_configuration": ep.sensor_type, "sequence_id": ep.sequence_id,
                               "episode_index": ep.episode_index, "frame_index": anchor,
                               "timestamp_anchor": float(ep.timestamps[anchor]),
                               "contact_phase": phases[anchor % 5], "sudden_event": anchor == 24,
                               "schema_version": "v1", "configuration_hash": "fixed",
                               "classifier_variant": "centered"})
        self.labels = labels
        self.labels_path = self.root / "centered_labels.parquet"
        self.manifest = self.root / "manifest.parquet"
        self.manifest_number = 0
        self.write_labels(labels)

    def write_labels(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.labels_path)

    def build(self):
        if self.manifest.exists():
            self.manifest_number += 1
            self.manifest = self.root / f"manifest_{self.manifest_number}.parquet"
        return create_manifest(self.source, self.labels_path, self.manifest)

    def test_policy_and_common_anchor_windows(self):
        self.assertEqual(dynamics5("No Contact", True), "sudden_event")
        self.assertIsNone(dynamics5("No Contact", False))
        self.assertEqual(dynamics5("Transition / Unknown", False), "transition")
        with self.assertRaises(ValueError):
            dynamics5("typo", False)
        report = self.build()
        self.assertEqual(report["exclusions"]["insufficient_history"], 57)
        self.assertGreater(report["exclusions"]["no_contact"], 0)
        reference = None
        for n in (1, 3, 5, 7, 10, 20):
            dataset = ManifestDataset(self.source, self.manifest, n)
            signature = [(r["sample_id"], r["dynamics5"], r["split"]) for r in dataset.rows]
            self.assertEqual(signature, reference or signature)
            reference = signature
            for i, row in enumerate(dataset.rows):
                item = dataset[i]
                anchor = row["anchor_frame_position"]
                self.assertEqual(tuple(item["tactile"].shape), (n, 3, 2, 2))
                self.assertEqual(tuple(item["pressure"].shape), (n, 12, 12))
                self.assertEqual(tuple(item["force"].shape), (n, 6))
                self.assertEqual(item["force"][:, 0].tolist(), list(range(anchor-n+1, anchor+1)))
                self.assertEqual((item["tactile"][:, 0, 0, 0]*255).round().int().tolist(),
                                 list(range(anchor-n+1, anchor+1)))
                self.assertEqual(item["metadata"]["frame_indices"].tolist(),
                                 list(range(100+anchor-n+1, 100+anchor+1)))

    def test_label_errors_and_coverage(self):
        duplicate = self.labels + [dict(self.labels[20])]
        self.write_labels(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()
        self.write_labels([r for r in self.labels if not (r["sequence_id"] == "seq0" and r["frame_index"] == 20)])
        self.assertEqual(self.build()["exclusions"]["missing_label"], 1)
        shifted = [dict(r) for r in self.labels]
        shifted[20]["timestamp_anchor"] += .02
        self.write_labels(shifted)
        self.assertEqual(self.build()["exclusions"]["timestamp_or_position_mismatch"], 1)
        foreign = self.labels + [{**self.labels[20], "episode_index": 9}]
        self.write_labels(foreign)
        with self.assertRaisesRegex(ValueError, "unknown episodes"):
            self.build()
        outside = self.labels + [{**self.labels[20], "frame_index": 500}]
        self.write_labels(outside)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.build()
        self.write_labels(self.labels)
        metadata = {"schema_version": "v1", "config_hash": "fixed",
                    "classifier_variant": "centered", "uses_future_observations": True,
                    "config": {"sudden_diagnostic": {"window_size": 20}}}
        (self.root / "centered_metadata.json").write_text(json.dumps(metadata))
        self.assertEqual(self.build()["sudden_diagnostic_window"], 20)
        metadata["config"]["sudden_diagnostic"]["window_size"] = 10
        (self.root / "centered_metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "trailing sudden"):
            self.build()

    def test_full_gallery_and_error_math(self):
        self.build()
        rows = pq.read_table(self.manifest).to_pylist()
        ids = gallery_selection(rows, "test", seed=7, max_pairs=4, min_gap=1)
        self.assertEqual(ids, gallery_selection(rows, "test", seed=7, max_pairs=4, min_gap=1))
        pair = [rows[0], rows[1]]
        query, summary = retrieval_by_dynamics(torch.tensor([[.1, .9], [.2, .8]]), pair)
        self.assertEqual([x["rank"] for x in query], [2, 1])
        self.assertEqual(summary[pair[0]["dynamics5"]]["gallery_size"], 2)
        actual = np.zeros((2, 2, 6))
        predicted = actual.copy()
        predicted[0, 0, 0] = 2
        predicted[1, 1, 3] = 4
        errors = force_errors(predicted, actual)
        self.assertAlmostEqual(errors["whole"]["mae"], 6/24)
        self.assertAlmostEqual(errors["whole"]["rmse"], np.sqrt(20/24))
        self.assertAlmostEqual(errors["endpoint"]["mae"], 4/12)
        x = torch.tensor([[0.], [1.], [2.]])
        y = 3*x+2
        weight, bias = ridge_fit(x, y, 0.)
        self.assertAlmostEqual(float(weight[0, 0]), 3)
        self.assertAlmostEqual(float(bias[0]), 2)

    def test_probe_freezes_encoder_and_uses_train_validation(self):
        self.build()
        dataset = ManifestDataset(self.source, self.manifest, 3)
        model = ProbeModel()
        before = model.tactile_encoder.weight.detach().clone()
        rows, report = evaluate_probe(model, dataset, device=torch.device("cpu"), batch_size=8)
        self.assertTrue(torch.equal(before, model.tactile_encoder.weight))
        self.assertFalse(model.tactile_encoder.weight.requires_grad)
        self.assertEqual({r["split"] for r in rows}, {"validation", "test"})
        self.assertIn(report["alpha"], report["alpha_candidates"])
        test_sequence = next(r["sequence_id"] for r in dataset.rows if r["split"] == "test")
        original = self.source._sensor_rows
        def shifted_test_force(ep, offsets):
            force, pressure = original(ep, offsets)
            return (force + 100 if ep.sequence_id == test_sequence else force), pressure
        self.source._sensor_rows = shifted_test_force
        changed_rows, changed_report = evaluate_probe(model, dataset, device=torch.device("cpu"), batch_size=8)
        self.assertEqual(report["alpha"], changed_report["alpha"])
        self.assertEqual(report["validation_normalized_mse"], changed_report["validation_normalized_mse"])
        self.assertNotEqual([r["force_prediction_mae"] for r in rows if r["split"] == "test"],
                            [r["force_prediction_mae"] for r in changed_rows if r["split"] == "test"])

    def test_shared_normalization_and_batch_plan(self):
        self.build()
        plans = []
        stats = []
        for n in (1, 20):
            dataset = ManifestDataset(self.source, self.manifest, n)
            train = [i for i, r in enumerate(dataset.rows) if r["split"] == "train"]
            ep_ids = {id(ep): i for i, ep in enumerate(dataset.episodes)}
            train_eps = {ep_ids[id(dataset._locate(i)[0])] for i in train}
            stats.append(compute_normalization(dataset, train_eps, 64, 1e-6,
                                               selected_indices=train, normalization_window_size=20))
            sampler = FixedValidationBatchSampler(dataset, train, batch_size=2, min_gap=20, seed=7)
            plans.append(list(sampler))
            self.assertEqual(sampler.report.violating_batches, 0)
            self.assertEqual(sampler.report.used_samples % 2, 0)
        self.assertEqual(plans[0], plans[1])
        for key in stats[0]:
            self.assertTrue(torch.equal(stats[0][key], stats[1][key]))

    def test_joint_loss_gradients_all_windows(self):
        class TinyTactile(torch.nn.Module):
            def __init__(self, output_dim, **kwargs):
                super().__init__()
                self.projection = torch.nn.Linear(3, output_dim)

            def forward(self, tactile):
                return self.projection(tactile.mean(dim=(1, 3, 4)))

        with patch("model.taf_adapter.TactileEncoder", TinyTactile):
            tactile_initial = []
            force_counts = []
            for n in (1, 3, 5, 7, 10, 20):
                model = TaFAdapter(window_size=n, force_encoder={"hidden_dim": 8,
                                     "pressure_latent_dim": 4, "force_latent_dim": 4},
                                     vq={"pressure_codebook_size": 8, "force_codebook_size": 8},
                                     tactile_initialization_seed=142)
                tactile_initial.append(model.tactile_encoder.projection.weight.detach().clone())
                force_counts.append(sum(p.numel() for p in model.force_encoder.parameters()))
                output = model(torch.rand(2, n, 3, 2, 2), torch.rand(2, n, 6),
                               torch.rand(2, n, 12, 12))
                self.assertEqual(tuple(output["reconstructed_force"].shape), (2, n, 6))
                output["loss_total"].backward()
                self.assertIsNotNone(model.tactile_encoder.projection.weight.grad)
                self.assertIsNotNone(model.force_encoder.force.encoder[0].weight.grad)
            self.assertTrue(all(torch.equal(tactile_initial[0], x) for x in tactile_initial[1:]))
            self.assertEqual(len(set(force_counts)), 6)


if __name__ == "__main__":
    unittest.main()
