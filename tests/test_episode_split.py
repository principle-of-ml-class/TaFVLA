"""Whole-episode repartition, legacy behavior and train-only statistics."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir

from tests.test_dynamics_comparison import FakeSource
from utils.common import validate_config
from utils.comparison_eval import save_gallery_plan
from utils.config import register_config_schema
from utils.data import compute_normalization, split_by_group
from utils.dynamics import ManifestDataset, sample_id, split_group_key
from utils.episode_split import repartition_episode_manifest


class EpisodeSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.original = self.root/"original.parquet"
        self.output = self.root/"episode/manifest.parquet"
        self.source = FakeSource(); template = self.source.episodes[0]; self.source.episodes = []
        rows = []
        for seq, count, split in (("a", 3, "train"), ("b", 3, "validation"), ("single", 1, "train"), ("test", 2, "test")):
            for ep_index in range(count):
                ep = copy.copy(template); ep.sequence_id = seq; ep.episode_index = ep_index
                self.source.episodes.append(ep)
                for anchor in range(19, ep.length):
                    rows.append(dict(sample_id=sample_id("sensor", seq, ep_index, anchor),
                                     sensor_configuration="sensor", sequence_id=seq, episode_index=ep_index,
                                     anchor_frame_position=anchor, source_frame_index=int(ep.frames[anchor]),
                                     timestamp_anchor=float(ep.timestamps[anchor]), contact_phase="Loading",
                                     sudden_event=False, dynamics5="loading", split=split,
                                     manifest_id="parent", split_seed=42, label_sha256="labels"))
        self.rows = rows; pq.write_table(pa.Table.from_pylist(rows), self.original)

    def test_determinism_row_identity_test_and_singletons(self):
        before = self.original.read_bytes()
        report = repartition_episode_manifest(self.original, self.output)
        other = self.root/"other/manifest.parquet"
        repeat = repartition_episode_manifest(self.original, other)
        self.assertEqual(report, repeat)
        self.assertTrue(pq.read_table(self.output).equals(pq.read_table(other)))
        self.assertEqual(self.original.read_bytes(), before)
        self.assertEqual(report["shared_train_validation_sequences"], 2)
        self.assertEqual(report["singleton_train_only_sequences"], [["sensor", "single"]])
        new = pq.read_table(self.output).to_pylist()
        self.assertEqual([r["sample_id"] for r in self.rows], [r["sample_id"] for r in new])
        self.assertEqual([r["sample_id"] for r in self.rows if r["split"] == "test"],
                         [r["sample_id"] for r in new if r["split"] == "test"])
        self.assertEqual(repartition_episode_manifest(self.original, self.output), report)
        self.assertNotEqual(repartition_episode_manifest(self.original, self.root/"seed43.parquet", seed=43)["manifest_id"], report["manifest_id"])
        with self.assertRaises(ValueError):
            repartition_episode_manifest(self.original, self.original)

    def test_legacy_sequence_and_episode_leakage_guards(self):
        legacy = ManifestDataset(self.source, self.original, 5, expected_split_unit="sequence")
        self.assertEqual(legacy.split_unit, "sequence")
        repartition_episode_manifest(self.original, self.output)
        dataset = ManifestDataset(self.source, self.output, 5, expected_split_unit="episode")
        self.assertEqual(dataset.split_unit, "episode")
        with self.assertRaisesRegex(ValueError, "split_unit"):
            ManifestDataset(self.source, self.output, 5, expected_split_unit="sequence")
        rows = pq.read_table(self.output).to_pylist()
        rows[0]["split"] = "validation" if rows[0]["split"] == "train" else "train"
        path = self.root/"leak.parquet"; pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(ValueError, "Episode crosses"):
            ManifestDataset(self.source, path, 5)
        rows = pq.read_table(self.output).to_pylist()
        for row in rows:
            row["split_unit"] = "sequence"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(ValueError, "Sequence crosses"):
            ManifestDataset(self.source, path, 5)
        rows = pq.read_table(self.output).to_pylist()
        for row in rows:
            if row["sequence_id"] == "a" and row["episode_index"] == 0:
                row["split"] = "validation" if row["split"] == "train" else "train"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            ManifestDataset(self.source, path, 5)

    def test_train_only_normalization_shared_across_windows(self):
        repartition_episode_manifest(self.original, self.output)
        normals = []
        for n in (1, 5, 20):
            dataset = ManifestDataset(self.source, self.output, n)
            indices = [i for i, r in enumerate(dataset.rows) if r["split"] == "train"]
            ep_lookup = {id(ep): i for i, ep in enumerate(dataset.episodes)}
            ep_ids = {ep_lookup[id(dataset._locate(i)[0])] for i in indices}
            original_reader = self.source._sensor_rows
            # Every held-out read is an error, rather than merely a different value.
            def train_only(ep, offsets):
                if ep_lookup[id(ep)] not in ep_ids:
                    raise AssertionError("Held-out sensor values were used for normalization")
                return original_reader(ep, offsets)
            self.source._sensor_rows = train_only
            try:
                normals.append(compute_normalization(dataset, ep_ids, 32, 1e-6, selected_indices=indices,
                                                     normalization_window_size=20))
            finally:
                self.source._sensor_rows = original_reader
        for key in normals[0]:
            self.assertTrue(all(torch.equal(normals[0][key], v[key]) for v in normals))
        row = self.rows[0]
        self.assertNotEqual(split_group_key(row, "episode"), split_group_key({**row, "sensor_configuration": "other"}, "episode"))

    def test_test_gallery_preservation_and_config(self):
        old = save_gallery_plan(self.rows, self.root/"old_gallery.json", seed=42, max_pairs=4)
        repartition_episode_manifest(self.original, self.output)
        rows = pq.read_table(self.output).to_pylist()
        new = save_gallery_plan(rows, self.root/"gallery.json", seed=42, max_pairs=4, preserved_test_ids=old["test"])
        self.assertEqual(old["test"], new["test"])
        self.assertNotEqual(old["manifest_id"], new["manifest_id"])
        register_config_schema()
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"config"), version_base="1.3"):
            seq = compose(config_name="taf_adapter_sparsh")
            ep = compose(config_name="taf_adapter_sparsh_episode")
        validate_config(seq); validate_config(ep)
        for section in ("tactile_encoder", "force_encoder", "vq", "loss"):
            self.assertEqual(seq[section], ep[section])
        for key in seq.training:
            if key not in {"output_dir"}:
                self.assertEqual(seq.training[key], ep.training[key])
        self.assertEqual(ep.data.split_unit, "episode")
        self.assertIsNone(ep.training.resume)
        for n in (1, 3, 5, 7, 10, 20):
            ep.data.window_size = n
            validate_config(ep)
            self.assertEqual(ep.comparison.manifest, "metadata/episode_split/manifest.parquet")

    def test_sensor_is_part_of_standard_episode_group(self):
        dataset = SimpleNamespace(episodes=[SimpleNamespace(sensor_type=s, sequence_id="same", episode_index=0)
                                            for s in ("a", "b")], _ends=[2, 4])
        train, val, _, _, train_groups, val_groups = split_by_group(dataset, unit="episode",
                                                                  validation_fraction=.5, seed=42)
        self.assertEqual(len(train), 2); self.assertEqual(len(val), 2)
        self.assertTrue(set(train_groups).isdisjoint(val_groups))

    def test_sequence_resume_is_rejected_before_data_loading(self):
        from script.train_adapter import run_training
        register_config_schema()
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"config"), version_base="1.3"):
            cfg = compose(config_name="taf_adapter_sparsh_episode", overrides=["training.resume=unused.pt"])
        with patch("script.train_adapter.load_checkpoint", return_value={"config": {"data": {"split_unit": "sequence"}}}), \
                patch("script.train_adapter.assert_tactile_checkpoint_compatible"), \
                patch("script.train_adapter.TaFDataset") as loader:
            with self.assertRaisesRegex(ValueError, "split unit differs"):
                run_training(cfg)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
