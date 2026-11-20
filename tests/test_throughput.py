"""Regression tests for cache bounds, frame identity, and deferred metrics."""

from __future__ import annotations

import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from dataset.taf_dataset import TaFDataset
from model.taf_adapter import TaFAdapter
from model.tactile_encoder import TactileEncoder
from utils.frame_cache import CachedWindowDataset, FrozenFrameCache
from utils.logging import DeviceMetricAccumulator, METRIC_KEYS


class _TinyBackbone(nn.Module):
    num_features = 4
    pretrained_cfg = {"input_size": (3, 2, 2), "mean": (0.0, 0.0, 0.0),
                      "std": (1.0, 1.0, 1.0)}

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 4)

    def forward(self, x):
        return self.linear(x.flatten(1))


class ThroughputTests(unittest.TestCase):
    def test_bounded_sensor_lru_preserves_index_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = []
            for number in range(3):
                path = Path(temporary) / f"{number}.parquet"
                indices = np.arange(number * 10, number * 10 + 3)
                pq.write_table(pa.table({
                    "index": indices,
                    "observation.force_torque": pa.array(
                        np.full((3, 6), number, dtype=np.float32).tolist()),
                    "observation.pressure_matrix": pa.array(
                        np.full((3, 144), number + 1, dtype=np.float32).tolist()),
                }), path)
                files.append(path)
            ds = object.__new__(TaFDataset)
            ds.sensor_cache_max_bytes = (3 * (8 + 6 * 4 + 144 * 4)) * 2
            ds._sensor_cache = OrderedDict()
            ds._sensor_cache_bytes = 0
            ds._sensor_shared_stats = None
            ds._pipeline_shared_stats = None
            ds._sensor_stats = {"parquet_reads": 0, "cache_hits": 0, "cache_misses": 0}
            episodes = [SimpleNamespace(parquet=p, row_start=0, episode_index=i)
                        for i, p in enumerate(files)]
            for ep in (episodes[0], episodes[1], episodes[0], episodes[2], episodes[1]):
                force, pressure = ds._sensor_rows(ep, np.array([1, 2]))
                self.assertTrue(np.all(force == ep.episode_index))
                self.assertTrue(np.all(pressure == ep.episode_index + 1))
                self.assertLessEqual(ds.sensor_cache_stats()["bytes"], ds.sensor_cache_max_bytes)
            stats = ds.sensor_cache_stats()
            self.assertEqual(stats["parquet_reads"], 4)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["cache_misses"], 4)
            self.assertEqual(stats["hit_rate"], 0.2)
            ds.sensor_cache_max_bytes = 0
            ds._sensor_cache.clear()
            ds._sensor_cache_bytes = 0
            ds._sensor_rows(episodes[0], np.array([0]))
            self.assertEqual(ds.sensor_cache_stats()["entries"], 0)

    def test_uint8_image_matches_previous_float_values(self):
        ds = object.__new__(TaFDataset)
        images = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
        ds.image_dtype = "uint8"
        compact = ds._image_tensor(images)
        self.assertEqual(compact.dtype, torch.uint8)
        ds.image_dtype = "float32"
        old = ds._image_tensor(images)
        self.assertTrue(torch.equal(compact.float().div(255), old))

    def test_frozen_frame_cache_reused_across_n_and_projection_trains(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"identity")
            with patch("timm.create_model", return_value=_TinyBackbone()):
                encoder = TactileEncoder(transformer_dim=4, output_dim=4,
                                         transformer_layers=1, num_heads=2,
                                         dropout=0.0, max_window_size=20)
            model = SimpleNamespace(tactile_encoder=encoder)
            frames = np.arange(21 * 12, dtype=np.uint8).reshape(21, 2, 2, 3)
            ep = SimpleNamespace(video=video, video_frame_indices=np.arange(21),
                                 video_timestamps=np.arange(21) / 30, length=21)
            calls = []

            def decode(_ep, offsets):
                calls.append(tuple(offsets))
                return frames[offsets], np.asarray(offsets)

            def sensors(_ep, offsets):
                return (np.zeros((len(offsets), 6), np.float32),
                        np.zeros((len(offsets), 12, 12), np.float32))

            source = SimpleNamespace(all_episodes=[ep], episodes=[ep], image_size=(2, 2),
                                     _images=decode, _sensor_rows=sensors)
            cache5 = FrozenFrameCache(source, model, Path(temporary) / "cache",
                                      device=torch.device("cpu"), amp_enabled=False,
                                      batch_size=4)
            first_count = len(calls)
            self.assertGreater(first_count, 0)
            cache20 = FrozenFrameCache(source, model, Path(temporary) / "cache",
                                       device=torch.device("cpu"), amp_enabled=False,
                                       batch_size=4)
            self.assertEqual(len(calls), first_count)
            for n, cache in ((5, cache5), (20, cache20)):
                offsets = np.arange(n)
                raw = torch.from_numpy(frames[offsets]).permute(0, 3, 1, 2).unsqueeze(0)
                raw_projected = encoder.encode_frames(raw)
                encoder.backbone_chunk_frames = 2
                chunked_projected = encoder.encode_frames(raw)
                encoder.backbone_chunk_frames = 0
                self.assertTrue(torch.allclose(raw_projected, chunked_projected, atol=1e-6))
                cached = cache.features(ep, offsets).unsqueeze(0)
                cached_projected = encoder.project_backbone_features(cached)
                self.assertTrue(torch.allclose(raw_projected, cached_projected, atol=1e-6))
                encoder.eval()
                self.assertTrue(torch.allclose(
                    encoder.aggregate_features(raw_projected),
                    encoder.aggregate_features(cached_projected), atol=1e-6))
                encoder.zero_grad(set_to_none=True)
                encoder.aggregate_features(cached_projected).sum().backward()
                self.assertIsNotNone(encoder.projection.weight.grad)
                self.assertIsNotNone(encoder.temporal.layers[0].linear1.weight.grad)
                # CachedWindowDataset needs no decoder for a sample.
                dataset = type("DatasetStub", (), {"window_size": n,
                    "source": source,
                    "_locate": lambda self, _i: (ep, 0), "__len__": lambda self: 1})()
                sample = CachedWindowDataset(dataset, cache)[0]
                self.assertEqual(sample["backbone_features"].shape, (n, 4))
            with patch("timm.create_model", return_value=_TinyBackbone()):
                adapter = TaFAdapter(window_size=5,
                    force_encoder={"hidden_dim": 8, "pressure_latent_dim": 2,
                                   "force_latent_dim": 2, "mlp_layers": 1},
                    vq={"pressure_codebook_size": 4, "force_codebook_size": 4},
                    tactile_encoder={"transformer_dim": 4, "transformer_layers": 1,
                                     "num_heads": 2, "dropout": 0.0, "max_window_size": 20})
            adapter.tactile_encoder = encoder
            adapter.eval()
            raw = torch.from_numpy(frames[:5]).permute(0, 3, 1, 2).unsqueeze(0)
            force = torch.zeros(1, 5, 6)
            pressure = torch.zeros(1, 5, 12, 12)
            with torch.no_grad():
                direct = adapter(raw, force, pressure)
                from_cache = adapter.forward_from_backbone_features(
                    cache5.features(ep, np.arange(5)).unsqueeze(0), force, pressure)
            for key in ("z_tactile", "loss_total", "loss_nce", "positive_cosine"):
                self.assertTrue(torch.allclose(direct[key], from_cache[key], atol=1e-6), key)
            stat = video.stat()
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            FrozenFrameCache(source, model, Path(temporary) / "cache",
                             device=torch.device("cpu"), amp_enabled=False,
                             batch_size=4)
            self.assertGreater(len(calls), first_count)

    def test_metric_accumulator_preserves_all_means(self):
        accumulator = DeviceMetricAccumulator()
        for number in (1.0, 3.0):
            accumulator.add({key: torch.tensor(number) for key in METRIC_KEYS})
        result = accumulator.pop_means("train")
        self.assertEqual(set(result), {f"train/{key}" for key in METRIC_KEYS})
        self.assertTrue(all(value == 2.0 for value in result.values()))
        self.assertEqual(accumulator.pop_means("train"), {})


if __name__ == "__main__":
    unittest.main()
