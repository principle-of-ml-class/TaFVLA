"""Benchmark orchestration checks without scanning data or using a GPU."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from script import profile_throughput


class ProfileThroughputTests(unittest.TestCase):
    def test_sensor_build_estimate_only_never_builds_or_initializes_model(self):
        from tests.test_pipeline import tiny_source
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = tiny_source(root)
            source.root_dir = root / 'taf_dataset'
            output = root / 'estimate'
            cache_dir = root / 'cache'
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch('sys.argv', ['profile_throughput','--data-root','unused',
                    '--manifest','unused.parquet','--output-dir',str(output),'--sensor-mmap-build-only',
                    '--estimate-only','--sensor-mmap-dir',str(cache_dir)]))
                stack.enter_context(patch('torch.cuda.is_available',return_value=False))
                stack.enter_context(patch.object(profile_throughput,'TaFDataset',return_value=source))
                stack.enter_context(patch.object(profile_throughput.SensorMmapCache,'prepare',
                                                 side_effect=AssertionError('cache build ran')))
                stack.enter_context(patch.object(profile_throughput,'build_model',
                                                 side_effect=AssertionError('model initialized')))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                profile_throughput.main()
            record = json.loads((output/'metrics.jsonl').read_text())
            self.assertEqual(record['record_type'],'sensor_mmap_estimate')
            self.assertEqual(record['payload_bytes'],8*608)
            self.assertFalse(cache_dir.exists())

    def test_sensor_cases_isolate_backend_and_replay_identical_optimizer_updates(self):
        from model.taf_adapter import TaFAdapter
        from utils.config import DataConfig, PerformanceConfig
        from tests.test_pipeline import tiny_source, _BenchmarkSamples, _BenchmarkBackbone
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = tiny_source(root)
            source.root_dir = root / 'taf_dataset'
            images = np.arange(8*8*8*3, dtype=np.uint8).reshape(8,8,8,3)
            source._images = lambda ep, offsets: (images[ep.video_frame_indices[offsets]],
                                                 ep.video_frame_indices[offsets].copy())
            dataset = _BenchmarkSamples(source)
            dataset.manifest_id = 'synthetic'
            with patch('timm.create_model', return_value=_BenchmarkBackbone()):
                model = TaFAdapter(window_size=2,
                    force_encoder={'hidden_dim':8,'pressure_latent_dim':2,'force_latent_dim':2,'mlp_layers':1},
                    vq={'pressure_codebook_size':4,'force_codebook_size':4},
                    tactile_encoder={'transformer_dim':4,'transformer_layers':1,'num_heads':2,
                                     'dropout':0.,'max_window_size':20})
            initial = {key:value.detach().clone() for key,value in model.state_dict().items()}
            cfg = SimpleNamespace(data=DataConfig(sensor_mmap_batch_rows=3),
                training=SimpleNamespace(amp=False,learning_rate=1e-4,weight_decay=1e-5,grad_clip_norm=1.),
                trainctl=SimpleNamespace(performance=PerformanceConfig(detailed_profiling=True,monitor_cache=True)))
            args = SimpleNamespace(sensor_mmap='compare',sensor_mmap_dir=None,sensor_mmap_rebuild=False,
                output_dir=root/'output',workers=[0],seed=42,video_cache_entries=2,group_reads=True,
                warmup_steps=1,steps=1,batch_size=256,prefetch_factor=1,check_samples=4,sensor_cache_mb=256)
            plan = [list(range(256)),list(range(256,512))]
            expected = hashlib.sha256()
            expected.update(json.dumps([f'sample-{i}' for i in plan[1]],separators=(',',':')).encode())
            frames = []
            for i in plan[1]:
                ep,start = dataset._locate(i)
                frames.append(ep.frames[start:start+2])
            expected.update(np.stack(frames).tobytes())
            snapshots, emitted = [], []
            real_run = profile_throughput._run
            def run(*params):
                row = real_run(*params)
                snapshots.append({key:value.detach().clone() for key,value in model.state_dict().items()})
                return row
            with patch.object(profile_throughput,'_run',side_effect=run):
                rows = profile_throughput._sensor_benchmark(model,dataset,source,list(range(512)),
                    plan,initial,cfg,args,torch.device('cpu'),expected.hexdigest(),'fixed-plan',emitted.append)
            self.assertEqual([r['sensor_case'] for r in rows],['parquet','mmap_cold','mmap_warm'])
            for key in snapshots[0]:
                self.assertTrue(torch.equal(snapshots[0][key],snapshots[1][key]),key)
                self.assertTrue(torch.equal(snapshots[0][key],snapshots[2][key]),key)
            self.assertEqual(rows[1]['sensor_mmap_preparation']['stats']['builds'],1)
            self.assertEqual(rows[2]['sensor_mmap_preparation']['stats']['builds'],0)
            for row in rows[1:]:
                self.assertEqual(row['perf/sensor_mmap_parquet_read_calls'],0)
                self.assertEqual(row['perf/sensor_mmap_conversion_calls'],0)
                self.assertEqual(row['perf/sensor_mmap_hit_rate'],1)
                self.assertTrue(row['sensor_correctness']['sensor_tensors_bitwise_equal'])
            summary = next(r for r in emitted if r['record_type']=='sensor_mmap_comparison')
            self.assertTrue(summary['sample_frame_order_equal'])
            self.assertIsNotNone(rows[2]['cpu_peak_sampled_pss_bytes'])
            self.assertIsNone(source._sensor_mmap)

    def test_plan_duration_is_independent_of_frame_offsets(self):
        episode = SimpleNamespace(frames=np.arange(1024, dtype=np.int64))
        dataset = SimpleNamespace(
            manifest_id="synthetic",
            rows=[{"split": "train", "sequence_id": "seq", "sample_id": f"sample-{i}"}
                  for i in range(512)],
            _locate=lambda index: (episode, 17 + index),
        )
        plan = [list(range(256)), list(range(256, 512))]
        sampler = MagicMock()
        sampler.__iter__.return_value = iter(plan)
        sampler.report.as_dict.return_value = {"batch_count": 2}
        model = MagicMock()
        model.to.return_value = model
        model.tactile_encoder.backbone_dim = 4
        model.state_dict.return_value = {"weight": torch.zeros(1)}
        consumed = hashlib.sha256()
        consumed.update(json.dumps([f"sample-{i}" for i in range(256, 512)],
                                   separators=(",", ":")).encode())
        consumed.update(np.arange(273, 529, dtype=np.int64).reshape(256, 1).tobytes())
        row = {"N": 1, "workers": 0, "mode": "baseline", "cache": False,
               "sample_frame_order_sha256": consumed.hexdigest()}
        normalization = {"manifest_id": "synthetic", "train_groups": ["seq"], "normalization": {}}
        estimate = {"unique_frames": 1024, "feature_bytes": 16384, "videos": 1}
        # Main's source, plan and model phases take 1, 3 and 2 seconds.
        clock = SimpleNamespace(perf_counter=MagicMock(side_effect=[100, 101, 200, 203, 300, 302]))
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            output = Path(temporary) / "run"
            stack.enter_context(patch("sys.argv", [
                "profile_throughput", "--data-root", "unused", "--manifest", "unused.parquet",
                "--output-dir", str(output), "--mode", "baseline", "--workers", "0",
                "--steps", "1", "--warmup-steps", "1", "--feature-dim", "4",
            ]))
            stack.enter_context(patch.object(profile_throughput, "time", clock))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch.object(profile_throughput, "TaFDataset"))
            stack.enter_context(patch.object(profile_throughput.FrozenFrameCache, "estimate", return_value=estimate))
            stack.enter_context(patch.object(profile_throughput, "ManifestDataset", return_value=dataset))
            stack.enter_context(patch.object(profile_throughput, "FixedValidationBatchSampler", return_value=sampler))
            stack.enter_context(patch("torch.load", return_value=normalization))
            stack.enter_context(patch.object(profile_throughput, "build_model", return_value=model))
            stack.enter_context(patch.object(profile_throughput, "_check_raw", return_value={"checked_samples": 4}))
            stack.enter_context(patch.object(profile_throughput, "_run", return_value=row))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            profile_throughput.main()
            records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        phases = {record["record_type"]: record for record in records}
        self.assertEqual(phases["initialization"]["source_seconds"], 1)
        self.assertEqual(phases["plan"]["plan_initialization_seconds"], 3)
        self.assertEqual(phases["model_initialization"]["seconds"], 2)


if __name__ == "__main__":
    unittest.main()
