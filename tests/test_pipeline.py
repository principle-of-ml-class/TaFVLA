"""Bounded CPU-only checks for pipeline identity, decoder ownership and spawn."""
from __future__ import annotations

import multiprocessing as mp
import tempfile
import subprocess
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset, _column_matrix, _column_matrix_legacy
from utils.pipeline import PipelineTelemetry, PipelineDataset, PipelineCollate, ProfiledBatch


def _reduce_tensor_by_value(tensor):
    return torch.from_numpy, (tensor.numpy(),)


def _sandbox_tensor_transport(_worker_id):
    # This sandbox denies PyTorch's AF_UNIX FD-sharing socket. Only the test
    # uses by-value tensor transport; production sharing is unchanged.
    from multiprocessing.reduction import ForkingPickler
    ForkingPickler.register(torch.Tensor, _reduce_tensor_by_value)


def tiny_source(directory):
    root = Path(directory)
    video = root / 'tiny.mp4'
    pixels = np.stack([np.full((16, 16, 3), i * 28, np.uint8) for i in range(8)])
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error', '-f','rawvideo',
        '-pixel_format','rgb24','-video_size','16x16','-framerate','10','-i','pipe:0',
        '-c:v','libaom-av1','-cpu-used','8','-threads','1','-crf','40','-g','4',
        '-pix_fmt','yuv420p',str(video)], input=pixels.tobytes(),
        capture_output=True, check=True, timeout=10)
    parquet = root / 'tiny.parquet'
    pq.write_table(pa.table({'index': np.arange(8),
        'observation.force_torque': np.arange(48, dtype=np.float32).reshape(8,6).tolist(),
        'observation.pressure_matrix': np.arange(8*144, dtype=np.float32).reshape(8,12,12).tolist()}), parquet)
    ds = object.__new__(TaFDataset)
    ds.window_size, ds.stride, ds.image_size, ds.image_dtype = 2, 1, (8, 8), 'uint8'
    ds.sensor_cache_max_bytes = 8 * 608
    ds.sensor_arrow_conversion, ds.video_cache_max_entries = True, 2
    ds._sensor_cache, ds._sensor_cache_bytes = OrderedDict(), 0
    ds._sensor_stats = {'parquet_reads': 0, 'cache_hits': 0, 'cache_misses': 0}
    ds._sensor_shared_stats = mp.get_context('spawn').Array('q', [0]*6)
    ds._pipeline_shared_stats = mp.get_context('spawn').Array('d', [0.]*4)
    ds._telemetry = PipelineTelemetry(True)
    ds._video_pool, ds._video_finalizer = OrderedDict(), None
    ds._video = ds._video_stream = ds._video_path = ds._video_pid = None
    ds._video_timestamp_cache = {}
    # Two episodes share one physical video; requests must stay inside each one.
    ds.episodes = []
    for start in (0, 4):
        ds.episodes.append(SimpleNamespace(parquet=parquet, video=video, row_start=start,
            episode_index=start//4, length=4, frames=np.arange(4), sensor_type='test',
            sequence_id='test', timestamps=np.arange(4)/10,
            video_timestamps=np.arange(start,start+4)/10, video_frame_indices=np.arange(start,start+4)))
    ds.all_episodes, ds._ends = ds.episodes, [3, 6]
    return ds


class _BenchmarkSamples:
    window_size = 2
    def __init__(self, source):
        self.source = source
    def __len__(self):
        return 512
    def _locate(self, index):
        return self.source._locate(index % len(self.source))
    def __getitem__(self, index):
        sample = self.source[index % len(self.source)]
        sample["metadata"]["sample_id"] = f"sample-{index}"
        return sample


class _BenchmarkBackbone(torch.nn.Module):
    num_features = 4
    pretrained_cfg = {"input_size": (3, 2, 2), "mean": (0., 0., 0.), "std": (1., 1., 1.)}
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 4)
    def forward(self, value):
        return self.linear(value.flatten(1))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Match TaFDataset's spawn context before allocating telemetry.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

    def test_arrow_values_slices_chunks_and_ragged(self):
        for shape in ((7,6), (7,144), (7,12,12)):
            raw = np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            column = pa.chunked_array([pa.array(raw[:3].tolist()), pa.array(raw[3:].tolist())])
            for candidate in (column, column.slice(1,4)):
                self.assertTrue(np.array_equal(_column_matrix(candidate,'test'),
                                               _column_matrix_legacy(candidate,'test')))
        ragged = pa.chunked_array([pa.array([[1.], [2.,3.]])])
        with self.assertRaisesRegex(ValueError, 'Ragged'):
            _column_matrix(ragged,'test')

    def test_av1_random_seek_reuse_matches_close_each_time(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = tiny_source(directory)
            order = [0, 4, 2, 3, 1, 5, 0]
            ds.video_cache_max_entries, ds.sensor_arrow_conversion = 0, False
            reference = [ds[i] for i in order]
            ds._sensor_cache.clear(); ds._sensor_cache_bytes = 0
            ds.video_cache_max_entries, ds.sensor_arrow_conversion = 2, True
            before_fd = len(list(Path('/proc/self/fd').iterdir()))
            for i, expected in zip(order, reference, strict=True):
                actual = ds[i]
                for key in ('tactile','force','pressure'):
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)
                for key, value in expected['metadata'].items():
                    if isinstance(value,torch.Tensor):
                        self.assertTrue(torch.equal(value,actual['metadata'][key]),key)
                self.assertLessEqual(len(ds._video_pool),2)
            state = ds.__getstate__()
            self.assertIsNone(state['_video'])
            self.assertEqual(len(state['_video_pool']),0)
            ds._close_video()
            self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())),before_fd)
            stats=ds.pipeline_total_stats()
            self.assertGreater(stats['video_close_calls'],0)
            self.assertGreater(stats['column_matrix_seconds'],0)
            self.assertGreater(stats['video_cache_hits'],0)
            # An invalid PTS must raise and close all retained containers.
            original = ds.episodes[0].video_timestamps.copy()
            ds.episodes[0].video_timestamps = original + .05
            with self.assertRaises(ValueError):
                ds[0]
            self.assertEqual(len(ds._video_pool),0)

    def test_profiled_spawn_persistent_workers_with_by_value_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = tiny_source(directory)
            plan = [[4,0],[2,3],[1,5]]
            expected = [[ds[i] for i in batch] for batch in plan]
            wrapped=PipelineDataset(ds,ds,group_reads=True)
            loader=DataLoader(Subset(wrapped,list(range(6))),batch_sampler=plan,
                num_workers=1,multiprocessing_context='spawn',persistent_workers=True,
                prefetch_factor=1,collate_fn=PipelineCollate(ds,detailed=True),
                worker_init_fn=_sandbox_tensor_transport,timeout=10)
            iterator=None
            try:
                for _ in range(2):
                    iterator=iter(loader)
                    for batch, samples in zip(iterator,expected,strict=True):
                        self.assertIsInstance(batch,ProfiledBatch)
                        for key in ('tactile','force','pressure'):
                            self.assertTrue(torch.equal(batch[key],torch.stack([s[key] for s in samples])))
                        self.assertTrue(torch.equal(batch['metadata']['frame_indices'],
                                                    torch.stack([s['metadata']['frame_indices'] for s in samples])))
            finally:
                if iterator is not None:
                    workers=list(iterator._workers)
                    iterator._shutdown_workers()
                    self.assertTrue(all(not w.is_alive() for w in workers))
                ds._close_video()
            stats=ds.pipeline_total_stats()
            self.assertEqual(stats['batch_prepare_calls'],6)
            self.assertEqual(stats['collate_calls'],6)
            self.assertEqual(stats['sample_prepare_calls'],12)

    def test_benchmark_raw_modes_replay_same_256_sample_batches_and_update(self):
        from model.taf_adapter import TaFAdapter
        from script.profile_throughput import _run
        from utils.config import PerformanceConfig
        with tempfile.TemporaryDirectory() as directory:
            source = tiny_source(directory)
            images = np.arange(8*8*8*3, dtype=np.uint8).reshape(8,8,8,3)
            source._images = lambda ep, offsets: (images[ep.video_frame_indices[offsets]],
                                                 ep.video_frame_indices[offsets].copy())
            dataset = _BenchmarkSamples(source)
            with patch("timm.create_model", return_value=_BenchmarkBackbone()):
                model = TaFAdapter(window_size=2,
                    force_encoder={"hidden_dim": 8, "pressure_latent_dim": 2,
                                   "force_latent_dim": 2, "mlp_layers": 1},
                    vq={"pressure_codebook_size": 4, "force_codebook_size": 4},
                    tactile_encoder={"transformer_dim": 4, "transformer_layers": 1,
                                     "num_heads": 2, "dropout": 0., "max_window_size": 20})
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            cfg = SimpleNamespace(training=SimpleNamespace(amp=False, learning_rate=1e-4,
                weight_decay=1e-5, grad_clip_norm=1.),
                trainctl=SimpleNamespace(performance=PerformanceConfig(detailed_profiling=True,
                                                                       monitor_cache=True)))
            args = SimpleNamespace(seed=42, video_cache_entries=2, group_reads=True,
                                   steps=1, warmup_steps=1, batch_size=256, prefetch_factor=1)
            plan = [list(range(256)), list(range(256,512))]
            rows, states = [], []
            for mode in ("baseline", "optimized"):
                rows.append(_run(model,dataset,source,list(range(512)),plan,initial,
                                 cfg,args,torch.device("cpu"),mode,0))
                states.append({key: value.detach().clone() for key,value in model.state_dict().items()})
            self.assertEqual(rows[0]["sample_frame_order_sha256"],rows[1]["sample_frame_order_sha256"])
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key],states[1][key]),key)
            self.assertEqual(rows[0]["batch_size"],256)
            self.assertGreater(rows[1]["perf/worker_sample_prepare_calls"],0)

    def test_profiled_pin_memory_hook_and_disabled_telemetry(self):
        batch=ProfiledBatch({'force':torch.zeros(2,1,6)})
        with patch('utils.pipeline.pin_memory',side_effect=lambda x:x) as pin:
            self.assertIs(batch.pin_memory(),batch)
            pin.assert_called_once()
        disabled=PipelineTelemetry(False)
        with patch('utils.pipeline.time.perf_counter',side_effect=AssertionError('timer ran')):
            with disabled.stage('collate'):
                pass
        self.assertEqual(disabled.snapshot(),{})


if __name__ == '__main__':
    unittest.main()
