"""Temporary sensor files only: identity, corruption, atomicity and worker ownership."""
from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing as mp
import os
import random
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import _column_matrix
from utils.pipeline import PipelineDataset, PipelineCollate
from utils.sensor_cache import SensorMmapCache, SensorCacheCorruptionError
from tests.test_pipeline import tiny_source, _sandbox_tensor_transport


def _write(path, rows=13, value=0):
    index = np.arange(500, 500+rows, dtype=np.int64)
    force = (np.arange(rows*6, dtype=np.float32).reshape(rows,6) / 7 + value)
    pressure = (np.arange(rows*144, dtype=np.float32).reshape(rows,12,12) / 13 + value)
    pq.write_table(pa.table({'index': index, 'observation.force_torque': force.tolist(),
                            'observation.pressure_matrix': pressure.tolist()}), path, row_group_size=11)
    return index, force, pressure


def _race(cache, source, barrier, connection):
    try:
        barrier.wait(timeout=10)
        force, pressure = cache.rows(source, 3, np.array([0, 2], dtype=np.int64))
        connection.send(('ok', force, pressure, cache.get(source).arrays['force'].flags.writeable))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        cache.close()
        connection.close()


class SensorCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'sensors.parquet'
        self.index, self.force, self.pressure = _write(self.source)
        self.cache = SensorMmapCache(self.root / 'cache', batch_rows=5, max_open_files=2)

    def tearDown(self):
        self.cache.close()
        self.temporary.cleanup()

    def test_chunked_build_preserves_all_bytes_and_episode_mapping(self):
        self.index, self.force, self.pressure = _write(self.source, rows=2057)
        checksum = hashlib.sha256(self.source.read_bytes()).hexdigest()
        py_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        np_rng = np.random.get_state()
        entry = self.cache.get(self.source)
        np.testing.assert_array_equal(entry.arrays['index'], self.index)
        self.assertEqual(entry.arrays['force'].tobytes(), self.force.tobytes())
        self.assertEqual(entry.arrays['pressure'].tobytes(), self.pressure.reshape(-1,144).tobytes())
        for array in entry.arrays.values():
            self.assertFalse(array.flags.writeable)
        # A later episode starts in the middle of the same physical Parquet.
        offsets = np.array([0, 1, 1024], dtype=np.int64)
        force, pressure = self.cache.rows(self.source, 1011, offsets)
        self.assertEqual(force.tobytes(), self.force[1011+offsets].tobytes())
        self.assertEqual(pressure.tobytes(), self.pressure[1011+offsets].tobytes())
        self.assertTrue(force.flags.writeable)
        self.assertTrue(pressure.flags.writeable)
        for n in (1, 3, 5, 7, 10, 20):
            force, pressure = self.cache.rows(self.source, 1011, np.arange(n, dtype=np.int64))
            self.assertEqual(force.tobytes(), self.force[1011:1011+n].tobytes())
            self.assertEqual(pressure.tobytes(), self.pressure[1011:1011+n].tobytes())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), checksum)
        self.assertEqual(random.getstate(), py_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        actual_np_rng = np.random.get_state()
        self.assertEqual(actual_np_rng[0], np_rng[0])
        np.testing.assert_array_equal(actual_np_rng[1], np_rng[1])
        self.assertEqual(actual_np_rng[2:], np_rng[2:])
        self.assertGreater(self.cache.stats()['conversion_calls'], 1)
        expected = SensorMmapCache.estimate([self.source, self.source])
        self.assertEqual(expected['payload_bytes'], 2057 * 608)

    def test_warm_reuse_never_opens_parquet(self):
        self.cache.prepare([self.source])
        warm = SensorMmapCache(self.root / 'cache')
        try:
            with patch('pyarrow.parquet.ParquetFile', side_effect=AssertionError('warm path decoded Parquet')):
                for _ in range(2):
                    force, pressure = warm.rows(self.source, 4, np.array([0,1]))
                    self.assertEqual(force.tobytes(), self.force[[4,5]].tobytes())
                    self.assertEqual(pressure.tobytes(), self.pressure[[4,5]].tobytes())
            self.assertEqual(warm.stats()['builds'], 0)
            self.assertEqual(warm.stats()['parquet_read_calls'], 0)
            self.assertEqual(warm.stats()['cache_hits'], 2)
        finally:
            warm.close()

    def test_source_change_invalidates_only_its_identity_and_preserves_old_files(self):
        self.cache.rows(self.source, 0, np.array([0]))
        old = self.cache.get(self.source).directory
        other_source = self.root / 'other.parquet'
        _write(other_source)
        other = self.cache.get(other_source).directory
        old_stat = self.source.stat()
        _, new_force, _ = _write(self.source, value=100)
        # Even restoring mtime cannot hide a normal file change (ctime/inode are included).
        os.utime(self.source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        force, _ = self.cache.rows(self.source, 0, np.array([0]))
        self.assertEqual(force.tobytes(), new_force[:1].tobytes())
        self.assertNotEqual(self.cache.get(self.source).directory, old)
        self.assertTrue((old / 'force.npy').is_file())
        self.assertEqual(self.cache.get(other_source).directory, other)

    def test_missing_and_truncated_arrays_are_rebuilt(self):
        for corruption in ('missing', 'truncated'):
            with self.subTest(corruption=corruption):
                entry = self.cache.get(self.source)
                path = entry.directory / 'force.npy'
                self.cache.close()
                if corruption == 'missing':
                    path.unlink()
                else:
                    path.write_bytes(b'partial')
                with self.assertWarnsRegex(RuntimeWarning, 'Rebuilding invalid sensor cache'):
                    force, _ = self.cache.rows(self.source, 0, np.array([0]))
                self.assertEqual(force.tobytes(), self.force[:1].tobytes())

    def test_valid_header_payload_corruption_raises_and_explicit_rebuild_repairs(self):
        entry = self.cache.get(self.source)
        array = entry.arrays['pressure']
        path, offset = entry.directory / 'pressure.npy', array.offset
        self.cache.close()
        with path.open('r+b') as file:
            file.seek(offset)
            value = file.read(1)
            file.seek(offset)
            file.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(SensorCacheCorruptionError, 'checksum mismatch'):
            self.cache.rows(self.source, 0, np.array([0]))
        self.assertEqual(len(self.cache._entries), 0)
        rebuilt = SensorMmapCache(self.root / 'cache', rebuild=True)
        try:
            _, pressure = rebuilt.rows(self.source, 0, np.array([0]))
            self.assertEqual(pressure.tobytes(), self.pressure[:1].tobytes())
        finally:
            rebuilt.close()

    def test_disk_full_at_publication_preserves_old_generation_and_pointer(self):
        old = self.cache.get(self.source)
        parent, pointer, _ = self.cache._paths(self.cache.identity(self.source))
        publication = pointer.read_bytes()
        old_dir = old.directory
        rebuilt = SensorMmapCache(self.root / 'cache', rebuild=True)
        try:
            with patch('utils.sensor_cache.os.replace', side_effect=OSError(errno.ENOSPC, 'disk full')):
                with self.assertRaises(OSError):
                    rebuilt.rows(self.source, 0, np.array([0]))
            self.assertEqual(pointer.read_bytes(), publication)
            self.assertTrue((old_dir / 'pressure.npy').is_file())
            self.assertFalse(list(parent.glob('.build-*')))
            self.assertFalse(list(parent.glob('.pointer-*')))
            force, _ = self.cache.rows(self.source, 0, np.array([0]))
            self.assertEqual(force.tobytes(), self.force[:1].tobytes())
        finally:
            rebuilt.close()

    def test_interrupted_first_build_never_publishes_partial_arrays(self):
        with patch('utils.sensor_cache.os.fsync', side_effect=OSError('interrupted write')):
            with self.assertRaises(OSError):
                self.cache.rows(self.source, 0, np.array([0]))
        parent, pointer, _ = self.cache._paths(self.cache.identity(self.source))
        self.assertFalse(pointer.exists())
        self.assertFalse(list(parent.glob('.build-*')))
        force, _ = self.cache.rows(self.source, 0, np.array([0]))
        self.assertEqual(force.tobytes(), self.force[:1].tobytes())

    def test_source_change_during_conversion_aborts_publication(self):
        changed = False
        def convert(column, name):
            nonlocal changed
            value = _column_matrix(column, name)
            if not changed:
                stat = self.source.stat()
                os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
                changed = True
            return value
        with patch('dataset.taf_dataset._column_matrix', side_effect=convert):
            with self.assertRaisesRegex(RuntimeError, 'Source changed while building'):
                self.cache.rows(self.source, 0, np.array([0]))
        self.assertFalse(list((self.root / 'cache').glob('*/*.json')))
        force, _ = self.cache.rows(self.source, 0, np.array([0]))
        self.assertEqual(force.tobytes(), self.force[:1].tobytes())

    def test_cache_inside_original_dataset_is_rejected_before_writing(self):
        from dataset.taf_dataset import TaFDataset
        original = self.root / 'taf_dataset'
        cache_dir = original / 'sensor_cache'
        with self.assertRaisesRegex(ValueError, 'outside the source'):
            TaFDataset(original, timestamp_cache_enabled=False, sensor_mmap_enabled=True,
                       sensor_mmap_dir=cache_dir)
        self.assertFalse(cache_dir.exists())

    def test_mapping_eviction_closes_handles_and_returned_tensors_remain_valid(self):
        self.cache.rows(self.source, 0, np.array([0]))
        self.cache.close()
        descriptor_count = len(list(Path('/proc/self/fd').iterdir()))
        retained = []
        old_arrays = []
        for number in range(7):
            path = self.root / f'source{number}.parquet'
            _write(path, value=number)
            force, _ = self.cache.rows(path, 0, np.array([0]))
            retained.append(torch.from_numpy(force))
            old_arrays.append(self.cache.get(path).arrays['force'])
            self.assertLessEqual(len(self.cache._entries), 2)
        self.assertTrue(all(a._mmap.closed for a in old_arrays[:-2]))
        self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())), descriptor_count + 8)
        for number, tensor in enumerate(retained):
            self.assertTrue(torch.equal(tensor, torch.from_numpy(self.force[:1] + number)))
        self.cache.close()
        self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())), descriptor_count)

    def test_two_spawned_workers_publish_one_generation(self):
        cache = SensorMmapCache(self.root / 'shared', batch_rows=4, shared_stats=True, rebuild=True)
        context = mp.get_context('spawn')
        barrier = context.Barrier(3)
        processes, readers = [], []
        try:
            for _ in range(2):
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=_race, args=(cache, self.source, barrier, child))
                process.start()
                child.close()
                processes.append(process)
                readers.append(parent)
            barrier.wait(timeout=10)
            for reader in readers:
                self.assertTrue(reader.poll(10), 'spawned cache reader timed out')
                result = reader.recv()
                self.assertEqual(result[0], 'ok', str(result))
                self.assertEqual(result[1].tobytes(), self.force[[3,5]].tobytes())
                self.assertFalse(result[3])
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(cache.stats()['builds'], 1)
            self.assertEqual(len(list((self.root / 'shared').glob('*/gen-*'))), 1)
        finally:
            for reader in readers:
                reader.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
            cache.close()

    def test_two_persistent_dataloader_workers_match_raw_av1_and_metadata(self):
        source = tiny_source(self.root)
        plan = [[4,0], [2,3], [1,5]]
        reference = [source[i] for batch in plan for i in batch]
        source._sensor_mmap = SensorMmapCache(self.root / 'loader-cache', batch_rows=3, shared_stats=True)
        wrapped = PipelineDataset(source, source, group_reads=True)
        loader = DataLoader(Subset(wrapped, list(range(6))), batch_sampler=plan,
            num_workers=2, multiprocessing_context='spawn', persistent_workers=True,
            prefetch_factor=1, collate_fn=PipelineCollate(source,detailed=True),
            worker_init_fn=_sandbox_tensor_transport, timeout=10)
        iterator = None
        try:
            for _ in range(2):
                iterator = iter(loader)
                actual = list(iterator)
                for j, batch in enumerate(actual):
                    samples = reference[2*j:2*j+2]
                    for key in ('tactile','force','pressure'):
                        self.assertTrue(torch.equal(batch[key],torch.stack([s[key] for s in samples])),key)
                    for key, value in samples[0]['metadata'].items():
                        if isinstance(value,torch.Tensor):
                            self.assertTrue(torch.equal(batch['metadata'][key],torch.stack([s['metadata'][key] for s in samples])),key)
                        else:
                            expected = [s['metadata'][key] for s in samples]
                            values = batch['metadata'][key]
                            self.assertEqual(values.tolist() if isinstance(values,torch.Tensor) else values,expected)
        finally:
            if iterator is not None:
                workers=list(iterator._workers)
                iterator._shutdown_workers()
                self.assertTrue(all(not worker.is_alive() for worker in workers))
            source._sensor_mmap.close()
            source._close_video()
        self.assertEqual(source._sensor_mmap.stats()['builds'],1)
        self.assertGreater(source._sensor_mmap.stats()['mapping_hits'],0)


if __name__ == '__main__':
    unittest.main()
