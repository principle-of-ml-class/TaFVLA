"""Opt-in worker telemetry and batch access without changing sample order or RNG."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
from collections import defaultdict

from torch.utils.data import Dataset, default_collate
from torch.utils.data._utils.pin_memory import pin_memory

STAGES = (
    "parquet_open", "parquet_read", "column_matrix", "sensor_lookup", "sensor_eviction",
    "frame_index_search", "sensor_gather", "sensor_tensor", "video_open", "video_seek", "av1_decode",
    "video_rgb", "image_tensor", "video_close", "sample_prepare", "batch_prepare",
    "collate", "feature_lookup", "video_decode",
)
COUNTERS = ("sensor_evictions", "sensor_oversize", "sensor_loaded_bytes", "decoded_frames",
            "video_cache_hits", "video_cache_misses", "video_evictions")


class PipelineTelemetry:
    """Worker elapsed times overlap. Publish once per batch, not once per frame."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.shared = mp.Array("d", 2 * len(STAGES) + len(COUNTERS)) if enabled else None
        self.local = defaultdict(float)
        # 128 process slots cover train/validation workers without a Manager server.
        self.worker_stats = mp.Array("d", 128 * 9) if enabled else None
        self._worker_slot = None

    def stage(self, name):
        return self._stage(name) if self.enabled else contextlib.nullcontext()

    @contextlib.contextmanager
    def _stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.local[name + "_seconds"] += time.perf_counter() - start
            self.local[name + "_calls"] += 1

    def count(self, name, value=1):
        if self.enabled:
            self.local[name] += value

    @staticmethod
    def keys():
        return [key for stage in STAGES for key in (stage + "_seconds", stage + "_calls")] + list(COUNTERS)

    def flush(self, *, cache_bytes=None, cache_entries=None):
        if self.shared is not None and self.local:
            with self.shared.get_lock():
                for i, key in enumerate(self.keys()):
                    self.shared[i] += self.local.get(key, 0)
            with self.worker_stats.get_lock():
                if self._worker_slot is None:
                    for slot in range(128):
                        pid = int(self.worker_stats[slot * 9])
                        if pid in (0, os.getpid()) or not os.path.exists(f"/proc/{pid}"):
                            self._worker_slot = slot * 9
                            for offset in range(9):
                                self.worker_stats[self._worker_slot + offset] = 0
                            self.worker_stats[self._worker_slot] = os.getpid()
                            break
                    if self._worker_slot is None:
                        raise RuntimeError("Pipeline telemetry exceeded 128 process slots")
                start = self._worker_slot
                for i, key in enumerate(("sample_prepare_seconds", "sample_prepare_calls",
                                         "batch_prepare_seconds", "batch_prepare_calls",
                                         "collate_seconds", "collate_calls"), 1):
                    self.worker_stats[start + i] += self.local.get(key, 0)
                if cache_bytes is not None:
                    self.worker_stats[start + 7] = cache_bytes
                    self.worker_stats[start + 8] = cache_entries
            self.local.clear()

    def snapshot(self):
        self.flush()
        if self.shared is None:
            return {}
        with self.shared.get_lock():
            result = dict(zip(self.keys(), self.shared[:]))
        with self.worker_stats.get_lock():
            for start in range(0, 128 * 9, 9):
                pid = int(self.worker_stats[start])
                if not pid:
                    continue
                names = ("sample_prepare_seconds", "sample_prepare_calls", "batch_prepare_seconds",
                         "batch_prepare_calls", "collate_seconds", "collate_calls",
                         "cache_resident_bytes", "cache_resident_entries")
                for i, name in enumerate(names, 1):
                    value = self.worker_stats[start + i]
                    if "_resident_" in name and not os.path.exists(f"/proc/{pid}"):
                        value = 0
                    result[f"process_{pid}_{name}"] = value
        return result

    def __getstate__(self):
        return {**self.__dict__, "local": defaultdict(float), "_worker_slot": None}


class ProfiledBatch:
    """Custom pin hook measures pinning, with no global PyTorch monkey patch."""
    def __init__(self, data):
        self.data = data
        self.collated_at = time.monotonic()
        self.pin_seconds = 0.0

    def pin_memory(self):
        start = time.perf_counter()
        self.data = pin_memory(self.data)
        self.pin_seconds = time.perf_counter() - start
        return self

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def items(self):
        return self.data.items()


class PipelineDataset(Dataset):
    def __init__(self, dataset, source, *, group_reads=False):
        self.dataset, self.source = dataset, source
        self.group_reads = group_reads

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with self.source._telemetry.stage("sample_prepare"):
            return self.dataset[index]

    def __getitems__(self, indices):
        order = list(range(len(indices)))
        if self.group_reads:
            # Subset calls us with global indices. Keep the sampler's output order.
            locations = [self.dataset._locate(i)[0] for i in indices]
            order.sort(key=lambda j: (str(locations[j].parquet), str(locations[j].video)))
        result = [None] * len(indices)
        with self.source._telemetry.stage("batch_prepare"):
            for j in order:
                result[j] = self[indices[j]]
        return result


class PipelineCollate:
    def __init__(self, source, *, detailed=False):
        self.source, self.detailed = source, detailed

    def __call__(self, samples):
        with self.source._telemetry.stage("collate"):
            batch = default_collate(samples)
        self.source._telemetry.flush(cache_bytes=self.source._sensor_cache_bytes,
                                      cache_entries=len(self.source._sensor_cache))
        return ProfiledBatch(batch) if self.detailed else batch
