"""Persistent, immutable generations of decompressed sensor arrays.

Writes stream bounded Arrow batches to NPY files. Readers map three arrays read
only and verify checksums for accessed row blocks, not the entire file per open.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import resource
import shutil
import tempfile
import time
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_VERSION = 1
_CONVERSION = "taf-sensor-float32-arrow-v1"
_CHUNK_ROWS = 1024
_ARRAYS = {"index": (np.dtype("int64"), ()), "force": (np.dtype("float32"), (6,)),
           "pressure": (np.dtype("float32"), (144,))}
_COLUMNS = ["index", "observation.force_torque", "observation.pressure_matrix"]
_COUNTERS = ("cache_hits", "cache_misses", "mapping_hits", "mapping_opens", "mapping_evictions",
             "builds", "repairs", "parquet_open_calls", "parquet_open_seconds", "parquet_read_calls",
             "parquet_read_seconds", "conversion_calls", "conversion_seconds", "build_seconds",
             "load_calls", "load_seconds", "access_calls", "access_seconds", "verify_seconds",
             "lock_wait_seconds", "bytes_written", "minor_page_faults", "major_page_faults")


class SensorCacheCorruptionError(RuntimeError):
    """A mapped row block differs from the published generation's checksum."""


@dataclass
class _Entry:
    arrays: dict
    metadata: dict
    directory: Path
    signatures: dict
    verified: dict = field(default_factory=lambda: {key: set() for key in _ARRAYS})

    def unchanged(self):
        try:
            return all(_signature(self.directory / (key + ".npy")) == value
                       for key, value in self.signatures.items())
        except OSError:
            return False

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


def _close_entries(entries):
    for entry in entries.values():
        entry.close()
    entries.clear()


def _signature(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SensorMmapCache:
    def __init__(self, directory, *, max_open_files=16, batch_rows=8192, rebuild=False,
                 shared_stats=False, stats_enabled=True, profile_page_faults=False):
        if max_open_files < 1 or batch_rows < 1:
            raise ValueError("Sensor mmap max_open_files and batch_rows must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.max_open_files, self.batch_rows = int(max_open_files), int(batch_rows)
        self.rebuild = bool(rebuild)
        self.rebuild_token = uuid.uuid4().hex if rebuild else None
        self.stats_enabled = stats_enabled
        self.profile_page_faults = profile_page_faults
        self._shared = mp.get_context("spawn").Array("d", len(_COUNTERS)) if shared_stats and stats_enabled else None
        self._stats = dict.fromkeys(_COUNTERS, 0.0)
        self._pending = dict.fromkeys(_COUNTERS, 0.0)
        self._entries = OrderedDict()
        self._rebuilt = set()
        self._finalizer = None
        self._pid = os.getpid()

    def _add(self, key, value=1):
        if self.stats_enabled:
            self._stats[key] += value
            self._pending[key] += value

    @contextlib.contextmanager
    def _timer(self, key, calls=None):
        if not self.stats_enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._add(key, time.perf_counter() - started)
            if calls:
                self._add(calls)

    def _flush(self):
        if self._shared is not None:
            with self._shared.get_lock():
                for i, key in enumerate(_COUNTERS):
                    self._shared[i] += self._pending[key]
            self._pending = dict.fromkeys(_COUNTERS, 0.0)

    def stats(self):
        if self._shared is None:
            return dict(self._stats)
        self._flush()
        with self._shared.get_lock():
            return dict(zip(_COUNTERS, self._shared[:]))

    @staticmethod
    def identity(source):
        source = Path(source).resolve(strict=True)
        stat = source.stat()
        return {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev,
                "format_version": _VERSION, "conversion": _CONVERSION}

    def _paths(self, identity):
        source_key = hashlib.sha256(identity["source"].encode()).hexdigest()
        identity_key = hashlib.sha256(_json_bytes(identity)).hexdigest()
        parent = self.directory / source_key
        return parent, parent / (identity_key + ".json"), identity_key

    def _load(self, pointer, identity):
        """Validate the publication, metadata and all NPY headers before mapping."""
        arrays = {}
        with self._timer("load_seconds", "load_calls"):
            try:
                publication = json.loads(pointer.read_bytes())
                generation = publication["generation"]
                if not re.fullmatch(r"gen-[0-9a-f]{32}", generation):
                    raise ValueError("invalid generation name")
                directory = pointer.parent / generation
                contents = (directory / "metadata.json").read_bytes()
                if hashlib.sha256(contents).hexdigest() != publication["metadata_sha256"]:
                    raise ValueError("metadata checksum mismatch")
                metadata = json.loads(contents)
                if metadata["identity"] != identity or metadata["chunk_rows"] != _CHUNK_ROWS:
                    raise ValueError("source identity/cache version mismatch")
                rows = metadata["rows"]
                if not isinstance(rows, int) or rows < 1:
                    raise ValueError("invalid row count")
                if not re.fullmatch(r"[0-9a-f]{64}", metadata["schema_sha256"]):
                    raise ValueError("invalid source schema fingerprint")
                signatures = {}
                for name, (dtype, tail) in _ARRAYS.items():
                    description = metadata["arrays"][name]
                    shape = (rows, *tail)
                    if description["dtype"] != dtype.str or description["shape"] != list(shape):
                        raise ValueError(f"invalid {name} schema")
                    hashes = description["sha256_blocks"]
                    if (len(hashes) != math.ceil(rows / _CHUNK_ROWS)
                            or not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)):
                        raise ValueError(f"invalid {name} block checksums")
                    path = directory / (name + ".npy")
                    signatures[name] = _signature(path)
                    if signatures[name][0] != description["file_bytes"]:
                        raise ValueError(f"truncated/extended {name} array")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if not isinstance(array, np.memmap):
                        if hasattr(array, "close"):
                            array.close()
                        raise ValueError(f"{name} is not a memory-mappable NPY array")
                    arrays[name] = array
                    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
                        raise ValueError(f"invalid {name} NPY header")
                    if array.offset + array.nbytes != description["file_bytes"]:
                        raise ValueError(f"invalid {name} NPY payload size")
                return _Entry(arrays, metadata, directory, signatures)
            except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
                for array in arrays.values():
                    array._mmap.close()
                return None, str(exc)

    def _publish(self, source, identity, parent, pointer):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dataset.taf_dataset import _column_matrix

        temporary = None
        # No old generation or pointer is removed, including on disk-full errors.
        with self._timer("build_seconds"):
            try:
                with self._timer("parquet_open_seconds", "parquet_open_calls"):
                    parquet = pq.ParquetFile(source)
                with parquet:
                    rows = parquet.metadata.num_rows
                    if rows < 1:
                        raise ValueError(f"Cannot cache empty sensor file: {source}")
                    estimated = rows * 608 + math.ceil(rows / _CHUNK_ROWS) * 240 + 8192
                    if shutil.disk_usage(parent).free < estimated:
                        raise OSError(f"Insufficient disk space for sensor cache ({estimated} bytes): {parent}")
                    metadata = {"identity": identity, "rows": rows, "chunk_rows": _CHUNK_ROWS,
                                "schema_sha256": hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest(),
                                "rebuild_token": self.rebuild_token, "arrays": {}}
                    temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=parent))
                    hashes = {name: [] for name in _ARRAYS}
                    hashers = {name: hashlib.sha256() for name in _ARRAYS}
                    with contextlib.ExitStack() as stack:
                        files = {name: stack.enter_context((temporary / (name + ".npy")).open("xb"))
                                 for name in _ARRAYS}
                        for name, (dtype, tail) in _ARRAYS.items():
                            np.lib.format.write_array_header_2_0(files[name], {
                                "descr": dtype.str, "fortran_order": False, "shape": (rows, *tail)})
                        reader = iter(parquet.iter_batches(batch_size=self.batch_rows, columns=_COLUMNS))
                        written = 0
                        while True:
                            started = time.perf_counter() if self.stats_enabled else None
                            try:
                                batch = next(reader)
                            except StopIteration:
                                break
                            if started is not None:
                                self._add("parquet_read_seconds", time.perf_counter() - started)
                                self._add("parquet_read_calls")
                            with self._timer("conversion_seconds", "conversion_calls"):
                                values = {"index": np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64),
                                          "force": _column_matrix(pa.chunked_array([batch.column(1)]), _COLUMNS[1]),
                                          "pressure": _column_matrix(pa.chunked_array([batch.column(2)]), _COLUMNS[2])}
                                for name, (dtype, tail) in _ARRAYS.items():
                                    if values[name].shape != (len(batch), *tail):
                                        raise ValueError(f"Unexpected {name} shape in {source}: {values[name].shape}")
                                    values[name] = np.ascontiguousarray(values[name], dtype=dtype)
                            for name, array in values.items():
                                files[name].write(memoryview(array).cast("B"))
                                cursor = 0
                                while cursor < len(batch):
                                    count = min(len(batch) - cursor, _CHUNK_ROWS - (written + cursor) % _CHUNK_ROWS)
                                    hashers[name].update(memoryview(array[cursor:cursor+count]).cast("B"))
                                    cursor += count
                                    if (written + cursor) % _CHUNK_ROWS == 0:
                                        hashes[name].append(hashers[name].hexdigest())
                                        hashers[name] = hashlib.sha256()
                            written += len(batch)
                        if written != rows:
                            raise ValueError(f"Parquet row count changed while caching: {source}")
                        for name, (dtype, tail) in _ARRAYS.items():
                            if written % _CHUNK_ROWS:
                                hashes[name].append(hashers[name].hexdigest())
                            files[name].flush()
                            os.fsync(files[name].fileno())
                            metadata["arrays"][name] = {"dtype": dtype.str, "shape": [rows, *tail],
                                "file_bytes": files[name].tell(), "sha256_blocks": hashes[name]}
                    if self.identity(source) != identity:
                        raise RuntimeError(f"Source changed while building sensor cache: {source}")
                    contents = _json_bytes(metadata)
                    with (temporary / "metadata.json").open("xb") as file:
                        file.write(contents)
                        file.flush()
                        os.fsync(file.fileno())
                    _fsync_directory(temporary)
                    generation = "gen-" + uuid.uuid4().hex
                    os.rename(temporary, parent / generation)
                    temporary = None
                    _fsync_directory(parent)
                    publication = {"generation": generation, "metadata_sha256": hashlib.sha256(contents).hexdigest()}
                    with tempfile.NamedTemporaryFile(dir=parent, prefix=".pointer-", delete=False) as file:
                        temporary = Path(file.name)
                        file.write(_json_bytes(publication))
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary, pointer)
                    temporary = None
                    _fsync_directory(parent)
                    self._add("builds")
                    self._add("bytes_written", sum(a["file_bytes"] for a in metadata["arrays"].values()) + len(contents))
            finally:
                if temporary is not None:
                    if temporary.is_dir():
                        shutil.rmtree(temporary)
                    else:
                        temporary.unlink(missing_ok=True)

    def get(self, source):
        """Return process-local mappings; never expose them to a tensor caller."""
        if self._pid != os.getpid():
            _close_entries(self._entries)
            self._stats = dict.fromkeys(_COUNTERS, 0.0)
            self._pending = dict.fromkeys(_COUNTERS, 0.0)
            self._pid = os.getpid()
            self._finalizer = None
        identity = self.identity(source)
        parent, pointer, key = self._paths(identity)
        entry = self._entries.get(key)
        if entry is not None and entry.unchanged():
            self._entries.move_to_end(key)
            self._add("mapping_hits")
            self._add("cache_hits")
            return entry
        if entry is not None:
            self._entries.pop(key).close()
        while len(self._entries) >= self.max_open_files:
            _, old = self._entries.popitem(last=False)
            old.close()
            self._add("mapping_evictions")
        loaded = self._load(pointer, identity)
        valid = isinstance(loaded, _Entry)
        force = (self.rebuild and key not in self._rebuilt
                 and (not valid or loaded.metadata.get("rebuild_token") != self.rebuild_token))
        if valid and not force:
            entry = loaded
            self._add("cache_hits")
        else:
            if valid:
                loaded.close()
            parent.mkdir(parents=True, exist_ok=True)
            with (parent / ".lock").open("a+b") as lock:
                started = time.perf_counter() if self.stats_enabled else None
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if started is not None:
                    self._add("lock_wait_seconds", time.perf_counter() - started)
                if self.identity(source) != identity:
                    raise RuntimeError(f"Source changed while waiting for sensor cache lock: {source}")
                loaded = self._load(pointer, identity)
                valid = isinstance(loaded, _Entry)
                force = (self.rebuild and key not in self._rebuilt
                         and (not valid or loaded.metadata.get("rebuild_token") != self.rebuild_token))
                if not valid or force:
                    if valid:
                        loaded.close()
                    elif pointer.exists():
                        self._add("repairs")
                        warnings.warn(f"Rebuilding invalid sensor cache for {source}: {loaded[1]}", RuntimeWarning)
                    self._add("cache_misses")
                    self._publish(source, identity, parent, pointer)
                    loaded = self._load(pointer, identity)
                    if not isinstance(loaded, _Entry):
                        raise RuntimeError(f"Published sensor cache is invalid: {source}: {loaded[1]}")
                else:
                    self._add("cache_hits")
                entry = loaded
        self._rebuilt.add(key)
        self._entries[key] = entry
        self._add("mapping_opens")
        if self._finalizer is None:
            from multiprocessing.util import Finalize
            self._finalizer = Finalize(self, _close_entries, args=(self._entries,), exitpriority=10)
        return entry

    def _verify(self, entry, name, positions):
        with self._timer("verify_seconds"):
            for chunk in np.unique(np.asarray(positions, dtype=np.int64) // _CHUNK_ROWS):
                chunk = int(chunk)
                if chunk in entry.verified[name]:
                    continue
                array = entry.arrays[name][chunk*_CHUNK_ROWS:(chunk+1)*_CHUNK_ROWS]
                actual = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                if actual != entry.metadata["arrays"][name]["sha256_blocks"][chunk]:
                    raise SensorCacheCorruptionError(
                        f"Sensor cache checksum mismatch: {entry.directory}/{name}.npy block {chunk}; "
                        "rebuild with data.sensor_mmap_rebuild=true (source data was not changed)")
                entry.verified[name].add(chunk)

    def rows(self, source, row_start, offsets):
        try:
            entry = self.get(source)
            before = resource.getrusage(resource.RUSAGE_SELF) if self.profile_page_faults else None
            with self._timer("access_seconds", "access_calls"):
                arrays = entry.arrays
                self._verify(entry, "index", [row_start])
                expected = arrays["index"][row_start] + offsets
                positions = np.searchsorted(arrays["index"], expected)
                if np.any(positions >= len(arrays["index"])):
                    raise ValueError("Episode row mapping does not match dataset index")
                self._verify(entry, "index", positions)
                if not np.array_equal(arrays["index"][positions], expected):
                    raise ValueError("Episode row mapping does not match dataset index")
                self._verify(entry, "force", positions)
                self._verify(entry, "pressure", positions)
                # Advanced indexing copies into writable arrays, safe after LRU eviction.
                force = arrays["force"][positions]
                pressure = arrays["pressure"][positions].reshape(-1, 12, 12)
            if before is not None:
                after = resource.getrusage(resource.RUSAGE_SELF)
                self._add("minor_page_faults", after.ru_minflt - before.ru_minflt)
                self._add("major_page_faults", after.ru_majflt - before.ru_majflt)
            return force, pressure
        except SensorCacheCorruptionError:
            # Do not retain a known-corrupt mapping after an exception.
            self.close()
            raise
        finally:
            self._flush()

    def prepare(self, sources):
        """Explicit CPU-only preparation. Call outside measured training intervals."""
        before = self.stats()
        rows = payload = 0
        sources = sorted(set(map(Path, sources)))
        started = time.perf_counter()
        try:
            for source in sources:
                entry = self.get(source)
                rows += entry.metadata["rows"]
                payload += sum(array.nbytes for array in entry.arrays.values())
        finally:
            self.close()
        after = self.stats()
        return {"files": len(sources), "rows": rows, "payload_bytes": payload,
                "prepare_seconds": time.perf_counter() - started,
                "stats": {key: after[key] - before[key] for key in before},
                "disk_bytes": self.disk_bytes()}

    def disk_bytes(self):
        """All versions, complete orphan generations and interrupted files included."""
        total = 0
        for path in self.directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # A writer may atomically rename its own temporary directory during inventory.
                continue
        return total

    @staticmethod
    def estimate(sources):
        import pyarrow.parquet as pq
        sources = sorted(set(map(Path, sources)))
        rows = sum(pq.read_metadata(path).num_rows for path in sources)
        return {"files": len(sources), "rows": rows, "payload_bytes": rows * 608,
                "note": "NPY headers, block checksums, generations and temporary files cost extra disk space"}

    def close(self):
        _close_entries(self._entries)
        self._flush()

    def __getstate__(self):
        return {**self.__dict__, "_entries": OrderedDict(), "_finalizer": None, "_pid": None,
                "_stats": dict.fromkeys(_COUNTERS, 0.0), "_pending": dict.fromkeys(_COUNTERS, 0.0)}

    def __del__(self):
        if hasattr(self, "_entries"):
            self.close()
