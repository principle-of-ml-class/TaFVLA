"""Persistent per-frame cache for a frozen tactile backbone only."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_VERSION = 1


def _digest_backbone(encoder, image_size) -> str:
    digest = hashlib.sha256()
    for name, value in encoder.backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(str(encoder.input_size).encode())
    digest.update(encoder.interpolation.encode())
    digest.update(str(encoder.backbone_chunk_frames).encode())
    digest.update(encoder.pixel_mean.detach().cpu().numpy().tobytes())
    digest.update(encoder.pixel_std.detach().cpu().numpy().tobytes())
    digest.update(str(image_size).encode())
    return digest.hexdigest()


class FrozenFrameCache:
    """Features are keyed by video PTS ordinal, independent of temporal N."""

    def __init__(self, source, model, directory: str | Path, *, device: torch.device,
                 amp_enabled: bool, batch_size: int = 32, rebuild: bool = False,
                 selected_ordinals: dict[Path, set[int]] | None = None,
                 max_open_files: int = 8) -> None:
        encoder = model.tactile_encoder
        if not encoder.freeze_backbone:
            raise ValueError("Frame caching requires a frozen backbone")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self.source = source
        self.encoder = encoder
        self.device = device
        self.amp_enabled = amp_enabled
        self.batch_size = batch_size
        self.rebuild = rebuild
        self.selected_ordinals = selected_ordinals
        self.directory = Path(directory).expanduser().resolve() / _digest_backbone(encoder, source.image_size)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._arrays = OrderedDict()
        self._ordinals = {}
        self._data_paths: dict[Path, Path] = {}
        self.build_hits = 0
        self.build_misses = 0
        started = time.perf_counter()
        self._build()
        self.build_seconds = time.perf_counter() - started
        self.disk_bytes = sum(path.stat().st_size for path in self._data_paths.values())

    def _identity(self, video: Path, entries: dict[int, tuple]) -> dict:
        stat = video.stat()
        ordered = sorted(entries)
        pts = [float(entries[index][0].video_timestamps[entries[index][1]]) for index in ordered]
        return {"version": _VERSION, "video": str(video.resolve()),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "ordinals": ordered, "timestamps": pts,
                "dim": self.encoder.backbone_dim,
                "dtype": "float16" if self.amp_enabled else "float32"}

    def _paths(self, video: Path, identity: dict) -> tuple[Path, Path, Path]:
        stem = hashlib.sha256(str(video.resolve()).encode()).hexdigest()
        version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        return (self.directory / f"{stem}-{version}.npy", self.directory / f"{stem}-{version}.json",
                self.directory / f"{stem}.lock")

    @staticmethod
    def _valid(data_path: Path, meta_path: Path, identity: dict) -> bool:
        if not data_path.is_file() or not meta_path.is_file():
            return False
        try:
            if json.loads(meta_path.read_text()) != identity:
                return False
            array = np.load(data_path, mmap_mode="r", allow_pickle=False)
            try:
                return (array.shape == (len(identity["ordinals"]), identity["dim"])
                        and array.dtype == np.dtype(identity["dtype"]))
            finally:
                array._mmap.close()
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _build(self) -> None:
        videos: dict[Path, list] = {}
        for ep in self.source.all_episodes:
            videos.setdefault(ep.video, []).append(ep)
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            for video, episodes in videos.items():
                entries: dict[int, tuple] = {}
                allowed = (self.selected_ordinals.get(video, set())
                           if self.selected_ordinals is not None else None)
                for ep in episodes:
                    for offset, ordinal in enumerate(ep.video_frame_indices):
                        if allowed is None or int(ordinal) in allowed:
                            entries.setdefault(int(ordinal), (ep, offset))
                if not entries:
                    continue
                identity = self._identity(video, entries)
                data_path, meta_path, lock_path = self._paths(video, identity)
                # Read old valid sidecars without replacing them. New sidecars are
                # immutable per identity, preventing AMP/subset processes from racing.
                legacy_meta = self.directory / (hashlib.sha256(str(video.resolve()).encode()).hexdigest() + ".json")
                if not meta_path.exists() and self._valid(data_path, legacy_meta, identity):
                    meta_path = legacy_meta
                with lock_path.open("a+b") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    if self.rebuild or not self._valid(data_path, meta_path, identity):
                        self.build_misses += 1
                        self._write(video, entries, identity, data_path, meta_path)
                    else:
                        self.build_hits += 1
                self._data_paths[video] = data_path
                self._ordinals[video] = np.asarray(identity["ordinals"], dtype=np.int64)
        finally:
            self.encoder.train(was_training)

    @property
    def hit_rate(self) -> float | None:
        total = self.build_hits + self.build_misses
        return self.build_hits / total if total else None

    def _write(self, video: Path, entries: dict[int, tuple], identity: dict,
               data_path: Path, meta_path: Path) -> None:
        ordered = identity["ordinals"]
        temporary_data = None
        temporary_meta = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npy", delete=False) as file:
                temporary_data = Path(file.name)
            array = np.lib.format.open_memmap(
                temporary_data, mode="w+", dtype=identity["dtype"],
                shape=(len(ordered), identity["dim"]),
            )
            cursor = 0
            while cursor < len(ordered):
                ep, offset = entries[ordered[cursor]]
                end = cursor + 1
                while (end < len(ordered) and end - cursor < self.batch_size
                       and entries[ordered[end]][0] is ep
                       and entries[ordered[end]][1] == offset + end - cursor):
                    end += 1
                offsets = np.arange(offset, offset + end - cursor, dtype=np.int64)
                images, _ = self.source._images(ep, offsets)
                # Independent frames are B=K,T=1, not a temporal window T=K.
                # The default build batch (32) can exceed max_window_size (20).
                tactile = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().unsqueeze(1)
                tactile = tactile.to(self.device)
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
                ):
                    features = self.encoder.encode_backbone(tactile)[:, 0]
                array[cursor:end] = features.cpu().numpy()
                cursor = end
            array.flush()
            del array
            stat = video.stat()
            if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
                raise RuntimeError(f"Video changed while computing frame features: {video}")
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory,
                                             suffix=".json", delete=False) as file:
                temporary_meta = Path(file.name)
                json.dump(identity, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_data, data_path)
            temporary_data = None
            os.replace(temporary_meta, meta_path)
            temporary_meta = None
        finally:
            if temporary_data is not None:
                temporary_data.unlink(missing_ok=True)
            if temporary_meta is not None:
                temporary_meta.unlink(missing_ok=True)

    @staticmethod
    def estimate(source, backbone_dim, *, amp_enabled):
        videos = {}
        for ep in source.all_episodes:
            videos.setdefault(ep.video, []).append(ep.video_frame_indices)
        frames = sum(len(np.unique(np.concatenate(parts))) for parts in videos.values())
        return {"unique_frames": frames, "videos": len(videos),
                "feature_bytes": frames * backbone_dim * (2 if amp_enabled else 4),
                "bytes_per_frame": backbone_dim * (2 if amp_enabled else 4)}

    def features(self, ep, offsets: np.ndarray) -> torch.Tensor:
        with self.source._stage("feature_lookup") if hasattr(self.source, "_stage") else contextlib.nullcontext():
            item = self._arrays.get(ep.video)
            if item is None:
                while len(self._arrays) >= self.max_open_files:
                    _, (_, old_array) = self._arrays.popitem(last=False)
                    old_array._mmap.close()
                array = np.load(self._data_paths[ep.video], mmap_mode="r", allow_pickle=False)
                item = (self._ordinals[ep.video], array)
                self._arrays[ep.video] = item
            else:
                self._arrays.move_to_end(ep.video)
            ordinals, array = item
            requested = ep.video_frame_indices[offsets]
            positions = np.searchsorted(ordinals, requested)
            if np.any(positions >= len(ordinals)) or not np.array_equal(ordinals[positions], requested):
                raise ValueError(f"Missing cached frame for {ep.video}")
            return torch.from_numpy(np.array(array[positions], copy=True))

    def close(self):
        for _, array in self._arrays.values():
            array._mmap.close()
        self._arrays.clear()

    def __del__(self):
        if hasattr(self, "_arrays"):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["encoder"] = None
        state["_arrays"] = OrderedDict()
        return state


class CachedWindowDataset(Dataset):
    """Use cached ViT outputs for training/validation windows without decoding video."""

    def __init__(self, dataset, cache: FrozenFrameCache) -> None:
        self.dataset = dataset
        self.cache = cache
        self.source = getattr(dataset, "source", dataset)
        self.window_size = dataset.window_size

    def __len__(self) -> int:
        return len(self.dataset)

    def _locate(self, index):
        return self.dataset._locate(index)

    def __getitem__(self, index: int) -> dict:
        ep, start = self.dataset._locate(index)
        offsets = np.arange(start, start + self.window_size, dtype=np.int64)
        force, pressure = self.source._sensor_rows(ep, offsets)
        if hasattr(self.source, "_sensor_tensors"):
            force, pressure = self.source._sensor_tensors(force, pressure)
        else:
            force, pressure = torch.from_numpy(force), torch.from_numpy(pressure)
        result = {"backbone_features": self.cache.features(ep, offsets),
                  "force": force, "pressure": pressure}
        if hasattr(self.dataset, "rows"):
            result["metadata"] = {"sample_id": self.dataset.rows[index]["sample_id"],
                                  "frame_indices": torch.from_numpy(ep.frames[offsets].copy())}
        return result
