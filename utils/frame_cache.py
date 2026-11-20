"""Persistent per-frame cache for a frozen tactile backbone only."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
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
                 selected_ordinals: dict[Path, set[int]] | None = None) -> None:
        encoder = model.tactile_encoder
        if not encoder.freeze_backbone:
            raise ValueError("Frame caching requires a frozen backbone")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.source = source
        self.encoder = encoder
        self.device = device
        self.amp_enabled = amp_enabled
        self.batch_size = batch_size
        self.rebuild = rebuild
        self.selected_ordinals = selected_ordinals
        self.directory = Path(directory).expanduser().resolve() / _digest_backbone(encoder, source.image_size)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._arrays: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._data_paths: dict[Path, Path] = {}
        self.build_hits = 0
        self.build_misses = 0
        self._build()

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
        return (self.directory / f"{stem}-{version}.npy", self.directory / f"{stem}.json",
                self.directory / f"{stem}.lock")

    @staticmethod
    def _valid(data_path: Path, meta_path: Path, identity: dict) -> bool:
        if not data_path.is_file() or not meta_path.is_file():
            return False
        try:
            if json.loads(meta_path.read_text()) != identity:
                return False
            array = np.load(data_path, mmap_mode="r", allow_pickle=False)
            return (array.shape == (len(identity["ordinals"]), identity["dim"])
                    and array.dtype == np.dtype(identity["dtype"]))
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
                with lock_path.open("a+b") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    if self.rebuild or not self._valid(data_path, meta_path, identity):
                        self.build_misses += 1
                        self._write(video, entries, identity, data_path, meta_path)
                    else:
                        self.build_hits += 1
                self._data_paths[video] = data_path
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
                tactile = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
                tactile = tactile.to(self.device)
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
                ):
                    features = self.encoder.encode_backbone(tactile)[0]
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

    def features(self, ep, offsets: np.ndarray) -> torch.Tensor:
        item = self._arrays.get(ep.video)
        if item is None:
            data_path = self._data_paths[ep.video]
            stem = hashlib.sha256(str(ep.video.resolve()).encode()).hexdigest()
            meta_path = self.directory / f"{stem}.json"
            metadata = json.loads(meta_path.read_text())
            array = np.load(data_path, mmap_mode="r", allow_pickle=False)
            item = (np.asarray(metadata["ordinals"], dtype=np.int64), array)
            self._arrays[ep.video] = item
        ordinals, array = item
        requested = ep.video_frame_indices[offsets]
        positions = np.searchsorted(ordinals, requested)
        if np.any(positions >= len(ordinals)) or not np.array_equal(ordinals[positions], requested):
            raise ValueError(f"Missing cached frame for {ep.video}")
        return torch.from_numpy(np.array(array[positions], copy=True))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["encoder"] = None
        state["_arrays"] = {}
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

    def __getitem__(self, index: int) -> dict:
        ep, start = self.dataset._locate(index)
        offsets = np.arange(start, start + self.window_size, dtype=np.int64)
        force, pressure = self.source._sensor_rows(ep, offsets)
        return {"backbone_features": self.cache.features(ep, offsets),
                "force": torch.from_numpy(force), "pressure": torch.from_numpy(pressure)}
