"""Reusable dataset splitting, auditing, normalization, and sampling."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Sampler

from dataset.taf_dataset import TaFDataset
from utils.live_progress import emit as progress_event


def episode_window_ranges(dataset: TaFDataset) -> list[range]:
    ranges = []
    previous = 0
    for end in dataset._ends:
        ranges.append(range(previous, end))
        previous = end
    return ranges


def split_by_group(
    dataset: TaFDataset, *, unit: str, validation_fraction: float, seed: int,
) -> tuple[list[int], list[int], set[int], set[int], list[str], list[str]]:
    """Split whole sequences/episodes so overlapping windows cannot leak."""
    if unit not in {"sequence", "episode"}:
        raise ValueError("data.split_unit must be 'sequence' or 'episode'")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for episode_id, episode in enumerate(dataset.episodes):
        if unit == "sequence":
            key = episode.sequence_id
        else:
            from utils.dynamics import split_group_key
            key = split_group_key({"sensor_configuration": episode.sensor_type,
                                   "sequence_id": episode.sequence_id,
                                   "episode_index": episode.episode_index}, "episode")
        groups[key].append(episode_id)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(
            f"Need at least two {unit} groups for a leakage-free train/validation split; got {len(keys)}"
        )
    random.Random(seed).shuffle(keys)
    val_count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    val_keys = set(keys[:val_count])
    train_keys = set(keys[val_count:])
    train_eps = {episode for key in train_keys for episode in groups[key]}
    val_eps = {episode for key in val_keys for episode in groups[key]}
    ranges = episode_window_ranges(dataset)
    train_indices = [index for episode in sorted(train_eps) for index in ranges[episode]]
    val_indices = [index for episode in sorted(val_eps) for index in ranges[episode]]
    if not train_indices or not val_indices:
        raise ValueError("Train/validation split produced an empty window set")
    return train_indices, val_indices, train_eps, val_eps, sorted(train_keys), sorted(val_keys)


def audit_temporal_alignment(dataset: TaFDataset, windows_per_episode: int) -> None:
    """Decode representative windows after constructor-level timestamp checks."""
    if dataset.invalid_sequences:
        raise RuntimeError(
            "Dataset contains skipped invalid sequences; refusing to train: "
            + repr(dataset.invalid_sequences)
        )
    if windows_per_episode < 1:
        raise ValueError("alignment_decode_windows_per_episode must be >= 1")
    ranges = episode_window_ranges(dataset)
    decoded = 0
    last_report = time.monotonic()
    for episode_id, window_range in enumerate(ranges):
        count = len(window_range)
        offsets = np.linspace(0, count - 1, min(windows_per_episode, count), dtype=int)
        for offset in np.unique(offsets):
            sample = dataset[window_range.start + int(offset)]
            for key in ("tactile", "force", "pressure"):
                if not torch.isfinite(sample[key]).all():
                    raise ValueError(f"Non-finite {key} in episode index {episode_id}")
            metadata = sample["metadata"]
            if not torch.all(torch.diff(metadata["frame_indices"]) == 1):
                raise ValueError(f"Non-contiguous frames in episode index {episode_id}")
            if not torch.all(torch.diff(metadata["timestamps"]) > 0):
                raise ValueError(f"Non-increasing sensor timestamps in episode index {episode_id}")
            if not torch.all(torch.diff(metadata["video_timestamps"]) > 0):
                raise ValueError(f"Non-increasing video timestamps in episode index {episode_id}")
            decoded += 1
        now = time.monotonic()
        if now - last_report >= 0.5 or episode_id + 1 == len(ranges):
            progress_event("Temporal Alignment Audit", current=episode_id + 1,
                           total=len(ranges), detail=f"decoded windows={decoded}")
            last_report = now
    print(f"Temporal alignment audit: PASS ({len(dataset.episodes)} episodes, {decoded} decoded windows)")


def compute_normalization(
    dataset: TaFDataset, train_episode_ids: set[int], chunk_frames: int, eps: float,
    selected_indices: list[int] | None = None,
    normalization_window_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-taxel/per-wrench-component population stats from train frames only."""
    if chunk_frames < 1 or eps <= 0:
        raise ValueError("normalization_chunk_frames and normalization_eps must be positive")
    pressure_sum = torch.zeros(12, 12, dtype=torch.float64)
    pressure_square_sum = torch.zeros_like(pressure_sum)
    force_sum = torch.zeros(6, dtype=torch.float64)
    force_square_sum = torch.zeros_like(force_sum)
    count = 0
    selected_offsets: dict[int, set[int]] | None = None
    if selected_indices is not None:
        selected_offsets = defaultdict(set)
        ids = {id(ep): i for i, ep in enumerate(dataset.episodes)}
        for index in selected_indices:
            ep, start = dataset._locate(index)
            end = start + dataset.window_size
            width = normalization_window_size or dataset.window_size
            if end-width < 0:
                raise ValueError("Normalization window crossed episode boundary")
            selected_offsets[ids[id(ep)]].update(range(end-width, end))
    total_frames = (sum(len(selected_offsets[episode_id]) for episode_id in train_episode_ids)
                    if selected_offsets is not None else
                    sum(dataset.episodes[episode_id].length for episode_id in train_episode_ids))
    started = time.monotonic()
    last_report = started
    for episode_id in sorted(train_episode_ids):
        episode = dataset.episodes[episode_id]
        all_offsets = (np.asarray(sorted(selected_offsets[episode_id]), dtype=np.int64)
                       if selected_offsets is not None else np.arange(episode.length, dtype=np.int64))
        for start in range(0, len(all_offsets), chunk_frames):
            offsets = all_offsets[start:start + chunk_frames]
            force_array, pressure_array = dataset._sensor_rows(episode, offsets)
            pressure = torch.from_numpy(pressure_array).double()
            force = torch.from_numpy(force_array).double()
            if not torch.isfinite(pressure).all() or not torch.isfinite(force).all():
                raise ValueError(
                    f"Non-finite force data in training episode "
                    f"{episode.sequence_id}/{episode.episode_index}"
                )
            pressure_sum += pressure.sum(0)
            pressure_square_sum += pressure.square().sum(0)
            force_sum += force.sum(0)
            force_square_sum += force.square().sum(0)
            count += len(offsets)
            now = time.monotonic()
            if now - last_report >= 0.5 or count == total_frames:
                progress_event("Normalization", current=count, total=total_frames,
                               detail=f"frames; elapsed={now-started:.1f}s")
                last_report = now
    if count < 2:
        raise ValueError("At least two training frames are required for normalization")
    pressure_mean = pressure_sum / count
    force_mean = force_sum / count
    pressure_std = (
        pressure_square_sum / count - pressure_mean.square()
    ).clamp_min(0).sqrt().clamp_min(eps)
    force_std = (
        force_square_sum / count - force_mean.square()
    ).clamp_min(0).sqrt().clamp_min(eps)
    print(f"Normalization stats: {count} unique training frames (validation excluded)")
    return {
        "pressure_mean": pressure_mean.float(),
        "pressure_std": pressure_std.float(),
        "force_mean": force_mean.float(),
        "force_std": force_std.float(),
    }


class TemporalDiverseBatchSampler(Sampler[list[int]]):
    """Greedily separate nearby same-episode windows without dropping data."""

    def __init__(
        self, dataset: TaFDataset, subset_indices: list[int], batch_size: int,
        min_gap: int, seed: int, progress_stage: str | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("InfoNCE training requires batch_size >= 2")
        self.batch_size = batch_size
        self.min_gap = max(0, min_gap)
        self.seed = seed
        self.epoch = 0
        self.progress_stage = progress_stage
        episode_ids = {id(episode): index for index, episode in enumerate(dataset.episodes)}
        self.items: list[tuple[int, int, int]] = []
        total = len(subset_indices)
        last_report = time.monotonic()
        if self.progress_stage:
            progress_event(self.progress_stage, current=0, total=total,
                           phase="Index", detail="locating candidate windows")
        for local_index, global_index in enumerate(subset_indices):
            episode, start = dataset._locate(global_index)
            self.items.append((local_index, episode_ids[id(episode)], start))
            if self.progress_stage and ((local_index + 1) % 5000 == 0 or local_index + 1 == total):
                now = time.monotonic()
                if now - last_report >= 0.5 or local_index + 1 == total:
                    progress_event(self.progress_stage,
                                   current=local_index + 1, total=total,
                                   phase="Index", detail="locating candidate windows")
                    last_report = now

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        items = list(self.items)
        random.Random(self.seed + self.epoch).shuffle(items)
        queue = deque(items)
        while queue:
            selected: list[tuple[int, int, int]] = []
            deferred: list[tuple[int, int, int]] = []
            for _ in range(len(queue)):
                item = queue.popleft()
                compatible = all(
                    item[1] != other[1] or abs(item[2] - other[2]) >= self.min_gap
                    for other in selected
                )
                if compatible and len(selected) < self.batch_size:
                    selected.append(item)
                else:
                    deferred.append(item)
                if len(selected) == self.batch_size:
                    break
            queue.extend(deferred)
            if len(selected) == 1 and queue:
                selected.append(queue.popleft())
            if len(selected) >= 2:
                yield [item[0] for item in selected]


@dataclass(frozen=True)
class ValidationSamplingReport:
    """Diagnostics for a fixed, full-size validation batch plan."""

    seed: int
    batch_size: int
    min_window_start_gap: int
    candidate_samples: int
    used_samples: int
    dropped_samples: int
    batch_count: int
    violating_batches: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def temporal_batch_violations(
    dataset: TaFDataset,
    subset_indices: list[int],
    batches: list[list[int]],
    min_gap: int,
    progress_stage: str | None = None,
) -> list[dict[str, int]]:
    """Return every same-episode pair that violates ``min_gap``.

    Batch indices are local to ``Subset(dataset, subset_indices)`` so this also
    audits the exact indices consumed by a DataLoader batch sampler.
    """
    episode_ids = {id(episode): index for index, episode in enumerate(dataset.episodes)}
    # Local indices are dense; a list avoids a second hash table over every anchor.
    located: list[tuple[int, int]] = []
    last_report = time.monotonic()
    if progress_stage:
        progress_event(progress_stage, current=0, total=len(subset_indices),
                       phase="Audit index", detail="checking source locations")
    for local_index, global_index in enumerate(subset_indices):
        episode, start = dataset._locate(global_index)
        located.append((episode_ids[id(episode)], int(start)))
        if progress_stage and ((local_index + 1) % 5000 == 0 or local_index + 1 == len(subset_indices)):
            now = time.monotonic()
            if now - last_report >= 0.5 or local_index + 1 == len(subset_indices):
                progress_event(progress_stage, current=local_index + 1, total=len(subset_indices),
                               phase="Audit index", detail="checking source locations")
                last_report = now

    violations: list[dict[str, int]] = []
    if progress_stage:
        progress_event(progress_stage, current=0, total=len(batches),
                       phase="Audit batches", detail="checking strict temporal gaps")
    last_report = time.monotonic()
    for batch_index, batch in enumerate(batches):
        by_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for offset, local in enumerate(batch):
            episode, start = located[local]
            by_episode[episode].append((offset, local, start))
        # Only same-episode pairs can violate the temporal constraint.
        # Iterate left and right positions in legacy order for identical reports.
        for left_offset, left_local in enumerate(batch):
            left_episode, left_start = located[left_local]
            for right_offset, right_local, right_start in by_episode[left_episode]:
                if right_offset <= left_offset:
                    continue
                distance = abs(left_start - right_start)
                if distance < min_gap:
                    violations.append({
                        "batch_index": batch_index,
                        "left_local_index": left_local,
                        "right_local_index": right_local,
                        "episode_list_index": left_episode,
                        "left_window_start": left_start,
                        "right_window_start": right_start,
                        "window_start_gap": distance,
                    })
        if progress_stage and ((batch_index + 1) % 500 == 0 or batch_index + 1 == len(batches)):
            now = time.monotonic()
            if now - last_report >= 0.5 or batch_index + 1 == len(batches):
                progress_event(progress_stage, current=batch_index + 1, total=len(batches),
                               phase="Audit batches", detail=f"violating pairs={len(violations):,}")
                last_report = now
    return violations


class FixedValidationBatchSampler(TemporalDiverseBatchSampler):
    """Build deterministic, strict temporal batches once for all evaluations.

    Unlike the training sampler, this evaluation-only variant never relaxes the
    temporal constraint and never emits a partial InfoNCE batch. Candidates that
    cannot be packed into a full compatible batch are explicitly reported.
    """

    def __init__(
        self,
        dataset: TaFDataset,
        subset_indices: list[int],
        batch_size: int,
        min_gap: int,
        seed: int,
        progress_stage: str | None = None,
    ) -> None:
        super().__init__(dataset, subset_indices, batch_size, min_gap, seed, progress_stage)
        self._batches, dropped = self._build_fixed_batches()
        violations = temporal_batch_violations(
            dataset, subset_indices, self._batches, self.min_gap, self.progress_stage,
        )
        self.report = ValidationSamplingReport(
            seed=self.seed,
            batch_size=self.batch_size,
            min_window_start_gap=self.min_gap,
            candidate_samples=len(self.items),
            used_samples=sum(map(len, self._batches)),
            dropped_samples=len(dropped),
            batch_count=len(self._batches),
            violating_batches=len({item["batch_index"] for item in violations}),
        )
        self.dropped_local_indices = [item[0] for item in dropped]
        self.violations = violations
        if violations:
            first = violations[0]
            raise RuntimeError(
                "Fixed validation sampler produced a temporal-gap violation: "
                f"batch={first['batch_index']}, episode={first['episode_list_index']}, "
                f"starts=({first['left_window_start']}, "
                f"{first['right_window_start']}), gap={first['window_start_gap']} "
                f"< required {self.min_gap}"
            )
        if not self._batches:
            raise RuntimeError(
                "Validation cannot form one full temporally diverse InfoNCE batch: "
                f"candidates={len(self.items)}, batch_size={self.batch_size}, "
                f"required_same_episode_gap={self.min_gap}."
            )

    @staticmethod
    def _compatible(
        item: tuple[int, int, int],
        selected: list[tuple[int, int, int]],
        min_gap: int,
    ) -> bool:
        return all(
            item[1] != other[1] or abs(item[2] - other[2]) >= min_gap
            for other in selected
        )

    def _build_fixed_batches(
        self, *, seed: int | None = None,
    ) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
        total = len(self.items)
        if self.progress_stage:
            progress_event(self.progress_stage, current=0, total=total,
                           phase="Shuffle", detail="shuffling candidates with local seed")
        shuffled = list(self.items)
        random.Random(self.seed if seed is None else seed).shuffle(shuffled)
        remaining = deque(shuffled)
        del shuffled
        batches: list[list[int]] = []
        dropped: list[tuple[int, int, int]] = []
        last_report = time.monotonic()
        scans = 0
        if self.progress_stage:
            progress_event(self.progress_stage, current=0, total=total,
                           phase="Packing", detail="building full temporal-gap batches")

        # Preserve the exact legacy greedy order, including the rejected prefix.
        # Only pop scanned candidates; never copy the unscanned suffix per batch.
        while len(remaining) >= self.batch_size:
            selected: list[tuple[int, int, int]] = []
            deferred: list[tuple[int, int, int]] = []
            starts_by_episode: dict[int, list[int]] = {}
            for _ in range(len(remaining)):
                item = remaining.popleft()
                _, episode, start = item
                starts = starts_by_episode.get(episode)
                if starts is None or all(abs(start - other) >= self.min_gap for other in starts):
                    selected.append(item)
                    if starts is None:
                        starts_by_episode[episode] = [start]
                    else:
                        starts.append(start)
                else:
                    deferred.append(item)
                scans += 1
                if len(selected) == self.batch_size:
                    break
                if self.progress_stage and scans % 4096 == 0:
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        progress_event(self.progress_stage,
                                       current=total - len(remaining) - len(deferred) - len(selected),
                                       total=total, phase="Packing",
                                       detail=f"batches={len(batches):,}; dropped={len(dropped):,}; scanned={scans:,}")
                        last_report = now
            # extendleft reverses its input, so reverse the deferred prefix first.
            remaining.extendleft(reversed(deferred))
            if len(selected) == self.batch_size:
                batches.append([item[0] for item in selected])
            else:
                dropped.extend(selected)
            if self.progress_stage and (len(batches) % 100 == 0 or len(remaining) < self.batch_size):
                now = time.monotonic()
                if now - last_report >= 0.5 or len(remaining) < self.batch_size:
                    progress_event(self.progress_stage, current=total - len(remaining), total=total,
                                   phase="Packing", detail=f"batches={len(batches):,}; dropped={len(dropped):,}; scanned={scans:,}")
                    last_report = now

        dropped.extend(remaining)
        if self.progress_stage:
            progress_event(self.progress_stage, current=total, total=total,
                           phase="Packing", detail=f"batches={len(batches):,}; dropped={len(dropped):,}; scanned={scans:,}")
        return batches, dropped

    def set_epoch(self, epoch: int) -> None:
        """Validation order is intentionally invariant across epochs."""

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from (list(batch) for batch in self._batches)


class EpochShuffledBatchSampler(FixedValidationBatchSampler):
    """Training-only full batches, strict temporal gap, new plan each epoch.

    Reuse the audited strict packing algorithm, not the fixed validation plan.
    A local seed + epoch makes the plan reproducible across runs and window sizes
    when the common anchors, gap, and batch size are the same.
    """

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if epoch == self.epoch:
            return
        epoch_seed = self.seed + epoch
        # Do not emit repeated sampler initialization progress into the training UI.
        progress_stage = self.progress_stage
        self.progress_stage = None
        try:
            batches, dropped = self._build_fixed_batches(seed=epoch_seed)
        finally:
            self.progress_stage = progress_stage
        if not batches:
            raise RuntimeError(
                f"Epoch {epoch}: no full temporally diverse training batch "
                f"(candidates={len(self.items)}, batch_size={self.batch_size}, "
                f"min_gap={self.min_gap})"
            )
        # _build_fixed_batches only admits compatible items and full batches.
        # Avoid a second quadratic all-pairs pass on every training epoch.
        self._batches = batches
        self.dropped_local_indices = [item[0] for item in dropped]
        self.violations = []
        self.report = ValidationSamplingReport(
            seed=epoch_seed,
            batch_size=self.batch_size,
            min_window_start_gap=self.min_gap,
            candidate_samples=len(self.items),
            used_samples=sum(map(len, batches)),
            dropped_samples=len(dropped),
            batch_count=len(batches),
            violating_batches=0,
        )
        self.epoch = epoch
