"""Validated offline labels and N-independent anchor manifest."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from dataset.taf_dataset import TaFDataset


WINDOWS = (1, 3, 5, 7, 10, 20)
DYNAMICS = ("loading", "unloading", "holding", "transition", "sudden_event")
REQUIRED_LABEL = {"sensor_configuration", "sequence_id", "episode_index", "frame_index",
                  "timestamp_anchor", "contact_phase", "sudden_event", "schema_version",
                  "configuration_hash", "classifier_variant"}


def dynamics5(phase: str, sudden: bool) -> str | None:
    if not isinstance(sudden, (bool, np.bool_)) or not isinstance(phase, str) or not phase:
        raise ValueError("Missing or invalid contact_phase/sudden_event")
    if phase not in {"Loading", "Unloading", "Holding", "Transition / Unknown", "No Contact"}:
        raise ValueError(f"Unrecognized contact_phase: {phase!r}")
    if sudden:
        return "sudden_event"
    return {"Loading": "loading", "Unloading": "unloading", "Holding": "holding",
            "Transition / Unknown": "transition", "No Contact": None}[phase]


def sample_id(sensor: str, sequence: str, episode: int, anchor: int) -> str:
    return f"{sensor}/{sequence}/episode-{episode}/anchor-{anchor}"


def split_group_key(row: dict, unit: str) -> str:
    """Keep legacy sequence keys; episode keys include the sensor identity."""
    if unit == "sequence":
        return row["sequence_id"]
    if unit != "episode":
        raise ValueError(f"Unsupported split unit: {unit}")
    return json.dumps([row["sensor_configuration"], row["sequence_id"],
                       int(row["episode_index"])], separators=(",", ":"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _force_torque_unit_metadata(dataset: TaFDataset) -> dict[str, object]:
    """Read only explicit source metadata; never infer physical units from axis names."""
    found = {}
    for ep in dataset.episodes:
        parquet = getattr(ep, "parquet", None)
        if parquet is None or ep.sequence_id in found:
            continue
        info_path = Path(parquet).parents[2] / "meta" / "info.json"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        entry = features.get("observation.force_torque", {}) if isinstance(features, dict) else {}
        if isinstance(entry, dict):
            units = entry.get("units", entry.get("unit"))
            if units is not None:
                found[ep.sequence_id] = units
    return found


def create_manifest(dataset: TaFDataset, labels_path: Path, output: Path, *,
                    split_seed: int = 42, fractions: tuple[float, float, float] = (.8, .1, .1),
                    timestamp_tolerance: float = 1e-3, max_window: int = 20,
                    metadata_path: Path | None = None) -> dict:
    """Match episode-local label positions to validated raw episode timestamps."""
    if dataset.window_size != 1 or dataset.stride != 1:
        raise ValueError("Manifest source dataset must use window_size=1, stride=1")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("split fractions must be three positive values summing to one")
    if timestamp_tolerance < 0:
        raise ValueError("timestamp_tolerance must be nonnegative")
    labels = pq.read_table(labels_path)
    missing = REQUIRED_LABEL - set(labels.column_names)
    if missing:
        raise ValueError(f"Missing centered label columns: {sorted(missing)}")
    rows = labels.to_pylist()
    provenance = {(r["schema_version"], r["configuration_hash"], r["classifier_variant"]) for r in rows}
    if (len(provenance) != 1 or next(iter(provenance))[2] != "centered" or
        not all(isinstance(x, str) and x for x in next(iter(provenance)))):
        raise ValueError(f"Mixed or non-centered classifier provenance: {provenance}")
    if metadata_path is None:
        sibling = labels_path.with_name("centered_metadata.json")
        metadata_path = sibling if sibling.is_file() else None
    metadata_sha = None
    sudden_window = None
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        version, config_hash, _ = next(iter(provenance))
        if (metadata.get("schema_version") != version or metadata.get("config_hash") != config_hash or
            metadata.get("classifier_variant") != "centered" or metadata.get("uses_future_observations") is not True):
            raise ValueError("Centered label metadata does not match label provenance")
        sudden_window = metadata.get("config", {}).get("sudden_diagnostic", {}).get("window_size")
        if sudden_window != 20:
            raise ValueError(f"Expected fixed trailing sudden diagnostic of 20 frames, got {sudden_window}")
        metadata_sha = _digest(metadata_path)
    label_map = {}
    duplicates = []
    for r in rows:
        key = (r["sensor_configuration"], r["sequence_id"], r["episode_index"], r["frame_index"])
        if key in label_map:
            duplicates.append(key)
        label_map[key] = r
    if duplicates:
        raise ValueError(f"Duplicate labels: {len(duplicates)}; first={duplicates[0]}")
    episode_map = {(ep.sensor_type, ep.sequence_id, ep.episode_index): ep for ep in dataset.episodes}
    foreign = [k for k in label_map if k[:3] not in episode_map]
    if foreign:
        raise ValueError(f"Labels reference unknown episodes: {len(foreign)}; first={foreign[0]}")
    out_of_range = [k for k in label_map if not isinstance(k[3], int) or k[3] < 0 or k[3] >= episode_map[k[:3]].length]
    if out_of_range:
        raise ValueError(f"Labels reference out-of-range episode positions: {len(out_of_range)}; first={out_of_range[0]}")
    sequences = sorted({ep.sequence_id for ep in dataset.episodes})
    random.Random(split_seed).shuffle(sequences)
    if len(sequences) < 3:
        raise ValueError("At least three sequences are needed for train/validation/test")
    n_test = max(1, round(len(sequences) * fractions[2]))
    n_val = max(1, round(len(sequences) * fractions[1]))
    if n_test + n_val >= len(sequences):
        raise ValueError("Split fractions leave no training sequence")
    split_map = {s: ("test" if i < n_test else "validation" if i < n_test+n_val else "train")
                 for i, s in enumerate(sequences)}
    reasons = Counter()
    exclusions_by_split = {split: Counter() for split in ("train", "validation", "test")}
    def exclude(sequence: str, reason: str):
        reasons[reason] += 1
        exclusions_by_split[split_map[sequence]][reason] += 1
    candidates = []
    for ep in dataset.episodes:
        for anchor in range(ep.length):
            if anchor < max_window - 1:
                exclude(ep.sequence_id, "insufficient_history")
                continue
            key = (ep.sensor_type, ep.sequence_id, ep.episode_index, anchor)
            label = label_map.get(key)
            if label is None:
                exclude(ep.sequence_id, "missing_label")
                continue
            try:
                label_time = float(label["timestamp_anchor"])
            except (TypeError, ValueError):
                label_time = float("nan")
            if (not isinstance(label["frame_index"], int) or not np.isfinite(label_time) or
                abs(label_time - float(ep.timestamps[anchor])) > timestamp_tolerance):
                exclude(ep.sequence_id, "timestamp_or_position_mismatch")
                continue
            try:
                group = dynamics5(label["contact_phase"], label["sudden_event"])
            except ValueError:
                exclude(ep.sequence_id, "invalid_label")
                continue
            if group is None:
                exclude(ep.sequence_id, "no_contact")
                continue
            candidates.append({
                "sample_id": sample_id(*key), "sensor_configuration": ep.sensor_type,
                "sequence_id": ep.sequence_id, "episode_index": ep.episode_index,
                "anchor_frame_position": anchor, "source_frame_index": int(ep.frames[anchor]),
                "timestamp_anchor": float(ep.timestamps[anchor]),
                "contact_phase": label["contact_phase"], "sudden_event": bool(label["sudden_event"]),
                "dynamics5": group,
            })
    if not candidates:
        raise ValueError(f"No usable anchors; exclusions={dict(reasons)}")
    for row in candidates:
        row["split"] = split_map[row["sequence_id"]]
    label_sha = _digest(labels_path)
    unit_metadata = _force_torque_unit_metadata(dataset)
    identity = {"label_sha256": label_sha, "label_metadata_sha256": metadata_sha,
                "schema_version": next(iter(provenance))[0],
                "configuration_hash": next(iter(provenance))[1], "split_seed": split_seed,
                "split_fractions": fractions, "max_window": max_window,
                "timestamp_tolerance": timestamp_tolerance, "sample_ids": [r["sample_id"] for r in candidates]}
    manifest_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for row in candidates:
        row.update(manifest_id=manifest_id, label_sha256=label_sha,
                   classifier_schema_version=identity["schema_version"],
                   classifier_configuration_hash=identity["configuration_hash"], split_seed=split_seed,
                   force_torque_unit_metadata=(json.dumps(unit_metadata[row["sequence_id"]])
                                               if row["sequence_id"] in unit_metadata else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        previous = pq.read_table(output, columns=["manifest_id"])["manifest_id"].to_pylist()
        if not previous or set(previous) != {manifest_id}:
            raise FileExistsError(f"Existing manifest has a different identity: {output}")
    pq.write_table(pa.Table.from_pylist(candidates), output)
    support = {}
    for split in ("train", "validation", "test"):
        part = [r for r in candidates if r["split"] == split]
        support[split] = {"anchors": len(part), "sequences": len({r["sequence_id"] for r in part}),
                          "episodes": len({(r["sequence_id"], r["episode_index"]) for r in part}),
                          "dynamics": dict(Counter(r["dynamics5"] for r in part)),
                          "sensors": dict(Counter(r["sensor_configuration"] for r in part)),
                          "by_dynamics": {group: {"anchors": sum(r["dynamics5"] == group for r in part),
                                                   "episodes": len({(r["sequence_id"], r["episode_index"]) for r in part if r["dynamics5"] == group}),
                                                   "sequences": len({r["sequence_id"] for r in part if r["dynamics5"] == group})}
                                          for group in DYNAMICS}}
    report = {"manifest_id": manifest_id, "label_sha256": label_sha, "label_metadata_sha256": metadata_sha,
              "sudden_diagnostic_window": sudden_window,
              "force_torque_unit_metadata_by_sequence": unit_metadata,
              "provenance": list(next(iter(provenance))),
              "split_seed": split_seed, "split_fractions": fractions, "exclusions": dict(reasons),
              "exclusions_by_split": {k: dict(v) for k, v in exclusions_by_split.items()},
              "support": support, "rows": len(candidates)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(support[split]["anchors"] == 0 for split in ("train", "validation", "test")):
        raise ValueError(f"At least one split has no usable anchors; coverage report: {output.with_suffix('.json')}")
    return report


class ManifestDataset(Dataset):
    """Read the same anchor list at any N, retaining the original raw sensor/video loader."""

    def __init__(self, source: TaFDataset, manifest_path: Path, window_size: int,
                 *, expected_split_unit: str | None = None):
        self.source = source
        self.window_size = window_size
        self.rows = pq.read_table(manifest_path).to_pylist()
        if not self.rows or len({r["manifest_id"] for r in self.rows}) != 1:
            raise ValueError("Manifest is empty or has mixed identities")
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Manifest contains duplicate sample IDs")
        self.manifest_id = self.rows[0]["manifest_id"]
        units = {r.get("split_unit", "sequence") for r in self.rows}
        if len(units) != 1 or not units <= {"sequence", "episode"}:
            raise ValueError("Manifest has mixed or invalid split units")
        self.split_unit = next(iter(units))
        if expected_split_unit is not None and self.split_unit != expected_split_unit:
            raise ValueError("Manifest split_unit differs from data.split_unit")
        ep_lookup = {(e.sensor_type, e.sequence_id, e.episode_index): e for e in source.episodes}
        ep_order = {key: i for i, key in enumerate(ep_lookup)}
        self.episodes = source.episodes
        self._locations = []
        self._ends = []
        counts = defaultdict(int)
        prior_position = (-1, -1)
        sequence_splits = {}
        episode_splits = {}
        from utils.live_progress import emit as progress_event
        for row_index, row in enumerate(self.rows, 1):
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError(f"Invalid manifest split: {row['split']}")
            ep_key = (row["sensor_configuration"], row["sequence_id"], row["episode_index"])
            if episode_splits.setdefault(ep_key, row["split"]) != row["split"]:
                raise ValueError(f"Episode crosses splits: {ep_key}")
            previous_split = sequence_splits.setdefault(row["sequence_id"], row["split"])
            if self.split_unit == "sequence" and previous_split != row["split"]:
                raise ValueError(f"Sequence crosses splits: {row['sequence_id']}")
            if row["dynamics5"] != dynamics5(row["contact_phase"], row["sudden_event"]):
                raise ValueError(f"Manifest dynamics differs from source labels: {row['sample_id']}")
            if row["sample_id"] != sample_id(row["sensor_configuration"], row["sequence_id"],
                                              row["episode_index"], row["anchor_frame_position"]):
                raise ValueError("Manifest sample ID does not match anchor identity")
            ep = ep_lookup.get((row["sensor_configuration"], row["sequence_id"], row["episode_index"]))
            anchor = int(row["anchor_frame_position"])
            if ep is None or anchor >= ep.length or anchor < 19 or anchor - window_size + 1 < 0:
                raise ValueError(f"Manifest anchor cannot be located: {row['sample_id']}")
            position = (ep_order[(ep.sensor_type, ep.sequence_id, ep.episode_index)], anchor)
            if position <= prior_position:
                raise ValueError("Manifest rows must follow source episode and anchor order")
            prior_position = position
            if int(ep.frames[anchor]) != row["source_frame_index"] or abs(float(ep.timestamps[anchor])-row["timestamp_anchor"]) > 1e-3:
                raise ValueError(f"Manifest source changed: {row['sample_id']}")
            self._locations.append((ep, anchor-window_size+1))
            counts[id(ep)] += 1
            if row_index % 5000 == 0 or row_index == len(self.rows):
                progress_event("Manifest Loading & Validation", current=row_index,
                               total=len(self.rows), detail="anchors validated")
        # Explicit manifest order may not be episode-contiguous; samplers use _locate.
        self._ends = list(np.cumsum([counts[id(ep)] for ep in self.episodes]))
        if self.split_unit == "episode" and "parent_manifest_id" in self.rows[0]:
            from utils.episode_split import episode_manifest_identity
            parents = {r["parent_manifest_id"] for r in self.rows}
            seeds = {r["split_seed"] for r in self.rows}
            if len(parents) != 1 or len(seeds) != 1 or self.manifest_id != episode_manifest_identity(
                    next(iter(parents)), int(next(iter(seeds))), episode_splits):
                raise ValueError("Episode manifest identity differs from actual episode assignment")

    def __len__(self):
        return len(self.rows)

    def _locate(self, index):
        return self._locations[index]

    def __getitem__(self, index):
        ep, start = self._locate(index)
        offsets = np.arange(start, start+self.window_size, dtype=np.int64)
        force, pressure = self.source._sensor_rows(ep, offsets)
        images, pts = self.source._images(ep, offsets)
        row = self.rows[index]
        if hasattr(self.source, "_image_tensor"):
            tactile = self.source._image_tensor(images)
        else:
            tactile = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
            if getattr(self.source, "image_dtype", "float32") != "uint8":
                tactile = tactile.float().div_(255)
        if hasattr(self.source, "_sensor_tensors"):
            force, pressure = self.source._sensor_tensors(force, pressure)
        else:
            force, pressure = torch.from_numpy(force), torch.from_numpy(pressure)
        return {"tactile": tactile,
                "force": force, "pressure": pressure,
                "metadata": {"sample_id": row["sample_id"], "dynamics5": row["dynamics5"],
                             "sensor_type": ep.sensor_type, "sequence_id": ep.sequence_id,
                             "episode_index": ep.episode_index, "frame_indices": torch.from_numpy(ep.frames[offsets].copy()),
                             "timestamps": torch.from_numpy(ep.timestamps[offsets].copy()),
                             "video_timestamps": torch.from_numpy(ep.video_timestamps[offsets].copy()),
                             "video_frame_indices": torch.from_numpy(ep.video_frame_indices[offsets].copy()),
                             "video_pts": torch.from_numpy(pts)}}

    @property
    def invalid_sequences(self):
        return self.source.invalid_sequences

    def _sensor_rows(self, ep, offsets):
        return self.source._sensor_rows(ep, offsets)
