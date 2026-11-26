"""Raw episodes -> split -> train calibration -> labels -> common experiment assets."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from utils.phase_classifier import PHASES, VERSION, PhaseConfig, classify_episode, fit_sensor_calibration, identity

SCHEMA = "phase4-manifest-v1"


@dataclass
class PhasePipelineConfig:
    data_root: str = "/ssdg/spl_yeongyoo/taf_dataset"
    sensors: list[str] = field(default_factory=lambda: ["custom_6x6_markers"])
    timestamp_cache_dir: str | None = None
    sensor_mmap_dir: str = "outputs/sensor_mmap"
    split_reference: str | None = "auto"
    split_seed: int = 42
    validation_fraction: float = .16778
    test_fraction: float = .1
    prepared_dir: str = "metadata/phase4/v1"
    results_dir: str = "outputs/phase4/v1"
    training_config: str = "taf_adapter_sparsh_episode"
    windows: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 20])
    seed: int = 42
    eval_batch_size: int = 16
    gallery_size: int = 256
    gallery_gap: int = 20
    gallery_seed: int = 42
    ridge_alphas: list[float] = field(default_factory=lambda: [.01, .1, 1., 10., 100.])
    classifier: PhaseConfig = field(default_factory=PhaseConfig)


def load_settings(path="config/phase4_pipeline.yaml", overrides=()):
    cfg = OmegaConf.merge(OmegaConf.structured(PhasePipelineConfig), OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.to_object(cfg)
    if cfg.windows != [1, 3, 5, 7, 10, 20] or cfg.seed != 42 or cfg.gallery_gap < 20 or cfg.gallery_size < 2:
        raise ValueError("Phase4 requires six common N, seed42, gallery gap>=20 and gallery size>=2")
    if cfg.eval_batch_size < 1 or not cfg.ridge_alphas or min(cfg.ridge_alphas) <= 0:
        raise ValueError("Invalid evaluation options")
    return cfg


def source_for_settings(cfg):
    from dataset.taf_dataset import TaFDataset
    return TaFDataset(cfg.data_root, window_size=1, stride=1, image_size=None, image_dtype="uint8",
                      sensor_types=cfg.sensors or None, sensor_mmap_enabled=True,
                      sensor_mmap_dir=cfg.sensor_mmap_dir, timestamp_cache_dir=cfg.timestamp_cache_dir,
                      show_initialization_progress=False)


def prepare_from_settings(cfg):
    try:
        source = source_for_settings(cfg)
    except Exception as error:
        parent = Path(cfg.prepared_dir).parent; parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".phase4_prepare_", dir=parent))
        (stage/"failure.json").write_text(json.dumps({"stage": "raw initialization/alignment", "error": repr(error),
                                                      "intended_output": str(Path(cfg.prepared_dir).resolve())}, indent=2))
        raise RuntimeError(f"Raw dataset initialization failed; diagnostics: {stage/'failure.json'}") from error
    return prepare_phase4(source, cfg)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def episode_key(ep):
    return ep.sensor_type, ep.sequence_id, int(ep.episode_index)


def split_reference_path(cfg):
    if cfg.split_reference == "auto":
        return next((p for p in map(Path, ("metadata/episode_split/manifest.json", "metadata/episode_split/manifest.parquet",
                                           "metadata/manifest.parquet")) if p.is_file()), None)
    return Path(cfg.split_reference) if cfg.split_reference is not None else None


def assert_settings_prepared(cfg, report):
    reference = split_reference_path(cfg)
    expected = {"classifier_id": cfg.classifier.configuration_id, "split_seed": cfg.split_seed,
                "validation_fraction": cfg.validation_fraction, "test_fraction": cfg.test_fraction,
                "gallery": [cfg.gallery_size, cfg.gallery_gap, cfg.gallery_seed],
                "reference_id": file_hash(reference) if reference is not None else None}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Preparation settings/reference changed; prepare into a new prepared_dir")
    if (sorted(cfg.sensors) != sorted(report["settings"]["sensors"]) or
            Path(cfg.data_root).resolve() != Path(report["settings"]["data_root"]).resolve()):
        raise ValueError("Prepared sensor/root selection differs from pipeline settings")


def raw_fingerprint(source):
    records = []
    for ep in source.episodes:
        row = dict(sensor_configuration=ep.sensor_type, sequence_id=ep.sequence_id,
                   episode_index=int(ep.episode_index), length=ep.length,
                   frames_sha256=hashlib.sha256(ep.frames.tobytes()).hexdigest(),
                   timestamps_sha256=hashlib.sha256(ep.timestamps.tobytes()).hexdigest(),
                   video_timestamps_sha256=hashlib.sha256(ep.video_timestamps.tobytes()).hexdigest(),
                   video_frame_indices_sha256=hashlib.sha256(ep.video_frame_indices.tobytes()).hexdigest())
        for name in ("parquet", "video"):
            path = getattr(ep, name, None)
            if path is not None:
                stat = Path(path).stat()
                row[name] = {"path": str(Path(path).resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        records.append(row)
    return records


def load_prepared(path, *, verify=True):
    path = Path(path); report = json.loads((path/"summary.json").read_text())
    if report.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported prepared schema")
    if verify:
        for file, digest in report["artifacts"].items():
            if file_hash(path/file) != digest:
                raise ValueError(f"Prepared artifact changed: {file}")
    return report


def _support(table, field):
    groups = table.group_by(["split", "sensor_configuration", "sequence_id", "episode_index", field]).aggregate(
        [(field, "count")]).to_pylist()
    result = {}
    for split in ("train", "validation", "test"):
        rows = [r for r in groups if r["split"] == split]
        counts = Counter()
        for r in rows:
            counts[r[field]] += r[field+"_count"]
        result[split] = {"count": sum(counts.values()), "phase": {p: counts[p] for p in PHASES},
                         "unknown_fraction": counts["unknown"]/sum(counts.values()) if rows else None,
                         "episodes": len({(r["sensor_configuration"], r["sequence_id"], r["episode_index"]) for r in rows}),
                         "sequences": len({(r["sensor_configuration"], r["sequence_id"]) for r in rows}),
                         "sensors": {s: sum(r[field+"_count"] for r in rows if r["sensor_configuration"] == s)
                                     for s in sorted({r["sensor_configuration"] for r in rows})}}
    return result


def prepare_phase4(source, cfg: PhasePipelineConfig):
    """No image decoding. Writes into a staging directory, publishes only a complete bundle."""
    from utils.episode_split import assign_raw_episodes
    from utils.dynamics import ManifestDataset, sample_id, split_group_key
    from utils.data import compute_normalization
    from utils.comparison_eval import save_gallery_plan
    output = Path(cfg.prepared_dir)
    records = raw_fingerprint(source)
    reference = []; reference_id = None
    ref = split_reference_path(cfg)
    if ref is not None:
        if not ref.is_file():
            raise FileNotFoundError(f"Configured split reference is missing: {ref}; use split_reference=null to create a split")
        if ref.suffix == ".json":
            saved = json.loads(ref.read_text())
            reference = saved["episode_assignments"] if "episode_assignments" in saved else saved["episodes"]
        else:
            table = pq.read_table(ref)
            episode_split = "split_unit" in table.column_names and table["split_unit"].unique().to_pylist() == ["episode"]
            reference = table.select(["sensor_configuration", "sequence_id", "episode_index", "split"]).to_pylist()
            if not episode_split:
                reference = [r for r in reference if r["split"] == "test"]
        reference_id = file_hash(ref)
    dataset_id = identity(records)
    classifier_id = cfg.classifier.configuration_id
    recipe = {"schema_version": SCHEMA, "dataset_id": dataset_id, "classifier_id": classifier_id,
              "reference_id": reference_id, "split_seed": cfg.split_seed, "validation_fraction": cfg.validation_fraction,
              "test_fraction": cfg.test_fraction, "Nmax": 20,
              "gallery": [cfg.gallery_size, cfg.gallery_gap, cfg.gallery_seed]}
    recipe_id = identity(recipe)
    if output.exists():
        previous = load_prepared(output)
        if previous["recipe_id"] != recipe_id:
            raise FileExistsError("Prepared dataset/config changed; select a new prepared_dir")
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".phase4_prepare_", dir=output.parent))
    current = None
    try:
        assignments = assign_raw_episodes(records, previous=reference, seed=cfg.split_seed,
                                           validation_fraction=cfg.validation_fraction, test_fraction=cfg.test_fraction)
        print(f"Phase4: split fixed for {len(records)} raw episodes; fitting Train-only calibration", flush=True)
        split_rows = [{**r, "split": assignments[(r["sensor_configuration"], r["sequence_id"], r["episode_index"])]} for r in records]
        split_id = identity(split_rows)
        (stage/"episode_split.json").write_text(json.dumps({"split_id": split_id, "dataset_id": dataset_id, "episodes": split_rows}, indent=2))
        sensors = sorted({ep.sensor_type for ep in source.episodes}); calibration = {}
        for sensor in sensors:
            current = {"sensor": sensor, "stage": "Train calibration"}
            print(f"Phase4: calibrating sensor {sensor}", flush=True)
            def episodes():
                nonlocal current
                for ep in source.episodes:
                    if ep.sensor_type == sensor and assignments[episode_key(ep)] == "train":
                        current = {"sensor": sensor, "sequence": ep.sequence_id, "episode": ep.episode_index,
                                   "stage": "Train calibration"}
                        f, p = source._sensor_rows(ep, np.arange(ep.length))
                        yield "train", ep.timestamps, f, p
            calibration[sensor] = fit_sensor_calibration(episodes(), cfg.classifier)
        calibration_id = identity(calibration)
        label_id = identity({"classifier": classifier_id, "calibration": calibration_id})
        manifest_payload = {**recipe, "split_id": split_id, "calibration_id": calibration_id}
        (stage/"manifest_identity.json").write_text(json.dumps(manifest_payload, indent=2))
        (stage/"calibration.json").write_text(json.dumps({"calibration_id": calibration_id, "by_sensor": calibration}, indent=2))
        labels_dir = stage/"labels"; labels_dir.mkdir()
        summaries = []; frame_counts = Counter()
        # Labels are partitioned by episode; only one episode's dense values is resident.
        with pq.ParquetWriter(stage/"manifest.parquet", _manifest_schema()) as writer:
            manifest_id = identity({**recipe, "split_id": split_id, "calibration_id": calibration_id})
            for number, ep in enumerate(source.episodes):
                current = episode_key(ep)
                f, p = source._sensor_rows(ep, np.arange(ep.length))
                arrays, audit = classify_episode(ep.timestamps, f, p, calibration[ep.sensor_type], cfg.classifier)
                audit.update(sensor_calibration_id=audit["calibration_id"], calibration_id=calibration_id)
                split = assignments[current]
                common = {"sensor_configuration": ep.sensor_type, "sequence_id": ep.sequence_id,
                          "episode_index": int(ep.episode_index), "split": split}
                table = pa.table({**{k: [v]*ep.length for k, v in common.items()},
                                  "frame_position": np.arange(ep.length), "source_frame_index": ep.frames,
                                  "timestamp": ep.timestamps, **{k: v.tolist() for k, v in arrays.items()},
                                  "calibration_id": [calibration_id]*ep.length,
                                  "sensor_calibration_id": [audit["sensor_calibration_id"]]*ep.length,
                                  "manifest_id": [manifest_id]*ep.length, "dataset_id": [dataset_id]*ep.length,
                                  "split_id": [split_id]*ep.length,
                                  "classifier_configuration_hash": [classifier_id]*ep.length,
                                  "label_schema_version": ["phase4-frame-labels-v1"]*ep.length,
                                  "classifier_version": [VERSION]*ep.length})
                filename = f"labels/episode_{number:06d}.parquet"
                pq.write_table(table, stage/filename)
                (stage/filename).with_suffix(".json").write_text(json.dumps(audit, indent=2))
                summaries.append({**common, "frames": ep.length, "anchors": max(0, ep.length-19),
                                  "manifest_id": manifest_id,
                                  "label_file": filename, "preprocessing": audit, "source_provenance": records[number],
                                  "phase_counts": dict(Counter(arrays["phase"])),
                                  "excluded_insufficient_history": min(19, ep.length)})
                frame_counts.update({(split, ep.sensor_type, str(phase)): int(count)
                                     for phase, count in Counter(arrays["phase"]).items()})
                rows = [{**common, "sample_id": sample_id(*current, anchor), "anchor_frame_position": anchor,
                         "source_frame_index": int(ep.frames[anchor]), "timestamp_anchor": float(ep.timestamps[anchor]),
                         "phase": str(arrays["phase"][anchor]), "schema_version": SCHEMA,
                         "manifest_id": manifest_id, "dataset_id": dataset_id, "split_id": split_id,
                         "calibration_id": calibration_id, "classifier_configuration_hash": classifier_id,
                         "label_sha256": label_id,
                         "split_unit": "episode", "split_seed": cfg.split_seed} for anchor in range(19, ep.length)]
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=_manifest_schema()))
                if (number+1) % 25 == 0 or number+1 == len(source.episodes):
                    print(f"Phase4: labels {number+1}/{len(source.episodes)} episodes", flush=True)
        (stage/"episodes.json").write_text(json.dumps(summaries, indent=2))
        dataset = ManifestDataset(source, stage/"manifest.parquet", 20, expected_split_unit="episode")
        train = [i for i, r in enumerate(dataset.rows) if r["split"] == "train"]
        if any(not any(r["split"] == split for r in dataset.rows) for split in ("train", "validation", "test")):
            raise ValueError("An entire split has no common Nmax=20 anchors")
        ep_ids = {id(ep): i for i, ep in enumerate(source.episodes)}
        train_eps = {ep_ids[id(dataset._locate(i)[0])] for i in train}
        normal = compute_normalization(dataset, train_eps, 8192, 1e-6, selected_indices=train, normalization_window_size=20)
        normal_path = f"normalization_{manifest_id[:8]}.pt"
        torch.save({"manifest_id": manifest_id, "train_groups": sorted({split_group_key(dataset.rows[i], "episode") for i in train}),
                    "normalization_window_size": 20, "normalization": normal}, stage/normal_path)
        gallery = save_gallery_plan(dataset.rows, stage/"gallery_plan.json", seed=cfg.gallery_seed,
                                    max_pairs=cfg.gallery_size, min_gap=cfg.gallery_gap)
        support = _support(pq.read_table(stage/"manifest.parquet"), "phase")
        report = {**recipe, "recipe_id": recipe_id, "manifest_id": manifest_id, "split_id": split_id,
                  "calibration_id": calibration_id, "episodes": len(summaries), "anchor_support": support,
                  "frame_support": [{"split": k[0], "sensor": k[1], "phase": k[2], "frames": v} for k, v in sorted(frame_counts.items())],
                  "exclusions": {"insufficient_history": sum(r["excluded_insufficient_history"] for r in summaries)},
                  "failures": [], "normalization_file": normal_path,
                  "gallery_size": {s: len(gallery[s]) for s in ("validation", "test")},
                  "settings": asdict(cfg), "offline_labels": True,
                  "sources_file": "docs/phase4_sources.json", "reference_source": str(ref) if ref is not None else "generated from raw episodes"}
        if identity(raw_fingerprint(source)) != dataset_id or (file_hash(ref) if ref is not None else None) != reference_id:
            raise ValueError("Raw source or split reference changed during preparation")
        report["artifacts"] = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage/"summary.json").write_text(json.dumps(report, indent=2))
        stage.rename(output)
        return report
    except Exception as error:
        (stage/"failure.json").write_text(json.dumps({"episode": current, "error": repr(error), "recipe": recipe,
                                                      "intended_output": str(output.resolve())}, indent=2))
        raise RuntimeError(f"Preparation failed; original data untouched, diagnostics: {stage/'failure.json'}") from error


def _manifest_schema():
    fields = {"sample_id": pa.string(), "sensor_configuration": pa.string(), "sequence_id": pa.string(),
              "episode_index": pa.int64(), "split": pa.string(), "anchor_frame_position": pa.int64(),
              "source_frame_index": pa.int64(), "timestamp_anchor": pa.float64(), "phase": pa.string(),
              "schema_version": pa.string(), "manifest_id": pa.string(), "dataset_id": pa.string(),
              "split_id": pa.string(), "calibration_id": pa.string(), "classifier_configuration_hash": pa.string(),
              "label_sha256": pa.string(), "split_unit": pa.string(), "split_seed": pa.int64()}
    return pa.schema(fields)
