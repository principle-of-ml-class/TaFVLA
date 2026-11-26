"""Repartition existing anchors without touching sensor data or video files."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils.dynamics import DYNAMICS, split_group_key


def episode_manifest_identity(parent_id: str, seed: int, assignments: dict) -> str:
    payload = {"protocol": "episode-repartition-v1", "parent_manifest_id": parent_id,
               "split_unit": "episode", "split_seed": seed,
               "episode_assignments": [[*key, value] for key, value in sorted(assignments.items())]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def manifest_support(table: pa.Table) -> dict:
    groups = table.group_by(["split", "sensor_configuration", "sequence_id", "episode_index",
                             "dynamics5"]).aggregate([("sample_id", "count")]).to_pylist()
    result = {}
    for split in ("train", "validation", "test"):
        part = [r for r in groups if r["split"] == split]
        count = lambda rows: sum(r["sample_id_count"] for r in rows)
        episodes = lambda rows: len({(r["sensor_configuration"], r["sequence_id"], r["episode_index"]) for r in rows})
        sequences = lambda rows: len({(r["sensor_configuration"], r["sequence_id"]) for r in rows})
        result[split] = {"windows": count(part), "episodes": episodes(part), "sequences": sequences(part),
                         "dynamics": {g: count([r for r in part if r["dynamics5"] == g]) for g in DYNAMICS},
                         "by_dynamics": {g: {"windows": count(p), "episodes": episodes(p), "sequences": sequences(p)}
                                         for g in DYNAMICS for p in [[r for r in part if r["dynamics5"] == g]]}}
    return result


def repartition_episode_manifest(original: Path, output: Path, *, seed: int = 42) -> dict:
    if original.resolve() == output.resolve():
        raise ValueError("Episode manifest must have a separate output path")
    table = pq.read_table(original)
    parent_ids = table["manifest_id"].unique().to_pylist()
    if len(parent_ids) != 1 or not len(table):
        raise ValueError("Original manifest must have one nonempty identity")
    if len(table["sample_id"].unique()) != len(table):
        raise ValueError("Original sample IDs are not unique")
    groups = table.group_by(["sensor_configuration", "sequence_id", "episode_index", "split"]).aggregate(
        [("sample_id", "count")]).to_pylist()
    assignments = {}; pools = defaultdict(list); counts = Counter()
    for row in groups:
        key = (row["sensor_configuration"], row["sequence_id"], row["episode_index"])
        if row["split"] not in {"train", "validation", "test"} or key in assignments:
            raise ValueError(f"Invalid split or original episode leakage: {key}")
        assignments[key] = row["split"]
        counts[row["split"]] += row["sample_id_count"]
        if row["split"] != "test":
            pools[key[:2]].append((key, row["sample_id_count"]))
    if not counts["train"] or not counts["validation"] or not counts["test"]:
        raise ValueError("Original train, validation and test must all be nonempty")
    fraction = counts["validation"] / (counts["train"] + counts["validation"])
    rng = random.Random(seed); options = {}; chosen = {}; singleton = []
    # Consider floor/ceil episode counts, select by window counts (not episode length assumptions).
    for seq, episodes in sorted(pools.items()):
        episodes.sort()
        if len(episodes) == 1:
            singleton.append(list(seq)); assignments[episodes[0][0]] = "train"
            continue
        n = len(episodes)
        sizes = {max(1, min(n-1, math.floor(n*fraction))), max(1, min(n-1, math.ceil(n*fraction)))}
        if sum(math.comb(n, size) for size in sizes) > 1_000_000:
            raise ValueError(f"Too many episode assignments to enumerate safely: {seq}, episodes={n}")
        candidates = []
        for size in sorted(sizes):
            for subset in itertools.combinations(episodes, size):
                candidates.append((sum(count for _, count in subset), tuple(key for key, _ in subset)))
        rng.shuffle(candidates)
        target = sum(count for _, count in episodes) * fraction
        options[seq] = (target, candidates)
        chosen[seq] = min(candidates, key=lambda item: abs(item[0]-target))
    # Deterministically reduce the total window-ratio error without sacrificing whole episodes.
    total = sum(item[0] for item in chosen.values())
    target_total = counts["validation"]
    while True:
        best = None; best_error = abs(total-target_total)
        for seq, (target, candidates) in options.items():
            for candidate in candidates:
                error = abs(total-chosen[seq][0]+candidate[0]-target_total)
                if error < best_error:
                    best_error = error; best = (seq, candidate)
        if best is None:
            break
        seq, candidate = best
        total += candidate[0]-chosen[seq][0]; chosen[seq] = candidate
    for seq, episodes in pools.items():
        validation = set(chosen[seq][1]) if seq in chosen else set()
        for key, _ in episodes:
            assignments[key] = "validation" if key in validation else "train"
    identity = episode_manifest_identity(parent_ids[0], seed, assignments)
    keys = zip(*(table[c].to_pylist() for c in ("sensor_configuration", "sequence_id", "episode_index")))
    splits = pa.array([assignments[key] for key in keys])
    updated = table.set_column(table.schema.get_field_index("split"), "split", splits)
    for name, values in (("manifest_id", pa.array([identity]*len(table))),
                         ("split_seed", pa.array([seed]*len(table), type=table["split_seed"].type)),
                         ("split_unit", pa.array(["episode"]*len(table))),
                         ("parent_manifest_id", pa.array([parent_ids[0]]*len(table)))):
        if name in updated.column_names:
            updated = updated.set_column(updated.schema.get_field_index(name), name, values)
        else:
            updated = updated.append_column(name, values)
    unchanged = [c for c in table.column_names if c not in {"split", "manifest_id", "split_seed", "split_unit", "parent_manifest_id"}]
    if not table.select(unchanged).equals(updated.select(unchanged)):
        raise AssertionError("Repartition changed source anchor data or row order")
    import pyarrow.compute as pc
    old_test = table.filter(pc.equal(table["split"], "test"))["sample_id"]
    new_test = updated.filter(pc.equal(updated["split"], "test"))["sample_id"]
    if not old_test.equals(new_test):
        raise AssertionError("Test sample IDs changed")
    shared = {k[:2] for k, v in assignments.items() if v == "train"} & {k[:2] for k, v in assignments.items() if v == "validation"}
    report = {"manifest_id": identity, "parent_manifest_id": parent_ids[0], "split_unit": "episode",
              "split_seed": seed, "algorithm": "floor/ceil episode count, window-balanced deterministic assignment v1",
              "original_manifest_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
              "original_support": manifest_support(table), "support": manifest_support(updated),
              "original_validation_fraction_in_train_validation": fraction,
              "validation_fraction_in_train_validation": total/(counts["train"]+counts["validation"]),
              "validation_window_count_difference": total-counts["validation"],
              "singleton_train_only_sequences": singleton, "shared_train_validation_sequences": len(shared),
              "episode_assignments": [{"sensor_configuration": k[0], "sequence_id": k[1], "episode_index": k[2],
                                       "split": v} for k, v in sorted(assignments.items())],
              "checks": {"sample_ids_and_row_order_preserved": True, "source_columns_unchanged": True,
                         "test_ids_and_order_preserved": True, "episode_overlap_count": 0}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not pq.read_table(output).equals(updated):
            raise FileExistsError(f"Existing episode manifest differs: {output}")
    else:
        tmp = output.with_suffix(".parquet.tmp")
        pq.write_table(updated, tmp); tmp.replace(output)
    report_path = output.with_suffix(".json")
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        raise FileExistsError(f"Existing episode report differs: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prepare_episode_assets(manifest: Path, *, original_gallery: Path, config_name: str) -> dict:
    """Use the production loader and normalization path; no image decoding or training."""
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from dataset.taf_dataset import TaFDataset
    from utils.common import validate_config
    from utils.config import register_config_schema
    from utils.data import compute_normalization
    from utils.dynamics import ManifestDataset
    from utils.comparison_eval import save_gallery_plan
    root = Path(__file__).resolve().parents[1]
    register_config_schema()
    with initialize_config_dir(config_dir=str(root/"config"), version_base="1.3"):
        cfg = compose(config_name=config_name)
    validate_config(cfg, require_data_root=True)
    if Path(cfg.comparison.manifest).resolve() != manifest.resolve() or cfg.data.split_unit != "episode":
        raise ValueError("Preparation config must reference this episode manifest")
    data = cfg.data
    sensors = pq.read_table(manifest, columns=["sensor_configuration"])["sensor_configuration"].unique().to_pylist()
    source = TaFDataset(data.root_dir, window_size=1, stride=1, image_size=None, image_dtype="uint8",
                        sensor_types=sensors, show_initialization_progress=False,
                        timestamp_cache_enabled=bool(data.timestamp_cache_enabled), timestamp_cache_dir=data.timestamp_cache_dir,
                        sensor_mmap_enabled=bool(data.sensor_mmap_enabled), sensor_mmap_dir=data.sensor_mmap_dir)
    dataset = ManifestDataset(source, manifest, int(data.window_size), expected_split_unit="episode")
    if int(dataset.rows[0]["split_seed"]) != int(data.split_seed):
        raise ValueError("Preparation config split_seed differs from manifest")
    old_plan = json.loads(original_gallery.read_text())
    parent = dataset.rows[0]["parent_manifest_id"]
    if old_plan["manifest_id"] != parent:
        raise ValueError("Preserved test gallery must belong to the original manifest")
    plan = save_gallery_plan(dataset.rows, manifest.with_name("gallery_plan.json"),
                             seed=int(cfg.comparison.gallery_seed), max_pairs=int(cfg.comparison.gallery_max_pairs),
                             min_gap=int(cfg.comparison.min_anchor_gap), preserved_test_ids=old_plan["test"])
    train = [i for i, r in enumerate(dataset.rows) if r["split"] == "train"]
    groups = sorted({split_group_key(dataset.rows[i], "episode") for i in train})
    ep_ids = {id(ep): i for i, ep in enumerate(dataset.episodes)}
    train_eps = {ep_ids[id(dataset._locate(i)[0])] for i in train}
    path = manifest.with_name(f"normalization_{dataset.manifest_id[:8]}.pt")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["manifest_id"] != dataset.manifest_id or saved["train_groups"] != groups:
            raise ValueError("Existing normalization has different train groups/manifest identity")
        normal = saved["normalization"]
    else:
        normal = compute_normalization(dataset, train_eps, int(data.normalization_chunk_frames), float(data.normalization_eps),
                                       selected_indices=train, normalization_window_size=max(cfg.comparison.window_sizes))
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"manifest_id": dataset.manifest_id, "split_unit": "episode", "train_groups": groups,
                    "normalization_window_size": max(cfg.comparison.window_sizes), "normalization": normal}, tmp)
        tmp.replace(path)
    if any(not torch.isfinite(v).all() for v in normal.values()) or any((normal[k] <= 0).any() for k in ("pressure_std", "force_std")):
        raise ValueError("Invalid train normalization")
    (manifest.parent / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    result = {"manifest_id": dataset.manifest_id, "normalization": str(path), "train_episode_groups": len(groups),
              "normalization_window_size": max(cfg.comparison.window_sizes), "validation_frames_used": 0,
              "gallery_sizes": {s: len(plan[s]) for s in ("validation", "test")},
              "gallery_id": hashlib.sha256(manifest.with_name("gallery_plan.json").read_bytes()).hexdigest(),
              "test_gallery_ids_and_order_preserved": plan["test"] == old_plan["test"],
              "gallery_support": plan["support"], "normalization_values": {k: v.tolist() for k, v in normal.items()}}
    (manifest.parent / "assets.json").write_text(json.dumps(result, indent=2))
    return result
