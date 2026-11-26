"""UI-facing episode and result readers; no classification or model inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def episode_labels(prepared_dir, episode):
    directory = Path(prepared_dir)
    table = pq.ParquetFile(directory/episode["label_file"]).read()
    rows = table.to_pydict()
    if len(rows["frame_position"]) != episode["frames"] or not np.array_equal(rows["frame_position"], np.arange(episode["frames"])):
        raise ValueError("Prepared episode labels have missing or mismatched positions")
    for key in ("sensor_configuration", "sequence_id", "episode_index", "split"):
        if set(rows[key]) != {episode[key]}:
            raise ValueError("Prepared episode labels belong to a different episode/split")
    if set(rows["manifest_id"]) != {episode["manifest_id"]} or set(rows["label_schema_version"]) != {"phase4-frame-labels-v1"}:
        raise ValueError("Episode labels have a different manifest/schema identity")
    return rows


def result_catalog(results_dir, manifest_id):
    records = []
    for path in sorted(Path(results_dir).rglob("endpoint/summary.json")):
        summary = json.loads(path.read_text())
        if summary["manifest_id"] == manifest_id:
            records.append({"directory": str(path.parent), "summary": summary})
    return records


def prediction_episode(result_dir, episode, method):
    result_dir = Path(result_dir)
    index = pq.read_table(result_dir/"prediction_index.parquet", filters=[
        ("sensor_configuration", "=", episode["sensor_configuration"]),
        ("sequence_id", "=", episode["sequence_id"]), ("episode_index", "=", episode["episode_index"]),
        ("split", "=", episode["split"]), ("method", "=", method)])
    rows = []
    for file in sorted(set(index["file"].to_pylist())):
        rows.extend(row for row in pq.ParquetFile(result_dir/file).read().to_pylist()
                    if row["sensor_configuration"] == episode["sensor_configuration"] and
                    row["sequence_id"] == episode["sequence_id"] and row["episode_index"] == episode["episode_index"] and
                    row["split"] == episode["split"] and row["method"] == method)
    return sorted(rows, key=lambda row: row["timestamp_anchor"])
