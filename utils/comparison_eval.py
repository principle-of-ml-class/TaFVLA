"""Shared-gallery retrieval and tactile-only ridge force probe."""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.dynamics import DYNAMICS, ManifestDataset


def gallery_selection(rows: list[dict], split: str, *, seed: int, max_pairs: int,
                      min_gap: int = 20) -> list[str]:
    """Choose episode-balanced IDs once, independently of model window length."""
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[(row["sensor_configuration"], row["sequence_id"], row["episode_index"])].append(row)
    rng = random.Random(seed)
    frequency = Counter(r["dynamics5"] for r in rows if r["split"] == split)
    queues = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda x: x["anchor_frame_position"])
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        rare_first = sorted(shuffled, key=lambda r: frequency[r["dynamics5"]])
        options = []
        for candidates in (ordered, rare_first):
            selected = []
            for row in candidates:
                if all(abs(row["anchor_frame_position"]-prior["anchor_frame_position"]) >= min_gap for prior in selected):
                    selected.append(row)
            options.append(selected)
        spaced = max(options, key=lambda picked: (len(picked), len({r["dynamics5"] for r in picked}),
                                                  sum(r["dynamics5"] == "sudden_event" for r in picked)))
        rng.shuffle(spaced)
        queues.append(spaced)
    rng.shuffle(queues)
    selected = []
    while queues and len(selected) < max_pairs:
        next_queues = []
        for queue in queues:
            if queue and len(selected) < max_pairs:
                selected.append(queue.pop())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return [r["sample_id"] for r in selected]


def save_gallery_plan(rows: list[dict], path: Path, *, seed: int, max_pairs: int,
                      min_gap: int = 20) -> dict:
    if max_pairs < 2 or min_gap < 20:
        raise ValueError("Comparison gallery requires max_pairs >= 2 and min_gap >= 20")
    plan = {split: gallery_selection(rows, split, seed=seed, max_pairs=max_pairs, min_gap=min_gap)
            for split in ("validation", "test")}
    for split in ("validation", "test"):
        if len(plan[split]) < 2:
            raise ValueError(f"{split} gallery has {len(plan[split])} pair(s); at least two are required. "
                             "Add sequences/episodes or review the 20-frame spacing and split.")
    plan.update(seed=seed, max_pairs=max_pairs, min_gap=min_gap,
                manifest_id=rows[0]["manifest_id"])
    lookup = {r["sample_id"]: r for r in rows}
    plan["support"] = {}
    for split in ("validation", "test"):
        selected = [lookup[s] for s in plan[split]]
        plan["support"][split] = {group: {
            "queries": sum(r["dynamics5"] == group for r in selected),
            "episodes": len({(r["sequence_id"], r["episode_index"]) for r in selected if r["dynamics5"] == group}),
            "sequences": len({r["sequence_id"] for r in selected if r["dynamics5"] == group}),
        } for group in DYNAMICS}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError(f"Existing gallery plan differs: {path}")
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def retrieval_by_dynamics(similarity: torch.Tensor, rows: list[dict]) -> tuple[list[dict], dict]:
    """Rank each query against the entire gallery, then group only query rows."""
    count = len(rows)
    if similarity.shape != (count, count):
        raise ValueError("Similarity shape must match common gallery")
    ranking = torch.argsort(similarity, dim=1, descending=True, stable=True)
    ranks = torch.argmax((ranking == torch.arange(count)[:, None]).long(), dim=1) + 1
    ties = (similarity == similarity.diagonal()[:, None]).sum(1)
    query = []
    for i, row in enumerate(rows):
        rank = int(ranks[i])
        query.append({"sample_id": row["sample_id"], "contact_state": row["dynamics5"],
                      "rank": rank, "recall_at_1": int(rank <= 1), "recall_at_5": int(rank <= 5),
                      "mrr": 1/rank, "positive_tie_count": int(ties[i]),
                      "sensor_configuration": row["sensor_configuration"],
                      "sequence_id": row["sequence_id"], "episode_index": row["episode_index"]})
    support = {}
    for group in ("all", *DYNAMICS):
        part = query if group == "all" else [r for r in query if r["contact_state"] == group]
        support[group] = {"queries": len(part), "episodes": len({(r["sequence_id"], r["episode_index"]) for r in part}),
                          "sequences": len({r["sequence_id"] for r in part}), "gallery_size": count,
                          "recall_at_1": float(np.mean([r["recall_at_1"] for r in part])) if part else None,
                          "recall_at_5": float(np.mean([r["recall_at_5"] for r in part])) if part else None,
                          "mrr": float(np.mean([r["mrr"] for r in part])) if part else None}
    return query, support


@torch.inference_mode()
def evaluate_retrieval(model, dataset: ManifestDataset, ids: list[str], *, device, batch_size: int) -> tuple[list[dict], dict]:
    lookup = {r["sample_id"]: i for i, r in enumerate(dataset.rows)}
    if len(ids) != len(set(ids)) or not set(ids) <= lookup.keys():
        raise ValueError("Gallery has duplicate or unknown sample IDs")
    rows = [dataset.rows[lookup[s]] for s in ids]
    loader = DataLoader(Subset(dataset, [lookup[s] for s in ids]), batch_size=batch_size, shuffle=False)
    model.eval()
    zs, cs, codes = [], [], []
    started = time.perf_counter()
    for batch in loader:
        output = model(tactile=batch["tactile"].to(device), force=batch["force"].to(device),
                       pressure=batch["pressure"].to(device))
        zs.append(output["z_tactile"].float().cpu())
        cs.append(output["c_force"].float().cpu())
        codes.extend(zip(output["codebook_indices"]["pressure"].cpu().tolist(),
                         output["codebook_indices"]["force"].cpu().tolist()))
    elapsed = time.perf_counter() - started
    similarity = F.normalize(torch.cat(zs), dim=1) @ F.normalize(torch.cat(cs), dim=1).T
    query, summary = retrieval_by_dynamics(similarity, rows)
    counts = Counter(codes)
    diagnostics = {"gallery_size": len(ids), "unique_code_pairs": len(counts),
                   "duplicate_code_pair_fraction": 1-len(counts)/len(ids),
                   "positive_tie_queries": sum(r["positive_tie_count"] > 1 for r in query),
                   "inference_seconds": elapsed, "inference_samples_per_second": len(ids)/elapsed}
    return query, {"groups": summary, "diagnostics": diagnostics}


def ridge_fit(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = x.double(), y.double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    weight = torch.linalg.solve(xc.T @ xc + alpha * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ yc)
    return weight, ym-xm@weight


def force_errors(prediction: np.ndarray, target: np.ndarray) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError("Force arrays must both have shape (samples, N, 6)")
    error = prediction-target
    def score(values):
        return {"mae": float(np.abs(values).sum()/values.size),
                "rmse": float(np.sqrt(np.square(values).sum()/values.size))}
    metrics = {"whole": score(error), "endpoint": score(error[:, -1:, :]),
               "force": score(error[..., :3]), "torque": score(error[..., 3:]),
               "endpoint_force": score(error[:, -1:, :3]),
               "endpoint_torque": score(error[:, -1:, 3:])}
    for axis in range(6):
        metrics[f"axis_{axis}"] = score(error[..., axis:axis+1])
        metrics[f"endpoint_axis_{axis}"] = score(error[:, -1:, axis:axis+1])
    return metrics


@torch.inference_mode()
def _embeddings(model, dataset, indices, device, batch_size):
    x, y = [], []
    model.eval()
    started = time.perf_counter()
    for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False):
        x.append(model.encode_tactile(batch["tactile"].to(device)).float().cpu())
        y.append(batch["force"].float())
    return torch.cat(x), torch.cat(y), time.perf_counter()-started


def evaluate_probe(model, dataset: ManifestDataset, *, device, batch_size: int,
                   alphas: tuple[float, ...] = (.01, .1, 1., 10., 100.)) -> tuple[list[dict], dict]:
    for p in model.tactile_encoder.parameters():
        p.requires_grad_(False)
    model.eval()
    subsets = {split: [i for i, r in enumerate(dataset.rows) if r["split"] == split]
               for split in ("train", "validation", "test")}
    if any(not values for values in subsets.values()):
        raise ValueError("All three splits are required for the probe")
    started = time.perf_counter()
    extracted = {split: _embeddings(model, dataset, ids, device, batch_size) for split, ids in subsets.items()}
    extraction_seconds = time.perf_counter() - started
    x_train, y_train, _ = extracted["train"]
    x_mean = x_train.double().mean(0)
    x_std = x_train.double().std(0, unbiased=False).clamp_min(1e-6)
    target_mean = model.force_mean.detach().cpu().double().reshape(1, 1, 6)
    target_std = model.force_std.detach().cpu().double().reshape(1, 1, 6)
    def prepare(split):
        x, y, _ = extracted[split]
        return (x.double()-x_mean)/x_std, ((y.double()-target_mean)/target_std).reshape(len(y), -1)
    xt, yt = prepare("train")
    xv, yv = prepare("validation")
    choices = []
    for alpha in alphas:
        w, b = ridge_fit(xt, yt, alpha)
        choices.append((float(((xv@w+b-yv)**2).mean()), alpha, w, b))
    _, alpha, w, b = min(choices, key=lambda item: (item[0], item[1]))
    rows = []
    groups = {}
    head_seconds = {}
    for split in ("validation", "test"):
        x, _ = prepare(split)
        y = extracted[split][1]
        head_started = time.perf_counter()
        prediction = ((x@w+b).reshape(-1, dataset.window_size, 6)*target_std+target_mean).numpy()
        head_seconds[split] = time.perf_counter()-head_started
        truth = y.numpy()
        for j, index in enumerate(subsets[split]):
            source = dataset.rows[index]
            err = force_errors(prediction[j:j+1], truth[j:j+1])
            rows.append({"sample_id": source["sample_id"], "sensor_configuration": source["sensor_configuration"],
                         "sequence_id": source["sequence_id"], "episode_index": source["episode_index"],
                         "contact_state": source["dynamics5"], "split": split,
                         "force_prediction_mae": err["whole"]["mae"], "force_prediction_rmse": err["whole"]["rmse"],
                         "endpoint_mae": err["endpoint"]["mae"], "endpoint_rmse": err["endpoint"]["rmse"],
                         "force_mae": err["force"]["mae"], "force_rmse": err["force"]["rmse"],
                         "torque_mae": err["torque"]["mae"], "torque_rmse": err["torque"]["rmse"],
                         "endpoint_force_mae": err["endpoint_force"]["mae"],
                         "endpoint_force_rmse": err["endpoint_force"]["rmse"],
                         "endpoint_torque_mae": err["endpoint_torque"]["mae"],
                         "endpoint_torque_rmse": err["endpoint_torque"]["rmse"],
                         **{f"axis_{axis}_{metric}": err[f"axis_{axis}"][metric]
                            for axis in range(6) for metric in ("mae", "rmse")},
                         **{f"endpoint_axis_{axis}_{metric}": err[f"endpoint_axis_{axis}"][metric]
                            for axis in range(6) for metric in ("mae", "rmse")}})
        groups[split] = {group: force_errors(prediction[[j for j, i in enumerate(subsets[split])
                                                     if group == "all" or dataset.rows[i]["dynamics5"] == group]],
                                             truth[[j for j, i in enumerate(subsets[split])
                                                    if group == "all" or dataset.rows[i]["dynamics5"] == group]])
                         for group in ("all", *DYNAMICS)
                         if any(group == "all" or dataset.rows[i]["dynamics5"] == group for i in subsets[split])}
    report = {"alpha": alpha, "alpha_candidates": alphas, "validation_normalized_mse": min(x[0] for x in choices),
              "probe_parameters": int(w.numel()+b.numel()), "feature_standardization": "train mean/std",
              "target_normalization": "train unique frame force mean/std from checkpoint",
              "extraction_seconds": extraction_seconds,
              "inference_seconds_by_split": {split: extracted[split][2]+head_seconds[split]
                                             for split in ("validation", "test")},
              "groups": groups}
    return rows, report


def write_rows(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".csv").exists():
        raise FileExistsError(f"Evaluation result already exists: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path)
    import csv
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
