"""Bounded-memory endpoint Ridge, Train-observed VQ decoding and mean evaluation."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from utils.phase_classifier import PHASES, identity

METHODS = ("ridge", "vq_decoding", "training_mean", "gt_vq_reconstruction")


class RidgeStatistics:
    """Sufficient statistics; memory depends on latent width, not dataset length."""
    def __init__(self, width, targets=150):
        self.n = 0; self.sx = np.zeros(width); self.sy = np.zeros(targets)
        self.xx = np.zeros((width, width)); self.xy = np.zeros((width, targets)); self.yy = np.zeros(targets)

    def add(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite Train embeddings or targets")
        self.n += len(x); self.sx += x.sum(0); self.sy += y.sum(0)
        self.xx += x.T@x; self.xy += x.T@y; self.yy += (y*y).sum(0)

    def fit(self, alpha):
        if self.n < 2 or alpha <= 0:
            raise ValueError("Ridge needs >=2 Train endpoints and positive alpha")
        xm, ym = self.sx/self.n, self.sy/self.n
        xs = np.sqrt(np.maximum(np.diag(self.xx)/self.n-xm*xm, 0)).clip(1e-6)
        ys = np.sqrt(np.maximum(self.yy/self.n-ym*ym, 0)).clip(1e-6)
        gram = (self.xx-self.n*np.outer(xm, xm))/np.outer(xs, xs)
        cross = (self.xy-self.n*np.outer(xm, ym))/np.outer(xs, ys)
        return {"x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "weights": np.linalg.solve(gram+alpha*np.eye(len(xm)), cross), "alpha": alpha}


def ridge_predict(x, probe):
    return (((np.asarray(x, np.float64)-probe["x_mean"])/probe["x_std"])@probe["weights"])*probe["y_std"]+probe["y_mean"]


def endpoint_error(prediction, target, wrench_std):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 150:
        raise ValueError("Endpoint prediction/target must be [samples,150]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite endpoint predictions/targets")
    error = prediction-target
    std = np.asarray(wrench_std, np.float64).reshape(6)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid Train wrench normalization")
    return {"wrench_sse_standardized": ((error[:, :6]/std)**2).sum(1),
            "wrench_count": np.full(len(error), 6), "axis_sse": error[:, :6]**2,
            "axis_ae": abs(error[:, :6]), "pressure_sse": (error[:, 6:]**2).sum(1),
            "pressure_count": np.full(len(error), 144)}


class MetricSums:
    def __init__(self):
        self.n = 0; self.sse = 0.; self.elements = 0; self.pressure_sse = 0.; self.pressure_elements = 0
        self.axis_sse = np.zeros(6); self.axis_ae = np.zeros(6); self.episodes = set(); self.total_n = 0

    def add(self, row):
        self.n += 1; self.sse += row["wrench_sse_standardized"]; self.elements += row["wrench_count"]
        self.pressure_sse += row["pressure_sse"]; self.pressure_elements += row["pressure_count"]
        self.axis_sse += row["axis_sse"]; self.axis_ae += row["axis_ae"]
        self.episodes.add((row["sensor_configuration"], row["sequence_id"], row["episode_index"]))
        self.total_n += row.get("selected_N", row["N"])

    def result(self):
        return {"anchors": self.n, "episodes": len(self.episodes),
                "wrench_sse_standardized": self.sse, "wrench_elements": self.elements,
                "endpoint_standardized_wrench_rmse": float(np.sqrt(self.sse/self.elements)) if self.elements else None,
                "endpoint_raw_pressure_rmse": float(np.sqrt(self.pressure_sse/self.pressure_elements)) if self.pressure_elements else None,
                "axis_raw_mae": (self.axis_ae/self.n).tolist() if self.n else None,
                "axis_raw_rmse": np.sqrt(self.axis_sse/self.n).tolist() if self.n else None,
                "raw_force_rmse": float(np.sqrt(self.axis_sse[:3].sum()/(3*self.n))) if self.n else None,
                "raw_torque_rmse": float(np.sqrt(self.axis_sse[3:].sum()/(3*self.n))) if self.n else None,
                "raw_force_mae": float(self.axis_ae[:3].sum()/(3*self.n)) if self.n else None,
                "raw_torque_mae": float(self.axis_ae[3:].sum()/(3*self.n)) if self.n else None,
                "mean_selected_N": self.total_n/self.n if self.n else None}


def summarize_predictions(files):
    sums = defaultdict(MetricSums)
    for file in sorted(map(Path, files)):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                for phase in ("all", row["phase"]):
                    key = (row["method"], row["split"], row["manifest_id"], row["checkpoint_id"], row["N"], phase)
                    sums[key].add(row)
    return [{**dict(zip(("method", "split", "manifest_id", "checkpoint_id", "N", "phase"), key)), **value.result()}
            for key, value in sorted(sums.items())]


def build_prediction_index(files, root):
    """Small file-to-episode index lets the UI avoid scanning dense prediction arrays."""
    root = Path(root); rows = []
    columns = ["sensor_configuration", "sequence_id", "episode_index", "split", "method"]
    for file in sorted(map(Path, files)):
        table = pq.ParquetFile(file).read(columns=columns)
        groups = {tuple(row[k] for k in columns) for row in table.to_pylist()}
        rows.extend({**dict(zip(columns, key)), "file": str(file.relative_to(root))} for key in sorted(groups))
    pq.write_table(pa.Table.from_pylist(rows), root/"prediction_index.parquet")


@torch.inference_mode()
def evaluate_endpoints(model, dataset, output: Path, provenance: dict, *, device="cpu", batch_size=16,
                       alphas=(.01, .1, 1., 10., 100.)):
    """One extraction per split into memmaps; probe fit never reads Test targets."""
    from utils.phase4_data import file_hash
    output = Path(output)
    evaluation_id = identity({**provenance, "batch_size": batch_size, "alphas": list(alphas), "protocol": "endpoint-eval-v1"})
    if output.exists():
        summary = json.loads((output/"summary.json").read_text())
        if summary["evaluation_id"] != evaluation_id:
            raise FileExistsError("Evaluation identity changed; choose a new output path")
        for name, digest in summary["artifact_sha256"].items():
            if file_hash(output/name) != digest:
                raise ValueError(f"Completed evaluation artifact changed: {name}")
        return summary
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".endpoint_eval_", dir=output.parent))
    model.eval(); device = torch.device(device)
    subsets = {split: [i for i, r in enumerate(dataset.rows) if r["split"] == split]
               for split in ("train", "validation", "test")}
    if any(len(indices) < 2 for indices in subsets.values()):
        raise ValueError("Endpoint evaluation needs >=2 anchors in every split")
    pair_set = set(); arrays = {}; widths = None
    try:
        for split, indices in subsets.items():
            directory = stage/"cache"/split; directory.mkdir(parents=True)
            y = np.lib.format.open_memmap(directory/"targets.npy", mode="w+", dtype="float32", shape=(len(indices), 150))
            recon = np.lib.format.open_memmap(directory/"gt_reconstruction.npy", mode="w+", dtype="float32", shape=(len(indices), 150))
            x = None; cursor = 0
            for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0):
                image, f, p = (batch[k].to(device) for k in ("tactile", "force", "pressure"))
                z = model.encode_tactile(image).float().cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError("Non-finite tactile embedding")
                if x is None:
                    widths = z.shape[1]
                    x = np.lib.format.open_memmap(directory/"embeddings.npy", mode="w+", dtype="float32", shape=(len(indices), widths))
                p_norm, f_norm = model._normalize(p, f)
                encoded = model.force_encoder(p_norm, f_norm)
                if split == "train":
                    pair_set.update(zip(encoded["pressure_indices"].cpu().tolist(), encoded["force_indices"].cpu().tolist()))
                stop = cursor+len(z); x[cursor:stop] = z
                y[cursor:stop] = torch.cat((f[:, -1], p[:, -1].flatten(1)), dim=1).float().cpu().numpy()
                rf = encoded["reconstructed_force"]*model.force_std+model.force_mean
                rp = encoded["reconstructed_pressure"]*model.pressure_std+model.pressure_mean
                recon[cursor:stop] = torch.cat((rf[:, -1], rp[:, -1].flatten(1)), dim=1).float().cpu().numpy()
                cursor = stop
            for a in (x, y, recon):
                a.flush()
            arrays[split] = (x, y, recon)
        pairs = torch.tensor(sorted(pair_set), dtype=torch.long)
        bank = {**provenance, "N": dataset.window_size, "source_split": "train", "train_endpoint_count": len(subsets["train"]),
                "method": "train-observed code-pair decoding", "pairs": pairs.tolist()}
        bank["bank_id"] = identity(bank)
        (stage/"code_pair_bank.json").write_text(json.dumps(bank, indent=2))
        stats = RidgeStatistics(widths)
        x, y, _ = arrays["train"]
        statistics_chunk = max(4096, batch_size)
        prediction_chunk = max(512, batch_size)
        for start in range(0, len(x), statistics_chunk):
            stats.add(x[start:start+statistics_chunk], y[start:start+statistics_chunk])
        std = model.force_std.detach().cpu().numpy().reshape(6)
        candidates = []
        xv, yv, _ = arrays["validation"]
        for alpha in alphas:
            probe = stats.fit(alpha); sse = 0.
            for start in range(0, len(xv), statistics_chunk):
                pred = ridge_predict(xv[start:start+statistics_chunk], probe)
                sse += endpoint_error(pred, yv[start:start+statistics_chunk], std)["wrench_sse_standardized"].sum()
            candidates.append((float(sse/(len(xv)*6)), alpha, probe))
        _, alpha, probe = min(candidates, key=lambda value: (value[0], value[1]))
        np.savez(stage/"ridge_probe.npz", **probe)
        prediction_files = []
        for split in ("validation", "test"):
            x, y, recon = arrays[split]
            for method in METHODS:
                directory = stage/"predictions"/split/method; directory.mkdir(parents=True)
                for start in range(0, len(x), prediction_chunk):
                    stop = min(start+prediction_chunk, len(x)); target = np.asarray(y[start:stop])
                    if method == "ridge":
                        prediction = ridge_predict(x[start:stop], probe)
                    elif method == "training_mean":
                        prediction = np.broadcast_to(probe["y_mean"], target.shape)
                    elif method == "gt_vq_reconstruction":
                        prediction = recon[start:stop]
                    else:
                        decoded = model.decode_tactile_embedding(torch.as_tensor(np.array(x[start:stop]), device=device), pairs.to(device))
                        prediction = torch.cat((decoded["force"][:, -1], decoded["pressure"][:, -1].flatten(1)), dim=1).float().cpu().numpy()
                    errors = endpoint_error(prediction, target, std)
                    rows = []
                    for j, index in enumerate(subsets[split][start:stop]):
                        meta = dataset.rows[index]
                        rows.append({**provenance, **{k: meta[k] for k in ("sample_id", "sensor_configuration", "sequence_id", "episode_index", "timestamp_anchor", "anchor_frame_position", "phase")},
                                     "split": split, "method": method, "N": dataset.window_size,
                                     "actual_wrench": target[j, :6].tolist(), "predicted_wrench": prediction[j, :6].tolist(),
                                     "actual_pressure": target[j, 6:].tolist(), "predicted_pressure": prediction[j, 6:].tolist(),
                                     **{k: value[j].tolist() for k, value in errors.items()}})
                    file = directory/f"part_{start:09d}.parquet"; pq.write_table(pa.Table.from_pylist(rows), file)
                    prediction_files.append(file)
        metrics = summarize_predictions(prediction_files)
        build_prediction_index(prediction_files, stage)
        pq.write_table(pa.Table.from_pylist(metrics), stage/"metrics.parquet")
        summary = {"evaluation_id": evaluation_id, **provenance, "N": dataset.window_size,
                   "ridge_alpha": alpha, "alpha_validation_mse": [[mse, a] for mse, a, _ in candidates],
                   "bank_id": bank["bank_id"], "bank_pairs": len(pairs), "counts": {s: len(v) for s, v in subsets.items()},
                   "metrics": metrics, "target": "common endpoint: wrench6 + pressure144",
                   "decoding_method": "train-observed code-pair decoding", "gt_reconstruction_is_diagnostic": True}
        summary.update(embedding_forward_batch_size=batch_size, statistics_chunk_size=statistics_chunk,
                       saved_prediction_chunk_size=prediction_chunk)
        summary["artifact_sha256"] = {name: file_hash(stage/name) for name in
                                      ("code_pair_bank.json", "ridge_probe.npz", "metrics.parquet", "prediction_index.parquet")}
        (stage/"summary.json").write_text(json.dumps(summary, indent=2))
        stage.rename(output)
        return summary
    except Exception as error:
        (stage/"failure.json").write_text(json.dumps({"error": repr(error), "evaluation_id": evaluation_id}))
        raise RuntimeError(f"Endpoint evaluation failed; diagnostics: {stage/'failure.json'}") from error
