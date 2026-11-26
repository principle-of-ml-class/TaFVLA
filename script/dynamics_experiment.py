"""Manifest, fixed-gallery retrieval, tactile force probe and seed summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from dataset.taf_dataset import TaFDataset
from utils.checkpoint import load_checkpoint
from utils.comparison_eval import (evaluate_probe, evaluate_retrieval, save_gallery_plan,
                                   write_rows)
from utils.dynamics import ManifestDataset, create_manifest, DYNAMICS
from utils.episode_split import repartition_episode_manifest, prepare_episode_assets
from utils.model_factory import build_model
from utils.logging import ExperimentLogger


def source_from_config(root, data):
    return TaFDataset(root, window_size=1, stride=1, image_size=tuple(data.image_size) if data.image_size is not None else None,
                      sensor_types=data.sensor_types, max_sequences=data.max_sequences,
                      skip_invalid_sequences=bool(data.skip_invalid_sequences),
                      timestamp_cache_enabled=bool(data.timestamp_cache_enabled),
                      timestamp_cache_dir=data.timestamp_cache_dir,
                      show_initialization_progress=bool(data.show_initialization_progress))


def load_run(args):
    checkpoint = load_checkpoint(Path(args.checkpoint))
    identity = checkpoint.get("comparison_identity")
    if identity is None:
        raise ValueError("Checkpoint was not trained in comparison mode")
    config = OmegaConf.create(checkpoint["config"])
    manifest = Path(args.manifest or config.comparison.manifest)
    root = args.data_root or config.data.root_dir
    source = source_from_config(root, config.data)
    dataset = ManifestDataset(source, manifest, int(identity["window_size"]),
                              expected_split_unit=str(config.data.split_unit))
    if dataset.manifest_id != identity["manifest_id"]:
        raise ValueError("Checkpoint and manifest identities differ")
    model = build_model(config, pretrained=False if config.tactile_encoder.get("backbone_type", "imagenet") == "imagenet" else None)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    run_id = f"{identity['manifest_id'][:8]}_N{identity['window_size']}_seed{identity['training_seed']}"
    return model, dataset, identity, config, run_id, device, checkpoint


def base_rows(rows, identity, run_id, split, config):
    for row in rows:
        row.update(model_run_id=run_id, model_window_size=identity["window_size"],
                   training_seed=identity["training_seed"], split_seed=identity["split_seed"],
                   manifest_id=identity["manifest_id"], label_sha256=identity["label_sha256"],
                   gallery_seed=identity["gallery_seed"], gallery_id=identity["gallery_id"],
                   sampling_seed=identity["sampling_seed"],
                   min_anchor_gap=identity["min_anchor_gap"],
                   forward_batch_size=identity["batch_size"], split=split,
                   learning_rate=float(config.training.learning_rate),
                   temperature=float(config.loss.temperature), loss_mode=str(config.loss.nce_mode))
    return rows


def aggregate(paths, output):
    import csv
    import math
    import statistics
    from collections import defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = []
    for path in paths:
        if str(path).endswith(".parquet"):
            data.extend(pq.read_table(path).to_pylist())
        else:
            with Path(path).open(newline="", encoding="utf-8") as stream:
                data.extend(csv.DictReader(stream))
    data = [r for r in data if r.get("split") == "test"]
    if not data:
        raise ValueError("No test rows to aggregate")
    metrics = [x for x in ("recall_at_1", "recall_at_5", "mrr", "force_prediction_mae",
                           "force_prediction_rmse", "endpoint_mae", "endpoint_rmse",
                           "force_mae", "force_rmse", "torque_mae", "torque_rmse",
                           "endpoint_force_mae", "endpoint_force_rmse",
                           "endpoint_torque_mae", "endpoint_torque_rmse",
                           "training_duration", "retrieval_inference_latency",
                           "probe_inference_latency") if any(r.get(x) is not None for r in data)]
    grouped = defaultdict(list)
    for row in data:
        grouped[(row["contact_state"], int(row["model_window_size"]), int(row["training_seed"]))].append(row)
    per_seed = defaultdict(list)
    for (group, n, seed), rows in grouped.items():
        measures = {}
        for metric in metrics:
            values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
            if values:
                measures[metric] = math.sqrt(sum(x*x for x in values)/len(values)) if metric.endswith("rmse") else statistics.mean(values)
        per_seed[(group, n)].append(measures)
    summary = []
    for group in DYNAMICS:
        for n in (1, 3, 5, 7, 10, 20):
            rows = [r for r in data if r["contact_state"] == group and int(r["model_window_size"]) == n]
            retrieval_rows = [r for r in rows if r.get("evaluation_type") == "retrieval"]
            probe_rows = [r for r in rows if r.get("evaluation_type") == "probe"]
            item = {"contact_state": group, "model_window_size": n, "available": bool(rows),
                    "retrieval_available": bool(retrieval_rows),
                    "probe_available": bool(probe_rows),
                    "queries": len({r["sample_id"] for r in retrieval_rows}),
                    "sequences": len({r["sequence_id"] for r in retrieval_rows}),
                    "episodes": len({(r["sequence_id"], r["episode_index"]) for r in retrieval_rows}),
                    "probe_samples": len({r["sample_id"] for r in probe_rows}),
                    "probe_sequences": len({r["sequence_id"] for r in probe_rows}),
                    "probe_episodes": len({(r["sequence_id"], r["episode_index"]) for r in probe_rows}),
                    "seed_count": len(per_seed[(group, n)])}
            for metric in metrics:
                values = [x[metric] for x in per_seed[(group, n)] if metric in x]
                item[f"{metric}_mean"] = statistics.mean(values) if values else None
                item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(item)
    for row in summary:
        for metric in metrics:
            baseline = next(x[f"{metric}_mean"] for x in summary if x["contact_state"] == row["contact_state"] and x["model_window_size"] == 1)
            current = row[f"{metric}_mean"]
            row[f"{metric}_delta_vs_N1"] = current-baseline if current is not None and baseline is not None else None
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dynamics_by_N.csv").exists():
        raise FileExistsError(f"Summary already exists: {output / 'dynamics_by_N.csv'}")
    with (output / "dynamics_by_N.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(summary), output / "dynamics_by_N.parquet")
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for group in DYNAMICS:
            part = [r for r in summary if r["contact_state"] == group and r[f"{metric}_mean"] is not None]
            if part:
                ax.errorbar([r["model_window_size"] for r in part], [r[f"{metric}_mean"] for r in part],
                            yerr=[r[f"{metric}_std"] or 0 for r in part], marker="o", label=group)
        ax.set(xlabel="Model window N", ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}_by_N.png", dpi=150)
        plt.close(fig)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith("phase4-"):
        from script.phase4 import main as phase4_main
        return phase4_main([sys.argv[1].removeprefix("phase4-"), *sys.argv[2:]])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--labels", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--split-seed", type=int, default=42)
    manifest.add_argument("--fractions", type=float, nargs=3, default=(.8, .1, .1))
    manifest.add_argument("--timestamp-tolerance", type=float, default=.001)
    manifest.add_argument("--label-metadata")
    manifest.add_argument("--max-sequences", type=int)
    manifest.add_argument("--gallery-seed", type=int, default=42)
    manifest.add_argument("--gallery-max-pairs", type=int, default=256)
    manifest.add_argument("--gallery-min-gap", type=int, default=20)
    episode = sub.add_parser("episode-split", help="Repartition existing anchors and prepare train-only statistics")
    episode.add_argument("--manifest", default="metadata/manifest.parquet")
    episode.add_argument("--output", default="metadata/episode_split/manifest.parquet")
    episode.add_argument("--split-seed", type=int, default=42)
    episode.add_argument("--original-gallery")
    episode.add_argument("--config-name", default="taf_adapter_sparsh_episode")
    for command in ("retrieval", "probe"):
        item = sub.add_parser(command)
        item.add_argument("--checkpoint", required=True)
        item.add_argument("--manifest")
        item.add_argument("--data-root")
        item.add_argument("--output", required=True)
        item.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        item.add_argument("--batch-size", type=int, default=16)
        if command == "retrieval":
            item.add_argument("--split", choices=("validation", "test"), default="test")
            item.add_argument("--gallery-plan")
    summary = sub.add_parser("aggregate")
    summary.add_argument("--results", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "episode-split":
        report = repartition_episode_manifest(Path(args.manifest), Path(args.output), seed=args.split_seed)
        assets = prepare_episode_assets(Path(args.output), config_name=args.config_name,
                                       original_gallery=Path(args.original_gallery) if args.original_gallery else
                                       Path(args.manifest).with_name("gallery_plan.json"))
        print(json.dumps({k: v for k, v in report.items() if k != "episode_assignments"}, indent=2))
        print(json.dumps({k: v for k, v in assets.items() if k != "normalization_values"}, indent=2))
    elif args.command == "manifest":
        source = TaFDataset(args.data_root, window_size=1, stride=1, max_sequences=args.max_sequences)
        report = create_manifest(source, Path(args.labels), Path(args.output), split_seed=args.split_seed,
                                 fractions=tuple(args.fractions), timestamp_tolerance=args.timestamp_tolerance,
                                 metadata_path=Path(args.label_metadata) if args.label_metadata else None)
        rows = pq.read_table(args.output).to_pylist()
        plan_path = Path(args.output).with_name("gallery_plan.json")
        plan = save_gallery_plan(rows, plan_path, seed=args.gallery_seed,
                                 max_pairs=args.gallery_max_pairs, min_gap=args.gallery_min_gap)
        print(json.dumps({**report, "gallery_plan": str(plan_path),
                          "gallery_support": plan["support"]}, indent=2))
    elif args.command == "aggregate":
        aggregate(args.results, Path(args.output))
    else:
        model, dataset, identity, config, run_id, device, checkpoint = load_run(args)
        output = Path(args.output)
        rows_by_id = {r["sample_id"]: r for r in dataset.rows}
        training_duration = checkpoint.get("training_duration_seconds")
        metrics_path = Path(args.checkpoint).parent / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                elapsed = item.get("train/elapsed_time_seconds", item.get("seconds"))
                if elapsed is not None and training_duration is None:
                    training_duration = max(training_duration or 0.0, float(elapsed))
        if args.command == "retrieval":
            plan_path = Path(args.gallery_plan or Path(args.manifest or config.comparison.manifest).with_name("gallery_plan.json"))
            plan = json.loads(plan_path.read_text())
            if (plan["manifest_id"] != identity["manifest_id"] or
                plan["seed"] != identity["gallery_seed"] or
                hashlib.sha256(plan_path.read_bytes()).hexdigest() != identity["gallery_id"]):
                raise ValueError("Gallery plan identity/seed differs from checkpoint")
            ids = plan[args.split]
            if len(ids) < 2:
                raise ValueError("Retrieval requires at least two common gallery samples")
            evaluation_start = time.perf_counter()
            rows, report = evaluate_retrieval(model, dataset, ids, device=device, batch_size=args.batch_size)
            evaluation_seconds = time.perf_counter() - evaluation_start
            base_rows(rows, identity, run_id, args.split, config)
            for row in rows:
                match = rows_by_id[row["sample_id"]]
                row.update(training_duration=training_duration,
                           force_torque_unit_metadata=match.get("force_torque_unit_metadata"),
                           gallery_size=len(ids),
                           inference_latency=report["diagnostics"]["inference_seconds"]/len(ids),
                           retrieval_inference_latency=report["diagnostics"]["inference_seconds"]/len(ids),
                           evaluation_type="retrieval")
            report.update(identity=identity, gallery_plan=str(plan_path), split=args.split,
                          measurement={"device": str(device), "batch_size": args.batch_size,
                                       "includes_decode": True, "timing": "wall clock"})
        else:
            evaluation_start = time.perf_counter()
            rows, report = evaluate_probe(model, dataset, device=device, batch_size=args.batch_size)
            evaluation_seconds = time.perf_counter() - evaluation_start
            for split in ("validation", "test"):
                base_rows([r for r in rows if r["split"] == split], identity, run_id, split, config)
            split_counts = Counter(r["split"] for r in rows)
            for row in rows:
                match = rows_by_id[row["sample_id"]]
                latency = report["inference_seconds_by_split"][row["split"]]/split_counts[row["split"]]
                row.update(training_duration=training_duration,
                           force_torque_unit_metadata=match.get("force_torque_unit_metadata"),
                           probe_alpha=report["alpha"], probe_parameters=report["probe_parameters"],
                           inference_latency=latency,
                           probe_inference_latency=latency,
                           evaluation_type="probe")
            report.update(identity=identity, evaluation="tactile_only_ridge_probe",
                          measurement={"device": str(device), "batch_size": args.batch_size,
                                       "includes_decode": True, "timing": "wall clock"})
        performance_key = ("perf/retrieval_seconds" if args.command == "retrieval"
                           else "perf/force_prediction_seconds")
        report["performance"] = {performance_key: evaluation_seconds}
        write_rows(rows, output)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary_path = Path(args.checkpoint).parent / "performance_summary.jsonl"
        if summary_path.is_file():
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "record_type": "offline_evaluation", "window_size": int(identity["window_size"]),
                    "seed": int(identity["training_seed"]), "evaluation": args.command,
                    "split": args.split if args.command == "retrieval" else None,
                    "duration_seconds": evaluation_seconds, "report": str(output.with_suffix(".json")),
                    "hardware": {"host": platform.node(), "device": str(device),
                                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                }, sort_keys=True) + "\n")
        if bool(config.wandb.enabled):
            tracking = ExperimentLogger(Path(args.checkpoint).parent, config.wandb,
                                        checkpoint.get("wandb"))
            try:
                tracking.start(config, model)
                tracking.log_wandb({performance_key: evaluation_seconds},
                                   int(checkpoint["step"]), allow_same_step=True)
            except Exception:
                warnings.warn("Offline evaluation performance could not be logged to W&B",
                              RuntimeWarning)
            finally:
                tracking.finish(exit_code=0)
        print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
