"""Validation-selected N and offline force-derived oracle routing; no model inference."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.endpoint_eval import MetricSums, build_prediction_index
from utils.phase_classifier import PHASES, PHASE_COLORS, identity

WINDOWS = (1, 3, 5, 7, 10, 20)


def select_policy(metrics):
    if any(r["split"] != "validation" for r in metrics):
        raise ValueError("Routing selection only accepts Validation metrics")
    policy = {"protocol": "offline force-derived oracle routing", "selection": "smallest N within 5% of minimum Validation RMSE",
              "minimum_anchors": 30, "minimum_episodes": 2, "methods": {}}
    for method in ("ridge", "vq_decoding"):
        rows = [r for r in metrics if r["method"] == method]
        groups = {phase: [r for r in rows if r["phase"] == phase] for phase in ("all", *PHASES)}
        def choose(values):
            if {int(r["N"]) for r in values} != set(WINDOWS) or len(values) != len(WINDOWS):
                raise ValueError(f"Need one metric for all six N: {method}")
            if len({(r["anchors"], r["episodes"]) for r in values}) != 1:
                raise ValueError("Routing compares inconsistent supports")
            best = min(values, key=lambda r: (r["endpoint_standardized_wrench_rmse"], r["N"]))
            selected = min(r["N"] for r in values if r["endpoint_standardized_wrench_rmse"] <= best["endpoint_standardized_wrench_rmse"]*1.05)
            return int(selected), int(best["N"])
        global_n, global_minimum = choose(groups["all"])
        decisions = {}
        for phase in PHASES:
            values = groups[phase]
            supported = values and min(r["anchors"] for r in values) >= 30 and min(r["episodes"] for r in values) >= 2
            minimum = choose(values)[1] if values else None
            eligible = bool(supported and phase != "unknown")
            decisions[phase] = {"selected_N": choose(values)[0] if eligible else global_n,
                                "minimum_error_N": minimum, "fallback": not eligible,
                                "reason": "Unknown uses global N" if phase == "unknown" else "phase support sufficient" if eligible else "insufficient phase support",
                                "anchors": values[0]["anchors"] if values else 0,
                                "episodes": values[0]["episodes"] if values else 0}
        policy["methods"][method] = {"global_N": global_n, "global_minimum_error_N": global_minimum, "phases": decisions}
    policy["policy_id"] = identity(policy)
    return policy


def iter_predictions(directory):
    for file in sorted(Path(directory).glob("part_*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            yield from batch.to_pylist()


def aggregate_and_route(evaluations: dict[int, Path], output: Path):
    if set(evaluations) != set(WINDOWS):
        raise ValueError(f"Need all N before routing; available: {sorted(evaluations)}")
    summaries = {n: json.loads((Path(path)/"summary.json").read_text()) for n, path in evaluations.items()}
    manifests = {s["manifest_id"] for s in summaries.values()}
    families = {s.get("experiment_id") for s in summaries.values()}
    if len(manifests) != 1 or len(families) != 1 or None in families:
        raise ValueError("Cannot aggregate different manifests or experiment configurations")
    for n, summary in summaries.items():
        if int(summary["N"]) != n:
            raise ValueError("Evaluation N does not match selected directory")
        if any(r["manifest_id"] != summary["manifest_id"] or r["checkpoint_id"] != summary["checkpoint_id"] or int(r["N"]) != n
               for r in summary["metrics"]):
            raise ValueError("Metric rows and checkpoint/manifest/N provenance differ")
    if len({s.get("normalization_id") for s in summaries.values()}) != 1 or len({s.get("split_id") for s in summaries.values()}) != 1:
        raise ValueError("N models use different Train normalization or split")
    metrics = [row for s in summaries.values() for row in s["metrics"]]
    policy = select_policy([r for r in metrics if r["split"] == "validation"])
    policy.update(manifest_id=next(iter(manifests)), experiment_id=next(iter(families)),
                  checkpoints={str(n): s["checkpoint_id"] for n, s in sorted(summaries.items())})
    policy.pop("policy_id"); policy["policy_id"] = identity(policy)
    output = Path(output)
    if output.exists():
        saved = json.loads((output/"policy.json").read_text())
        if saved != policy:
            raise FileExistsError("Routing policy/inputs changed; use a new analysis output")
        if (output/"summary.json").exists():
            return json.loads((output/"summary.json").read_text())
        raise FileExistsError("Incomplete routing output retained; select a new output path")
    output.mkdir(parents=True)
    # Freeze before reading any Test prediction rows.
    (output/"policy.json").write_text(json.dumps(policy, indent=2))
    route_metrics = []
    try:
        for split in ("validation", "test"):
            for method in ("ridge", "vq_decoding"):
                readers = [iter_predictions(Path(evaluations[n])/"predictions"/split/method) for n in WINDOWS]
                sums = {phase: MetricSums() for phase in ("all", *PHASES)}
                directory = output/"predictions"/split/method; directory.mkdir(parents=True)
                chunk = []; number = 0
                for rows in itertools.zip_longest(*readers):
                    if any(r is None for r in rows):
                        raise ValueError("N models have different endpoint counts")
                    base = rows[0]
                    for n, row in zip(WINDOWS, rows):
                        if (row["sample_id"], row["phase"], row["split"], row["manifest_id"], row["method"]) != (
                                base["sample_id"], base["phase"], split, base["manifest_id"], method):
                            raise ValueError("Routing endpoints/order/phase differ between N")
                        if row["checkpoint_id"] != summaries[n]["checkpoint_id"] or not np.array_equal(row["actual_wrench"], base["actual_wrench"]) or not np.array_equal(row["actual_pressure"], base["actual_pressure"]):
                            raise ValueError("Routing checkpoint or endpoint targets differ")
                    selected_n = policy["methods"][method]["phases"][base["phase"]]["selected_N"]
                    selected = dict(rows[WINDOWS.index(selected_n)])
                    selected.update(method=f"oracle_phase_routing/{method}", base_method=method, selected_N=selected_n,
                                    selected_checkpoint_id=selected["checkpoint_id"], checkpoint_id=policy["policy_id"], N=0,
                                    routing="offline force-derived oracle; not tactile-only online routing")
                    sums["all"].add(selected); sums[selected["phase"]].add(selected)
                    chunk.append(selected)
                    if len(chunk) == 128:
                        pq.write_table(pa.Table.from_pylist(chunk), directory/f"part_{number:09d}.parquet")
                        chunk = []; number += 1
                if chunk:
                    pq.write_table(pa.Table.from_pylist(chunk), directory/f"part_{number:09d}.parquet")
                if not sums["all"].n:
                    raise ValueError("No saved predictions for routing")
                for phase, value in sums.items():
                    route_metrics.append({"method": f"oracle_phase_routing/{method}", "split": split, "phase": phase,
                                          "manifest_id": policy["manifest_id"], "checkpoint_id": policy["policy_id"], "N": 0, **value.result()})
        all_metrics = metrics+route_metrics
        build_prediction_index(output.glob("predictions/*/*/part_*.parquet"), output)
        pq.write_table(pa.Table.from_pylist(all_metrics), output/"comparison.parquet")
        columns = ["method", "split", "manifest_id", "checkpoint_id", "N", "phase", "anchors", "episodes",
                   "endpoint_standardized_wrench_rmse", "endpoint_raw_pressure_rmse", "mean_selected_N"]
        with (output/"comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(all_metrics)
        _plot(all_metrics, output)
        result = {"policy": policy, "metrics": all_metrics, "mean_N_is_latency": False,
                  "interpretation": "single seed42 pilot; oracle phases use episode-wide and future force"}
        (output/"summary.json").write_text(json.dumps(result, indent=2))
        return result
    except Exception as error:
        (output/"failure.json").write_text(json.dumps({"error": repr(error)}))
        raise


def _plot(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split in ("validation", "test"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, phase in zip(axes.flat, ("all", *PHASES)):
            for method in ("ridge", "vq_decoding", "training_mean"):
                rows = sorted((r for r in metrics if r["split"] == split and r["phase"] == phase and r["method"] == method), key=lambda r:r["N"])
                ax.plot([r["N"] for r in rows], [r["endpoint_standardized_wrench_rmse"] for r in rows], marker="o", label=method)
            for method in ("ridge", "vq_decoding"):
                route = next((r for r in metrics if r["split"] == split and r["phase"] == phase and r["method"] == f"oracle_phase_routing/{method}"), None)
                if route and route["anchors"]:
                    ax.axhline(route["endpoint_standardized_wrench_rmse"], linestyle="--", label=f"oracle/{method}")
            ax.set(title=phase, xlabel="N (mean N is not latency)", ylabel="Endpoint standardized wrench RMSE"); ax.legend(fontsize=7)
        axes.flat[-1].axis("off"); fig.tight_layout(); fig.savefig(output/f"{split}_phase_comparison.png", dpi=140); plt.close(fig)


def save_retrieval_comparison(rows, output):
    """Gallery retrieval is reported separately from tactile-only force prediction."""
    output = Path(output); table = pa.Table.from_pylist(rows)
    target = output/"retrieval.parquet"
    if target.exists():
        if not pq.read_table(target).equals(table):
            raise FileExistsError("Saved retrieval aggregation differs; choose a new analysis output")
    else:
        pq.write_table(table, target)
    if not (output/"retrieval.csv").exists():
        with (output/"retrieval.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split in ("validation", "test"):
        if (output/f"{split}_retrieval_by_phase.png").exists():
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, phase in zip(axes.flat, ("all", *PHASES)):
            part = sorted((r for r in rows if r["split"] == split and r["phase"] == phase), key=lambda r:r["N"])
            for metric in ("recall_at_1", "recall_at_5", "mrr"):
                ax.plot([r["N"] for r in part], [r[metric] for r in part], marker="o", label=metric)
            ax.set(title=phase, xlabel="N", ylabel="Fixed-gallery retrieval"); ax.legend(fontsize=8)
        axes.flat[-1].axis("off"); fig.tight_layout(); fig.savefig(output/f"{split}_retrieval_by_phase.png", dpi=140); plt.close(fig)
