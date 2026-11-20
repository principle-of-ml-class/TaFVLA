"""Compare measured TaFVLA training performance for N=1,3,5,7,10,20."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

WINDOWS = (1, 3, 5, 7, 10, 20)


def _hardware_key(hardware: dict) -> tuple:
    return (hardware.get("host"), hardware.get("machine"),
            hardware.get("processor"), hardware.get("device"), hardware.get("gpu"))


def collect(paths: list[Path]) -> tuple[list[dict], list[int]]:
    runs = []
    hardware_keys = set()
    protocol_keys = set()
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        sessions = [row for row in records if row.get("record_type") != "offline_evaluation"]
        if not sessions:
            raise ValueError(f"No training performance summary in {path}")
        first = sessions[0]
        if int(first["window_size"]) not in WINDOWS:
            raise ValueError(f"Unexpected temporal window N={first['window_size']} in {path}")
        if any((row["window_size"], row["seed"]) !=
               (first["window_size"], first["seed"]) for row in sessions):
            raise ValueError(f"Mixed N/seed in {path}")
        hardware = first["hardware"]
        if any(_hardware_key(row["hardware"]) != _hardware_key(hardware) for row in sessions):
            raise ValueError(f"Mixed hardware within {path}")
        hardware_keys.add(_hardware_key(hardware))
        protocol_keys.add(tuple(json.dumps(first.get(key), sort_keys=True) for key in (
            "batch_size", "gradient_accumulation_steps", "num_workers",
            "prefetch_factor", "amp", "feature_cache_enabled", "split_unit", "loss_mode",
            "comparison_mode", "manifest_id", "target_steps", "split_seed",
            "stride", "image_size", "backbone"
        )))
        active = sum(float(row.get("training_active_seconds") or 0) for row in sessions)
        completed = sum(int(row["final_step"]) - int(row["initial_step"]) for row in sessions)
        frames = sum((row.get("average_frames_per_second") or 0) *
                     (row.get("training_active_seconds") or 0) for row in sessions)
        totals = defaultdict(float)
        for row in sessions:
            for name, seconds in row.get("evaluation_total_seconds", {}).items():
                totals[name] += float(seconds)
        last = sessions[-1]
        offline = [row for row in records if row.get("record_type") == "offline_evaluation"
                   and row.get("hardware", {}).get("host") == hardware.get("host")
                   and row.get("hardware", {}).get("gpu") == hardware.get("gpu")]
        test_retrieval = [row for row in offline if row["evaluation"] == "retrieval"
                          and row.get("split") == "test"]
        force = [row for row in offline if row["evaluation"] == "probe"]
        parquet_hits = sum(row.get("parquet_cache", {}).get("cache_hits", 0) for row in sessions)
        parquet_misses = sum(row.get("parquet_cache", {}).get("cache_misses", 0) for row in sessions)
        feature_hits = sum(row.get("feature_cache", {}).get("hits", 0) for row in sessions)
        feature_misses = sum(row.get("feature_cache", {}).get("misses", 0) for row in sessions)
        runs.append({
            "N": int(last["window_size"]), "seed": int(last["seed"]),
            "train_windows": int(last["train_windows"]),
            "steps_per_second": completed / active if active else None,
            "frames_per_second": frames / active if active else None,
            "gpu_peak_allocated_gib": max((row.get("peak_gpu_allocated_gib") or 0)
                                           for row in sessions) or None,
            "gpu_peak_reserved_gib": max((row.get("peak_gpu_reserved_gib") or 0)
                                          for row in sessions) or None,
            "cpu_peak_rss_gib": max((row.get("peak_cpu_rss_gib") or 0)
                                    for row in sessions) or None,
            "total_training_seconds": float(last["training_duration_seconds"]),
            "validation_seconds": totals.get("validation"),
            "retrieval_seconds": (float(test_retrieval[-1]["duration_seconds"])
                                  if test_retrieval else totals.get("alignment_retrieval",
                                                                      totals.get("retrieval"))),
            "retrieval_source": ("offline_test" if test_retrieval else
                                 "training_alignment" if "alignment_retrieval" in totals
                                 or "retrieval" in totals else None),
            "force_prediction_seconds": (float(force[-1]["duration_seconds"])
                                         if force else None),
            "checkpoint_seconds": totals.get("checkpoint"),
            "parquet_cache_hit_rate": (parquet_hits / (parquet_hits + parquet_misses)
                                       if parquet_hits + parquet_misses else None),
            "feature_cache_hit_rate": (feature_hits / (feature_hits + feature_misses)
                                       if feature_hits + feature_misses else None),
            "hardware": hardware, "summary_path": str(path),
        })
    if len(hardware_keys) > 1:
        raise ValueError("Runs used different hardware; compare each hardware group separately")
    if len(protocol_keys) > 1:
        raise ValueError("Runs used different training protocols; compare each protocol separately")
    return sorted(runs, key=lambda x: (x["N"], x["seed"])), sorted(set(WINDOWS) - {x["N"] for x in runs})


def write_comparison(rows: list[dict], missing: list[int], output: Path) -> list[str]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "performance_by_N.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "hardware": json.dumps(row["hardware"], sort_keys=True)})
    (output / "performance_by_N.json").write_text(
        json.dumps({"runs": rows, "missing_N": missing}, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    graphs = []
    for name, ylabel, filename in (
        ("steps_per_second", "Optimizer steps / second", "steps_per_second_by_N.png"),
        ("frames_per_second", "Tactile frames / second", "frames_per_second_by_N.png"),
        ("gpu_peak_allocated_gib", "GPU peak allocated (GiB)", "gpu_memory_by_N.png"),
        ("total_training_seconds", "Total training time (seconds)", "total_time_by_N.png"),
    ):
        if not any(row[name] is not None for row in rows):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for seed in sorted({row["seed"] for row in rows}):
            points = [row for row in rows if row["seed"] == seed and row[name] is not None]
            if points:
                ax.plot([row["N"] for row in points], [row[name] for row in points],
                        marker="o", label=f"seed {seed}")
        ax.set(xlabel="Temporal window N", ylabel=ylabel, xticks=WINDOWS)
        ax.grid(True, alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        graphs.append(filename)
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, missing = collect(args.summaries)
    graphs = write_comparison(rows, missing, args.output_dir)
    print("N | Seed | Steps/s | Frames/s | GPU Peak GiB | CPU Peak GiB | Total Time s")
    print("--- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        value = lambda name: "N/A" if row[name] is None else f"{row[name]:.3f}"
        print(f"{row['N']} | {row['seed']} | {value('steps_per_second')} | "
              f"{value('frames_per_second')} | {value('gpu_peak_allocated_gib')} | "
              f"{value('cpu_peak_rss_gib')} | {value('total_training_seconds')}")
    print(json.dumps({"missing_N": missing, "graphs": graphs, "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
