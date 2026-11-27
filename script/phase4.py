"""Portable preparation, bounded evaluation and sequential N=1,3,5,7,10,20 runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.phase_classifier import identity
from utils.phase4_data import file_hash, load_prepared, load_settings, source_for_settings, assert_settings_prepared

ROOT = Path(__file__).resolve().parents[1]


def implementation_id():
    paths = ["dataset/taf_dataset.py", "model/taf_adapter.py", "model/force_encoder.py", "model/vector_quantizer.py",
             "model/tactile_encoder.py", "model/sparsh.py", "model/sparsh_backbone.py", "model/losses.py",
             "utils/data.py", "utils/dynamics.py", "utils/codebook_init.py", "utils/optimization.py",
             "utils/phase_classifier.py", "utils/phase4_data.py", "utils/endpoint_eval.py", "utils/phase_routing.py",
             "script/train_adapter.py", "script/phase4.py"]
    return identity({path: file_hash(ROOT/path) for path in paths})


def training_config(settings, n=5):
    from utils.config import register_config_schema
    from utils.common import validate_config
    register_config_schema()
    with initialize_config_dir(config_dir=str(ROOT/"config"), version_base="1.3"):
        cfg = compose(config_name=settings.training_config)
    cfg.data.root_dir = str(Path(settings.data_root).resolve())
    cfg.data.sensor_types = settings.sensors or None
    cfg.data.timestamp_cache_dir = settings.timestamp_cache_dir
    cfg.data.sensor_mmap_dir = settings.sensor_mmap_dir
    cfg.data.window_size = n; cfg.data.split_unit = "episode"; cfg.data.split_seed = settings.split_seed
    cfg.comparison.enabled = True; cfg.comparison.manifest = str(Path(settings.prepared_dir).resolve()/"manifest.parquet")
    cfg.comparison.training_seeds = [42]
    cfg.comparison.gallery_seed = settings.gallery_seed; cfg.comparison.gallery_max_pairs = settings.gallery_size
    cfg.comparison.min_anchor_gap = settings.gallery_gap
    cfg.training.seed = 42; cfg.training.resume = None
    validate_config(cfg, require_data_root=True)
    return cfg


def experiment(settings):
    prepared = load_prepared(settings.prepared_dir)
    assert_settings_prepared(settings, prepared)
    cfg = OmegaConf.to_container(training_config(settings), resolve=True)
    cfg["training"]["output_dir"] = "<per-N-output>"
    cfg["wandb"]["name"] = "<per-N-name>"
    recipe = {"protocol": "phase4-single-seed-v1", "manifest_id": prepared["manifest_id"],
              "implementation_id": implementation_id(),
              "model_training": cfg, "windows": settings.windows, "seed": 42,
              "eval_batch_size": settings.eval_batch_size, "ridge_alphas": settings.ridge_alphas}
    experiment_id = identity(recipe)
    output = Path(settings.results_dir)/prepared["manifest_id"][:8]/experiment_id[:12]
    return prepared, {**cfg, "implementation_id": recipe["implementation_id"]}, experiment_id, output


def _assert_checkpoint(checkpoint, cfg, manifest_id, *, require_finished=False):
    from utils.checkpoint import assert_tactile_checkpoint_compatible
    assert_tactile_checkpoint_compatible(checkpoint, cfg)
    saved = checkpoint.get("comparison_identity", {})
    if saved.get("manifest_id") != manifest_id or saved.get("window_size") != int(cfg.data.window_size):
        raise ValueError("Checkpoint manifest/N mismatch; never relabel or transplant an old checkpoint")
    expected = {"gallery_id": file_hash(Path(cfg.comparison.manifest).with_name("gallery_plan.json")),
                "batch_size": int(cfg.training.batch_size), "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
                "training_seed": int(cfg.training.seed), "split_seed": int(cfg.data.split_seed),
                "sampling_seed": int(cfg.comparison.sampling_seed), "gallery_seed": int(cfg.comparison.gallery_seed),
                "min_anchor_gap": int(cfg.comparison.min_anchor_gap)}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint gallery/batch/sampling identity mismatch")
    tags = [tag for tag in cfg.wandb.tags if tag.startswith("phase4-experiment:")]
    if tags and any(tag not in checkpoint["config"]["wandb"]["tags"] for tag in tags):
        raise ValueError("Checkpoint pipeline implementation/cohort identity mismatch")
    current = OmegaConf.to_container(cfg, resolve=True)
    for section in ("tactile_encoder", "force_encoder", "vq", "loss"):
        if checkpoint["config"][section] != current[section]:
            raise ValueError(f"Checkpoint configuration mismatch: {section}")
    for key in ("batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay", "max_steps", "amp", "seed", "batch_sampler", "scheduler"):
        if checkpoint["config"]["training"].get(key) != current["training"].get(key):
            raise ValueError(f"Checkpoint training mismatch: {key}")
    if require_finished and int(checkpoint["step"]) != int(cfg.training.max_steps):
        raise ValueError("Evaluation requires the fixed optimizer update budget to be complete")


def evaluate(settings, checkpoint_path, output, experiment_id, *, frozen_config=None):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from utils.checkpoint import load_checkpoint
    from utils.dynamics import ManifestDataset
    from utils.model_factory import build_model
    from utils.comparison_eval import evaluate_retrieval
    from utils.endpoint_eval import evaluate_endpoints
    prepared = load_prepared(settings.prepared_dir)
    assert_settings_prepared(settings, prepared)
    checkpoint_path = Path(checkpoint_path); checkpoint = load_checkpoint(checkpoint_path)
    n = int(checkpoint["config"]["data"]["window_size"])
    cfg = frozen_config if frozen_config is not None else training_config(settings, n)
    if frozen_config is None:
        cfg.wandb.tags = [*cfg.wandb.tags, f"phase4-experiment:{experiment_id[:40]}"]
    _assert_checkpoint(checkpoint, cfg, prepared["manifest_id"], require_finished=True)
    normal_path = Path(settings.prepared_dir)/prepared["normalization_file"]
    normal = torch.load(normal_path, weights_only=True, map_location="cpu")["normalization"]
    if any(not torch.equal(value, checkpoint["normalization"][key]) for key, value in normal.items()):
        raise ValueError("Checkpoint normalization does not match prepared Train statistics")
    source = source_for_settings(settings)
    from utils.phase4_data import raw_fingerprint
    if identity(raw_fingerprint(source)) != prepared["dataset_id"]:
        raise ValueError("Raw source changed since preparation")
    dataset = ManifestDataset(source, Path(settings.prepared_dir)/"manifest.parquet", n, expected_split_unit="episode")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose physical GPU 0 only for evaluation")
    device = torch.device("cuda:0")
    model = build_model(cfg); model.load_state_dict(checkpoint["model"], strict=True); model.to(device).eval()
    provenance = {"manifest_id": prepared["manifest_id"], "checkpoint_id": file_hash(checkpoint_path),
                  "normalization_id": file_hash(normal_path), "split_id": prepared["split_id"],
                  "experiment_id": experiment_id, "seed": 42, "checkpoint_step": int(checkpoint["step"])}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    binding = output/"identity.json"
    if binding.exists() and json.loads(binding.read_text()) != provenance:
        raise FileExistsError("Evaluation output belongs to a different checkpoint/configuration")
    binding.write_text(json.dumps(provenance, indent=2))
    gallery = json.loads((Path(settings.prepared_dir)/"gallery_plan.json").read_text())
    for split in ("validation", "test"):
        report_path = output/f"retrieval_{split}.json"
        if report_path.exists():
            if json.loads(report_path.read_text())["identity"] != provenance:
                raise ValueError("Retrieval identity changed")
            continue
        rows, report = evaluate_retrieval(model, dataset, gallery[split], device=device, batch_size=settings.eval_batch_size)
        for row in rows:
            row.update(**provenance, N=n, method="retrieval", split=split, phase=row["contact_state"])
        pq.write_table(pa.Table.from_pylist(rows), output/f"retrieval_{split}.parquet")
        report.update(identity=provenance, N=n, split=split, gallery_id=file_hash(Path(settings.prepared_dir)/"gallery_plan.json"))
        report_path.write_text(json.dumps(report, indent=2))
    endpoint = evaluate_endpoints(model, dataset, output/"endpoint", provenance, device=device,
                                  batch_size=settings.eval_batch_size, alphas=settings.ridge_alphas)
    (output/"summary.json").write_text(json.dumps({"identity": provenance, "N": n, "endpoint_evaluation_id": endpoint["evaluation_id"]}, indent=2))


def select_gpu_zero():
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return uuid


def _write_state(path, state):
    temporary = path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(state, indent=2)); temporary.replace(path)


def run_windows(settings, windows, *, dry_run=False):
    from utils.checkpoint import load_checkpoint
    prepared, recipe, experiment_id, output = experiment(settings)
    commands = {}
    for n in windows:
        base = output/f"N{n}_seed42"; training_output = base/"train"
        run_dir = training_output/f"{prepared['manifest_id'][:8]}_N{n}_seed42"
        cfg = training_config(settings, n)
        cfg.training.output_dir = str(training_output)
        cfg.wandb.name = f"Phase4-N{n}-seed42"; cfg.wandb.group = f"phase4-{experiment_id[:12]}"
        cfg.wandb.tags = [*cfg.wandb.tags, f"phase4-experiment:{experiment_id[:40]}"]
        train = ["uv", "run", "--no-sync", "python", "-m", "script.train_adapter",
                 f"--config-path={base.resolve()}", "--config-name=resolved_training"]
        command = ["uv", "run", "--no-sync", "python", "-m", "script.terminal_monitor", "--metrics", str(run_dir/"metrics.jsonl"),
                   "--total-steps", str(recipe["training"]["max_steps"]), "--", "uv", "run", "--no-sync", "trainctl", "run",
                   "--name", f"Phase4-N{n}-seed42", "--checkpoint-dir", str(run_dir),
                   "--total-steps", str(recipe["training"]["max_steps"]), "--", *train]
        commands[n] = (command, run_dir, base, cfg)
    if dry_run:
        print(json.dumps({"experiment_id": experiment_id, "output": str(output), "commands": {n:c[0] for n,c in commands.items()}}, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    # Two pipelines must never start concurrent GPU work. No other user's process is controlled.
    lock_path = ROOT/"outputs/phase4/gpu0.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        select_gpu_zero()
        from utils.phase4_data import raw_fingerprint
        source = source_for_settings(settings)
        if identity(raw_fingerprint(source)) != prepared["dataset_id"]:
            raise ValueError("Raw source changed since preparation; training was not started")
        del source
        state_path = output/"state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"experiment_id": experiment_id,
                        "implementation_id": recipe["implementation_id"], "manifest_id": prepared["manifest_id"], "N": {}, "failures": []}
        if state["experiment_id"] != experiment_id or state["manifest_id"] != prepared["manifest_id"]:
            raise ValueError("Pipeline ledger identity mismatch")
        for n, (command, run_dir, base, cfg) in commands.items():
            record = state["N"].setdefault(str(n), {})
            active_stage = "training"
            try:
                if implementation_id() != recipe["implementation_id"]:
                    raise ValueError("Pipeline implementation changed during execution; select a new experiment cohort")
                base.mkdir(parents=True, exist_ok=True)
                snapshot = base/"resolved_training.yaml"
                content = "defaults:\n  - taf_adapter_schema\n  - _self_\n" + OmegaConf.to_yaml(cfg, resolve=True)
                if snapshot.exists() and snapshot.read_text() != content:
                    raise ValueError("Frozen per-N training configuration changed")
                snapshot.write_text(content)
                checkpoints = sorted(run_dir.glob("step_*.pt"))
                if (run_dir/"last.pt").exists():
                    last = load_checkpoint(run_dir/"last.pt")
                    latest_numbered = int(checkpoints[-1].stem.split("_")[-1]) if checkpoints else -1
                    if int(last["step"]) >= latest_numbered:
                        checkpoints.append(run_dir/"last.pt")
                    del last
                checkpoint_path = None; saved_step = 0
                if checkpoints:
                    checkpoint_path = checkpoints[-1]; checkpoint = load_checkpoint(checkpoint_path)
                    _assert_checkpoint(checkpoint, cfg, prepared["manifest_id"])
                    saved_step = int(checkpoint["step"]); del checkpoint
                if saved_step < int(cfg.training.max_steps):
                    if checkpoint_path:
                        command = [*command, f"training.resume={checkpoint_path}"]
                    elif run_dir.exists() and any(run_dir.iterdir()):
                        raise FileExistsError("Incomplete training has no resumable checkpoint; preserve it and choose a new results_dir")
                    record["training"] = "running"; _write_state(state_path, state)
                    subprocess.run(command, cwd=ROOT, check=True)
                    checkpoint_path = run_dir/"last.pt"
                    saved = load_checkpoint(checkpoint_path); _assert_checkpoint(saved, cfg, prepared["manifest_id"], require_finished=True); del saved
                checkpoint_id = file_hash(checkpoint_path)
                if record.get("checkpoint_id") is not None and record["checkpoint_id"] != checkpoint_id:
                    raise ValueError("Completed checkpoint bytes changed; preserve old evaluation and choose a new results_dir")
                record.update(training="complete", checkpoint=str(checkpoint_path), checkpoint_id=checkpoint_id)
                _write_state(state_path, state)
                active_stage = "evaluation"
                if record.get("evaluation") == "complete":
                    saved_evaluation = json.loads((base/"evaluation/summary.json").read_text())
                    if (saved_evaluation["identity"]["checkpoint_id"] != checkpoint_id or
                            saved_evaluation["identity"]["manifest_id"] != prepared["manifest_id"] or
                            saved_evaluation["identity"]["experiment_id"] != experiment_id):
                        raise ValueError("Completed evaluation identity differs from checkpoint/cohort")
                    endpoint_dir = base/"evaluation/endpoint"
                    endpoint_report = json.loads((endpoint_dir/"summary.json").read_text())
                    if endpoint_report["checkpoint_id"] != checkpoint_id or endpoint_report["manifest_id"] != prepared["manifest_id"]:
                        raise ValueError("Endpoint results belong to another checkpoint/manifest")
                    if any(file_hash(endpoint_dir/name) != digest for name, digest in endpoint_report["artifact_sha256"].items()):
                        raise ValueError("Completed endpoint evaluation artifacts changed")
                if record.get("evaluation") != "complete":
                    record["evaluation"] = "running"; _write_state(state_path, state)
                    evaluate(settings, checkpoint_path, base/"evaluation", experiment_id, frozen_config=cfg)
                    record["evaluation"] = "complete"; _write_state(state_path, state)
            except Exception as error:
                record[active_stage] = "failed"; record["failed_stage"] = active_stage
                record["error"] = repr(error)
                state["failures"].append({"N": n, "error": repr(error), "time": time.time()})
                _write_state(state_path, state)
                print(f"N={n} FAILED: {error}", file=sys.stderr)
        if all(state["N"].get(str(n), {}).get("evaluation") == "complete" for n in settings.windows):
            state["analysis"] = "running"; _write_state(state_path, state)
            try:
                aggregate(settings)
                state["analysis"] = "complete"; _write_state(state_path, state)
            except Exception as error:
                state["analysis"] = "failed"
                state["failures"].append({"stage": "analysis", "error": repr(error), "time": time.time()})
                _write_state(state_path, state)
                raise
        elif any(state["N"][str(n)].get("evaluation") != "complete" for n in windows):
            raise RuntimeError(f"Some stages failed; completed stages retained: {state_path}")


def aggregate(settings, output_path=None):
    from utils.phase_routing import aggregate_and_route, save_retrieval_comparison
    prepared, recipe, experiment_id, output = experiment(settings)
    evaluations = {n: output/f"N{n}_seed42/evaluation/endpoint" for n in settings.windows}
    if any(json.loads((path/"summary.json").read_text())["experiment_id"] != experiment_id for path in evaluations.values()):
        raise ValueError("Evaluation results belong to another experiment cohort")
    analysis = Path(output_path) if output_path is not None else output/"analysis"
    result = aggregate_and_route(evaluations, analysis)
    # Retrieval uses the same fixed gallery for every N and four endpoint phases.
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    for n in settings.windows:
        for split in ("validation", "test"):
            report = json.loads((output/f"N{n}_seed42/evaluation/retrieval_{split}.json").read_text())
            rows.extend({**report["identity"], "N": n, "method": "retrieval", "split": split, "phase": phase, **metrics}
                        for phase, metrics in report["groups"].items())
    save_retrieval_comparison(rows, analysis)
    print(f"Saved analysis: {analysis}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "evaluate", "aggregate", "ui"])
    parser.add_argument("--config", default="config/phase4_pipeline.yaml")
    parser.add_argument("--set", action="append", default=[], help="Pipeline setting key=value; schema checked")
    parser.add_argument("--n", type=int, nargs="+", choices=[1, 3, 5, 7, 10, 20])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint"); parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.n is not None and args.command != "run":
        parser.error("--n applies only to run")
    if args.dry_run and args.command != "run":
        parser.error("--dry-run applies only to run")
    if args.checkpoint is not None and args.command != "evaluate":
        parser.error("--checkpoint applies only to evaluate")
    if args.output is not None and args.command not in {"evaluate", "aggregate"}:
        parser.error("--output applies only to evaluate/aggregate; set prepared_dir/results_dir in pipeline settings")
    settings = load_settings(args.config, args.set)
    if args.command == "prepare":
        from utils.phase4_data import prepare_from_settings
        report = prepare_from_settings(settings)
        print(json.dumps({"prepared_dir": settings.prepared_dir, **{key: report[key] for key in
                         ("manifest_id", "dataset_id", "episodes", "anchor_support", "exclusions", "timestamp_diagnostics", "reference_source")}}, indent=2))
    elif args.command == "run":
        run_windows(settings, args.n or settings.windows, dry_run=args.dry_run)
    elif args.command == "aggregate":
        aggregate(settings, args.output)
    elif args.command == "ui":
        subprocess.run(["uv", "run", "--no-sync", "python", "-m", "streamlit", "run", "app.py", "--", "--config", args.config,
                        *sum((["--set", item] for item in args.set), [])], cwd=ROOT, check=True)
    else:
        if not args.checkpoint or not args.output:
            parser.error("evaluate requires --checkpoint and --output")
        select_gpu_zero(); _, _, experiment_id, _ = experiment(settings)
        evaluate(settings, args.checkpoint, args.output, experiment_id)


if __name__ == "__main__":
    main()
