"""Four real anchors, forward/backward only; no optimizer updates or training run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.taf_dataset import TaFDataset
from utils.checkpoint import assert_tactile_checkpoint_compatible, load_checkpoint, save_checkpoint
from utils.common import set_seed, validate_config
from utils.config import register_config_schema
from utils.dynamics import ManifestDataset, split_group_key
from utils.model_factory import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/episode_split_validation"


def grad_norm(module):
    return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None) ** .5


def main():
    started = time.monotonic()
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only physical GPU 0 by its NVIDIA UUID")
    torch.set_num_threads(4); torch.cuda.set_device(0)
    register_config_schema()
    with initialize_config_dir(config_dir=str(ROOT/"config"), version_base="1.3"):
        cfg = compose(config_name="taf_adapter_sparsh_episode")
    validate_config(cfg, require_data_root=True)
    reference_path = ROOT/"outputs/comparison/sparsh_runs/06dc93e7_N5_seed42/last.pt"
    reference = load_checkpoint(reference_path)
    current = OmegaConf.to_container(cfg, resolve=True)
    allowed = {"data": {"split_unit"}, "comparison": {"manifest"},
               "training": {"output_dir"}, "wandb": {"name", "group"}}
    differences = {}
    for section, values in current.items():
        old = reference["config"][section]
        if isinstance(values, dict):
            for key in set(values) | set(old):
                if values.get(key) != old.get(key):
                    if key not in allowed.get(section, set()):
                        raise AssertionError(f"Unexpected resolved config change: {section}.{key}")
                    differences[f"{section}.{key}"] = [old.get(key), values.get(key)]
        elif values != old:
            raise AssertionError(f"Unexpected config section change: {section}")
    assert reference["step"] == 10000 and cfg.training.resume is None
    del reference
    manifest = ROOT/cfg.comparison.manifest
    source = TaFDataset(cfg.data.root_dir, window_size=1, image_size=None, image_dtype="uint8",
                        sensor_types=["custom_6x6_markers"], sensor_mmap_enabled=True,
                        sensor_mmap_dir=ROOT/cfg.data.sensor_mmap_dir, show_initialization_progress=False)
    dataset = ManifestDataset(source, manifest, 5, expected_split_unit="episode")
    normal = torch.load(manifest.with_name(f"normalization_{dataset.manifest_id[:8]}.pt"), map_location="cpu", weights_only=True)
    keys = {s: sorted({split_group_key(r, "episode") for r in dataset.rows if r["split"] == s})
            for s in ("train", "validation", "test")}
    assert normal["manifest_id"] == dataset.manifest_id and normal["train_groups"] == keys["train"]
    assert all(not set(keys[a]) & set(keys[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    train_sequences = {r["sequence_id"] for r in dataset.rows if r["split"] == "train"}
    val_sequences = {r["sequence_id"] for r in dataset.rows if r["split"] == "validation"}
    sequence = sorted(train_sequences & val_sequences)[0]
    selected = []
    for split in ("train", "validation"):
        candidates = [i for i, r in enumerate(dataset.rows) if r["sequence_id"] == sequence and r["split"] == split]
        selected.extend([candidates[len(candidates)//3], candidates[2*len(candidates)//3]])
    samples = [dataset[i] for i in selected]
    rows = [dataset.rows[i] for i in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(rows), OUT/"sample_selection.parquet")
    set_seed(int(cfg.training.seed))
    model = build_model(cfg, tactile_initialization_seed=int(cfg.training.seed)+10000).to("cuda:0")
    model.set_normalization(**normal["normalization"])
    model.train()
    batch = {key: torch.stack([sample[key] for sample in samples]).to("cuda:0") for key in ("tactile", "force", "pressure")}
    outputs = model(**batch)
    assert torch.isfinite(outputs["loss_total"])
    outputs["loss_total"].backward()
    encoder = model.tactile_encoder
    gradients = {name: grad_norm(module) for name, module in {
        "backbone": encoder.backbone, "pooling": encoder.spatial_pool, "projection": encoder.projection,
        "temporal": encoder.temporal, "force_mlp": model.force_encoder.force.encoder,
        "pressure_mlp": model.force_encoder.pressure.encoder,
        "force_codebook": model.force_encoder.force.quantizer,
        "pressure_codebook": model.force_encoder.pressure.quantizer}.items()}
    assert not encoder.backbone.training and gradients["backbone"] == 0
    assert all(value > 0 for name, value in gradients.items() if name != "backbone")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=bool(cfg.training.amp), dtype=torch.float16):
        validation = model(**{key: value[2:] for key, value in batch.items()})
        assert torch.isfinite(validation["loss_total"]) and torch.isfinite(validation["z_tactile"]).all()
    # Validate the new split metadata in a small step-zero checkpoint; never resume a sequence run.
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.training.learning_rate,
                                 weight_decay=cfg.training.weight_decay)
    plan_path = manifest.with_name("gallery_plan.json")
    identity = {"manifest_id": dataset.manifest_id, "split_unit": "episode",
                "gallery_id": hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    roundtrip = OUT/"fresh_step_zero.pt"
    save_checkpoint(roundtrip, model=model, optimizer=optimizer, scaler=torch.amp.GradScaler("cuda", enabled=False),
                    step=0, epoch=0, config=cfg, normalization=normal["normalization"],
                    train_keys=keys["train"], val_keys=keys["validation"], comparison_identity=identity)
    saved = load_checkpoint(roundtrip); assert_tactile_checkpoint_compatible(saved, cfg)
    assert saved["comparison_identity"] == identity
    assert saved["split"]["train_groups"] == keys["train"] and saved["split"]["validation_groups"] == keys["validation"]
    result = {"status": "passed", "gpu_uuid": uuid, "gpu_name": torch.cuda.get_device_name(0),
              "optimizer_updates": 0, "training_launched": False, "N": 5,
              "manifest_id": dataset.manifest_id, "reference_checkpoint": str(reference_path.relative_to(ROOT)),
              "reference_step": 10000, "resolved_config_changes": differences,
              "input_shape": list(batch["tactile"].shape), "embedding_shape": list(outputs["z_tactile"].shape),
              "loss_total": float(outputs["loss_total"].detach()), "amp_eval_finite": True,
              "gradients": gradients, "checkpoint_identity_verified": True,
              "train_validation_common_sequence": sequence,
              "selected": [{k: row[k] for k in ("sample_id", "split", "episode_index")} for row in rows],
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
              "elapsed_seconds": time.monotonic()-started}
    (OUT/"smoke.json").write_text(json.dumps(result, indent=2))
    (OUT/"resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
