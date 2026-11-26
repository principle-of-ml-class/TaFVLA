"""Bounded real-data smoke check; no full training or experiment overwrite."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.taf_dataset import TaFDataset
from utils.codebook_init import initialize_codebooks, sample_train_windows
from utils.common import set_seed, validate_config
from utils.config import register_config_schema
from utils.dynamics import ManifestDataset, split_group_key
from utils.model_factory import build_model
from utils.optimization import UpdateLRScheduler, gradient_metrics, optimizer_update


def _timeout(*_):
    raise TimeoutError("Smoke check exceeded 28 seconds")


def main():
    # Fail promptly if a cold dataset/cache makes this more than a lightweight check.
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(28)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    physical_zero = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_zero or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only physical GPU 0 using its NVIDIA UUID")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    register_config_schema()
    with initialize_config_dir(config_dir=str(root/"config"), version_base="1.3"):
        cfg = compose(config_name="taf_adapter_sparsh_episode_stable")
    validate_config(cfg, require_data_root=True)
    set_seed(int(cfg.training.seed))
    print("[smoke] loading existing manifest and normalization", flush=True)
    source = TaFDataset(cfg.data.root_dir, window_size=1, image_size=None, image_dtype="uint8",
                        sensor_types=["custom_6x6_markers"], show_initialization_progress=False,
                        sensor_mmap_enabled=True, sensor_mmap_dir=root/cfg.data.sensor_mmap_dir)
    manifest = root/cfg.comparison.manifest
    dataset = ManifestDataset(source, manifest, 5, expected_split_unit="episode")
    train = [i for i, row in enumerate(dataset.rows) if row["split"] == "train"]
    # Two sequences avoid validating every large sensor cache in a bounded smoke check.
    sequences = sorted({dataset.rows[i]["sequence_id"] for i in train})[:2]
    smoke_train = [i for i in train if dataset.rows[i]["sequence_id"] in sequences]
    normalization = torch.load(manifest.with_name(f"normalization_{dataset.manifest_id[:8]}.pt"),
                               map_location="cpu", weights_only=True)
    assert normalization["manifest_id"] == dataset.manifest_id
    assert normalization["train_groups"] == sorted({split_group_key(dataset.rows[i], "episode") for i in train})
    model = build_model(cfg, tactile_initialization_seed=int(cfg.training.seed)+10000).to("cuda:0")
    model.set_normalization(**normalization["normalization"])
    # Keep the production 512x128 codebooks; shrink only this validation pool/iterations.
    init_options = dict(OmegaConf.to_container(cfg.vq))
    init_options.update(kmeans_samples=1024, kmeans_iters=3)
    print(f"[smoke] setup {time.monotonic()-started:.2f}s; sensor-only K-means: 1024 windows, 2 sequences, 3 iterations", flush=True)
    with patch.object(source, "_images", side_effect=AssertionError("No image decoding during initialization")), \
         patch.object(model.tactile_encoder, "forward", side_effect=AssertionError("No Sparsh during initialization")):
        initialization = initialize_codebooks(model, dataset, smoke_train, init_options)
    assert initialization["sample_count"] == 1024
    for name in ("pressure", "force"):
        book = getattr(model.force_encoder, name).quantizer.codebook.weight
        assert tuple(book.shape) == (512, 128) and torch.isfinite(book).all()
        assert initialization[name]["quantization_distance_after"] < initialization[name]["quantization_distance_before"]
    saved_books = [getattr(model.force_encoder, name).quantizer.codebook.weight.detach().clone()
                   for name in ("pressure", "force")]
    with patch("utils.codebook_init.sample_train_windows", side_effect=AssertionError("Resume must not resample")):
        initialize_codebooks(model, dataset, train, init_options, resume=True)
    for saved, name in zip(saved_books, ("pressure", "force"), strict=True):
        torch.testing.assert_close(saved, getattr(model.force_encoder, name).quantizer.codebook.weight, rtol=0, atol=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = UpdateLRScheduler(optimizer, cfg.training.scheduler, cfg.training.max_steps, cfg.training.learning_rate)
    # Lower validation-only scale avoids a random first-step overflow obscuring backward checks.
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    chosen, _ = sample_train_windows(dataset, smoke_train, 4, 42)
    losses = []
    model.train()
    print("[smoke] two real minibatches, forward/backward with AMP and accumulation=2", flush=True)
    for offset in (0, 2):
        samples = [dataset[i] for i in chosen[offset:offset+2]]
        batch = {key: torch.stack([s[key] for s in samples]).to("cuda:0") for key in ("tactile", "force", "pressure")}
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**batch)
        assert torch.isfinite(outputs["loss_total"])
        scaler.scale(outputs["loss_total"]/2).backward()
        losses.append(float(outputs["loss_total"].detach()))
        assert scheduler.completed_updates == 0
    scaler.unscale_(optimizer)
    metrics = gradient_metrics(model, cfg.training.grad_clip_norm)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert float(metrics["train/gradient_norm_after_clipping"]) <= cfg.training.grad_clip_norm+1e-5
    assert all(p.grad is None for p in model.tactile_encoder.backbone.parameters())
    assert any(p.grad is not None for p in model.force_encoder.pressure.quantizer.parameters())
    updated = optimizer_update(optimizer, scaler, scheduler)
    assert updated and scheduler.completed_updates == 1
    # Real CUDA GradScaler overflow check, isolated from the trained model.
    tiny = torch.nn.Linear(2, 1).cuda()
    tiny_opt = torch.optim.SGD(tiny.parameters(), lr=1e-4)
    tiny_schedule = UpdateLRScheduler(tiny_opt, cfg.training.scheduler, 10000, 1e-4)
    tiny_scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    tiny_scaler.scale(tiny(torch.ones(2, 2, device="cuda")).sum()).backward()
    next(tiny.parameters()).grad.fill_(float("inf"))
    tiny_scaler.unscale_(tiny_opt)
    before = tiny.weight.detach().clone()
    assert not optimizer_update(tiny_opt, tiny_scaler, tiny_schedule)
    torch.testing.assert_close(before, tiny.weight, rtol=0, atol=0)
    assert tiny_schedule.completed_updates == 0
    assert tiny_opt.param_groups[0]["lr"] == tiny_schedule.lr_for_update(1)
    result = {"status": "passed", "gpu_uuid": physical_zero, "gpu": torch.cuda.get_device_name(0),
              "manifest_id": dataset.manifest_id, "initialization": initialization,
              "smoke_overrides": {"kmeans_samples": 1024, "kmeans_iters": 3, "sequences": sequences, "batch_size": 2,
                                  "gradient_accumulation_steps": 2, "amp_initial_scale": 128},
              "minibatches": 2, "optimizer_updates": 1, "losses": losses,
              "gradients": {key: float(value) for key, value in metrics.items()},
              "scheduler_lr": {str(n): scheduler.lr_for_update(n) for n in (1, 500, 10000)},
              "cuda_amp_skip_verified": True, "resume_no_reinitialization_verified": True,
              "training_launched": False, "elapsed_seconds": time.monotonic()-started}
    parent = root/"outputs/stability_validation"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="smoke_", dir=parent))
    (out/"smoke.json").write_text(json.dumps(result, indent=2))
    (out/"resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    signal.alarm(0)
    print(json.dumps({"status": "passed", "output": str(out), "elapsed_seconds": result["elapsed_seconds"],
                      "initialization_seconds": initialization["elapsed_seconds"],
                      "distances": {name: [initialization[name][f"quantization_distance_{when}"] for when in ("before", "after")]
                                    for name in ("pressure", "force")}}, indent=2))


if __name__ == "__main__":
    main()
