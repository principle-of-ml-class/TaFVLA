"""Real official-weight Sparsh smoke checks, GPU 0 only, zero optimizer updates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.taf_dataset import TaFDataset
from utils.alignment import evaluate_alignment
from utils.checkpoint import save_checkpoint, load_checkpoint, assert_tactile_checkpoint_compatible
from utils.common import set_seed, validate_config
from utils.config import register_config_schema
from utils.dynamics import ManifestDataset
from utils.model_factory import build_model
from utils.trainctl import TrainctlReporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/sparsh_validation"


def norm(module):
    return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None) ** .5


def run():
    started = time.time()
    row = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True).strip()
    uuid = row.split(",")[0].strip()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only physical GPU 0 by its NVIDIA UUID in CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    register_config_schema()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        cfg = compose(config_name="taf_adapter_sparsh")
    validate_config(cfg)
    (OUT / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(42)
    reporter = TrainctlReporter()
    reporter.start(name="SparshIntegrationSmoke", total_steps=1, message="공식 Sparsh Small 통합 smoke test: GPU 0, optimizer update 없음")
    normal = torch.load(ROOT / "metadata/normalization_06dc93e7.pt", map_location="cpu", weights_only=True)
    results = {"gpu_uuid": uuid, "gpu": row, "optimizer_updates": 0, "synthetic": []}
    for n in [1, 5, 20]:
        model = build_model(cfg, window_size=n).to("cuda:0")
        model.set_normalization(**normal["normalization"])
        model.train()
        assert not model.tactile_encoder.backbone.training
        assert all(not p.requires_grad for p in model.tactile_encoder.backbone.parameters())
        shapes = []
        hook = model.tactile_encoder.spatial_pool.register_forward_pre_hook(lambda module, args: shapes.append(list(args[0].shape)))
        batch_size = 4 if n == 5 else 2
        with torch.no_grad():
            outputs = model(torch.randint(0, 256, (batch_size,n,3,96,128), device="cuda:0", dtype=torch.uint8),
                            torch.randn(batch_size,n,6,device="cuda:0"), torch.randn(batch_size,n,12,12,device="cuda:0"))
        hook.remove()
        assert outputs["z_tactile"].shape == (batch_size, 256)
        assert shapes == [[batch_size*n,300,384]]
        assert torch.isfinite(outputs["z_tactile"]).all() and torch.isfinite(outputs["loss_total"])
        results["synthetic"].append({"N":n,"input":[batch_size,n,3,96,128],"pool_input":shapes,
                                      "embedding":list(outputs["z_tactile"].shape),"loss":float(outputs["loss_total"])})
        del model, outputs
        torch.cuda.empty_cache()
    # Read only four existing training anchors; never rebuild manifest/statistics.
    table = pq.read_table(ROOT / "metadata/manifest.parquet", filters=[("split", "=", "train")])
    candidates = np.random.default_rng(42).permutation(len(table))
    chosen = []; sequences = set()
    for i in candidates:
        sequence = table["sequence_id"][int(i)].as_py()
        if sequence not in sequences:
            chosen.append(int(i)); sequences.add(sequence)
        if len(chosen) == 4:
            break
    selected = table.take(pa.array(sorted(chosen)))
    pq.write_table(selected, OUT / "sample_selection.parquet")
    source = TaFDataset(cfg.data.root_dir, window_size=1, image_size=None, image_dtype="uint8",
                        sensor_types=["custom_6x6_markers"], show_initialization_progress=False,
                        sensor_mmap_enabled=True, sensor_mmap_dir=ROOT / "outputs/sensor_mmap",
                        video_cache_max_entries=2)
    dataset = ManifestDataset(source, OUT / "sample_selection.parquet", 5)
    samples = [dataset[i] for i in range(4)]
    batch = {k:torch.stack([s[k] for s in samples]).to("cuda:0") for k in ["tactile","force","pressure"]}
    model = build_model(cfg).to("cuda:0")
    model.set_normalization(**normal["normalization"])
    model.train()
    outputs = model(**batch)
    books = [model.force_encoder.pressure.quantizer.codebook.weight, model.force_encoder.force.quantizer.codebook.weight]
    routed = torch.autograd.grad(outputs["loss_nce"], books, allow_unused=True, retain_graph=True)
    assert all(g is None for g in routed)
    outputs["loss_total"].backward()
    encoder = model.tactile_encoder
    gradients = {name:norm(module) for name,module in {
        "backbone":encoder.backbone,"pooling":encoder.spatial_pool,"projection":encoder.projection,
        "temporal":encoder.temporal,"position":encoder.position,"force_mlp":model.force_encoder.force.encoder,
        "pressure_mlp":model.force_encoder.pressure.encoder,"force_codebook":model.force_encoder.force.quantizer,
        "pressure_codebook":model.force_encoder.pressure.quantizer,
        "force_decoder":model.force_encoder.force.decoder,"pressure_decoder":model.force_encoder.pressure.decoder}.items()}
    assert gradients["backbone"] == 0
    assert all(v > 0 for k,v in gradients.items() if k != "backbone")
    assert encoder.summary_token.grad.abs().sum() > 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        amp = model(**batch)
        assert torch.isfinite(amp["loss_total"]) and torch.isfinite(amp["z_tactile"]).all()
    results.update(real_input=list(batch["tactile"].shape), real_loss=float(outputs["loss_total"].detach()),
                   gradients=gradients, amp_finite=True, checkpoint_identity=encoder.checkpoint_identity(),
                   parameter_groups={name:sum(p.numel() for p in module.parameters()) for name,module in {
                       "backbone":encoder.backbone,"pooling":encoder.spatial_pool,"projection":encoder.projection,
                       "temporal":encoder.temporal,"force_encoder":model.force_encoder}.items()},
                   frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    path = OUT / "roundtrip.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scaler=torch.amp.GradScaler("cuda",enabled=False),
                    step=0,epoch=0,config=cfg,normalization=normal["normalization"],
                    train_keys=normal["train_groups"],val_keys=[])
    saved = load_checkpoint(path)
    assert_tactile_checkpoint_compatible(saved,cfg)
    reloaded = build_model(cfg).to("cuda:0")
    reloaded.load_state_dict(saved["model"],strict=True)
    model.eval();reloaded.eval()
    with torch.no_grad():
        a=model.encode_tactile(batch["tactile"]);b=reloaded.encode_tactile(batch["tactile"])
    assert torch.equal(a,b)
    results["checkpoint_roundtrip"] = {"strict":True,"embedding_max_delta":float((a-b).abs().max())}
    del reloaded, saved
    old_checkpoint = load_checkpoint(ROOT / "outputs/comparison/runs/06dc93e7_N5_seed42/last.pt")
    try:
        assert_tactile_checkpoint_compatible(old_checkpoint,cfg)
    except ValueError:
        results["imagenet_resume_blocked"] = True
    else:
        raise AssertionError("ImageNet resume was not blocked")
    del old_checkpoint
    alignment = evaluate_alignment(model, DataLoader(samples,batch_size=2), torch.device("cuda:0"))
    results["alignment_smoke"] = alignment.metrics
    model.zero_grad(set_to_none=True)
    fine_cfg = OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    fine_cfg.tactile_encoder.freeze_backbone = False
    fine = build_model(fine_cfg).to("cuda:0")
    fine.set_normalization(**normal["normalization"])
    fine.train()
    small = {k:v[:2] for k,v in batch.items()}
    fine(**small)["loss_total"].backward()
    results["unfrozen_backbone_gradient_norm"] = norm(fine.tactile_encoder.backbone)
    assert fine.tactile_encoder.backbone.training and results["unfrozen_backbone_gradient_norm"] > 0
    results.update(status="passed",elapsed_seconds=time.time()-started,
                   peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    (OUT / "smoke.json").write_text(json.dumps(results,indent=2))
    reporter.finish(f"Sparsh official Small smoke PASS | N=1,5,20 | real batch4 | freeze/unfreeze/AMP/checkpoint/retrieval | optimizer updates=0 | {OUT}")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    run()
