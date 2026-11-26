"""Bounded synthetic bundle and UI smoke; no GPU/training/Discord/W&B calls."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from tests.test_phase4 import tiny_model, write_lerobot_fixture
from utils.dynamics import ManifestDataset
from utils.endpoint_eval import evaluate_endpoints
from utils.phase4_data import PhasePipelineConfig, prepare_phase4, source_for_settings
from utils.phase_routing import aggregate_and_route, save_retrieval_comparison


def main():
    started = time.monotonic(); torch.set_num_threads(2)
    parent = Path("outputs/phase4_validation"); parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="synthetic_", dir=parent)).resolve()
    raw = output/"raw"; raw.mkdir(); write_lerobot_fixture(raw)
    cfg = PhasePipelineConfig(data_root=str(raw), sensors=["sensor"], split_reference=None,
                              prepared_dir=str(output/"prepared"), results_dir=str(output/"results"), gallery_size=4,
                              timestamp_cache_dir=str(output/"timestamp_cache"), sensor_mmap_dir=str(output/"sensor_mmap"))
    source = source_for_settings(cfg)
    with patch.object(source, "_images", side_effect=AssertionError("No decoder during prepare")):
        prepared = prepare_phase4(source, cfg)
    config_path = output/"pipeline.yaml"; OmegaConf.save(OmegaConf.create(asdict(cfg)), config_path)
    normal = torch.load(Path(cfg.prepared_dir)/prepared["normalization_file"], weights_only=True)["normalization"]
    paths = {}; retrieval_metrics = []
    from utils.comparison_eval import evaluate_retrieval
    import pyarrow as pa
    import pyarrow.parquet as pq
    gallery = json.loads((Path(cfg.prepared_dir)/"gallery_plan.json").read_text())
    for n in cfg.windows:
        torch.manual_seed(42)
        model = tiny_model(n).eval(); model.set_normalization(**normal)
        dataset = ManifestDataset(source, Path(cfg.prepared_dir)/"manifest.parquet", n)
        paths[n] = output/f"results/cohort/N{n}_seed42/evaluation/endpoint"
        provenance = {"manifest_id": prepared["manifest_id"], "checkpoint_id": f"fresh-synthetic-{n}",
                      "normalization_id": "synthetic-train-normalization", "split_id": prepared["split_id"], "experiment_id": "synthetic"}
        evaluate_endpoints(model, dataset, paths[n], provenance, batch_size=32, alphas=(.1,1.))
        for split in ("validation", "test"):
            rows, report = evaluate_retrieval(model, dataset, gallery[split], device="cpu", batch_size=2)
            for row in rows:
                row.update(**provenance, N=n, method="retrieval", split=split, phase=row["contact_state"])
            pq.write_table(pa.Table.from_pylist(rows), paths[n].parent/f"retrieval_{split}.parquet")
            report.update(identity=provenance, N=n, split=split)
            retrieval_metrics.extend({**provenance, "N": n, "method": "retrieval", "split": split,
                                      "phase": phase, **metrics} for phase,metrics in report["groups"].items())
            (paths[n].parent/f"retrieval_{split}.json").write_text(json.dumps(report, indent=2))
    aggregate_and_route(paths, output/"results/cohort/analysis")
    save_retrieval_comparison(retrieval_metrics, output/"results/cohort/analysis")
    # Exercise actual Streamlit pages against this local bundle, never reclassifying in UI.
    from streamlit.testing.v1 import AppTest
    import sys
    pages = []
    with patch.object(sys, "argv", ["app.py", "--config", str(config_path)]):
        app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=10).run()
        if app.exception or app.error:
            raise AssertionError(f"Explorer UI failed: {app.exception or app.error}")
        pages.append("Dataset Explorer")
        for page in ("Phase Inspector", "Dataset Statistics", "Model Analysis"):
            app.sidebar.radio[0].set_value(page).run()
            if app.exception or app.error:
                raise AssertionError(f"{page} failed: {app.exception or app.error}")
            pages.append(page)
    result = {"status": "passed", "scope": "synthetic CPU fixture; not real model performance",
              "prepared": cfg.prepared_dir, "manifest_id": prepared["manifest_id"], "windows": cfg.windows,
              "pages": pages, "optimizer_updates": 0, "gpu_used": False,
              "tracking_runs_created": False, "elapsed_seconds": time.monotonic()-started}
    result["production_loader_and_ui_unmocked"] = True
    (output/"smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({**result, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
